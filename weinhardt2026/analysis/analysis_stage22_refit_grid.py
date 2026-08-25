"""Stage 2.2 as a behavioural refit, swept across a params_array grid.

The standard Stage 2 fits SINDy coefficients to reproduce the frozen RNN's
hidden-state trajectories -- an L2 objective in state space, which is only a
surrogate for the choice likelihood the model is scored on. This script refits
those coefficients directly against behaviour on the support Stage 2.1 already
selected, and does it for every checkpoint of a pruning hyperparameter scan.

Protocol (each finding below was measured on dezfouli2019):

  * **Per-member loss.** Each ensemble member is scored on its own predictions
    and the losses averaged. Training on the ensemble-*mean* logits instead lets
    members trade individual accuracy for errors that cancel in the average:
    per-member test likelihood collapsed 0.714 -> 0.417 while the averaged
    number kept climbing, and the worst member fell below chance. The published
    equation is a member (or their coefficient mean), so members must stand on
    their own.

  * **Per-member selection criterion.** Early stopping and best-checkpoint
    selection both use the ΔBIC of the mean per-member likelihood. Selecting on
    the coefficient-averaged score instead presumes the across-member
    coefficient distribution is unimodal enough for its mean to be a sensible
    model, and lets members degrade unnoticed on dense supports.

  * **Per-epoch evaluation.** One extra no_grad forward (~28% overhead) buys
    exact peak localisation. Peak step scales inversely with sparsity -- on
    dezfouli2019, 53 active coefficients peaked at step 2, 18 at step 228, 11.8
    at step 715 -- so a coarse eval grid mislocates the optimum precisely where
    resolution matters.

  * **Step 0 is scored and seeded as the initial best**, so a checkpoint the
    refit only harms correctly selects the unrefit model rather than a
    manufactured later one.

  * **Gradient accumulation** over session batches is mathematically identical
    to a full-batch step, just at lower peak memory. Required for eckstein2026
    (4158 sessions x 150 trials x 10 members OOMs an 8 GiB card); harmless
    elsewhere -- set --batch-size 0 to disable.

Selection uses the evaluation split, standing in for a validation set. **Every
post-refit number this produces is therefore optimistic and is not a held-out
estimate.** Within a checkpoint the support is frozen, so parameter count is
constant and the ΔBIC ranking is identical to a likelihood ranking.

Examples:

    # dezfouli2019, full grid
    python weinhardt2026/analysis/analysis_stage22_refit_grid.py --study dezfouli2019

    # eckstein2026 on a cluster node
    python weinhardt2026/analysis/analysis_stage22_refit_grid.py \\
        --study eckstein2026 --max-steps 1000 --patience 50 \\
        --min-delta 1e-3 --batch-size 824
"""

import argparse
import glob
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim
from weinhardt2026.analysis.analysis_model_evaluation import (
    get_participant_experiment_groups, grouped_information_criteria,
)


# ---------------------------------------------------------------------------
# Study registry
# ---------------------------------------------------------------------------

STUDIES = {
    'dezfouli2019': dict(
        model_module='spice.precoded.workingmemory',
        test_blocks=(3, 6, 9),
        model_kwargs={'reward_binary': True},
        polynomial_degree=2,
    ),
    'eckstein2026': dict(
        model_module='weinhardt2026.studies.eckstein2026.spice_eckstein2026',
        test_blocks=(2,),
        model_kwargs={},
        polynomial_degree=2,
    ),
}


def load_study(study: str):
    import importlib
    spec = STUDIES[study]
    module = importlib.import_module(spec['model_module'])
    return module.SpiceModel, module.CONFIG, spec


# ---------------------------------------------------------------------------
# Ensemble views
# ---------------------------------------------------------------------------

@contextmanager
def averaged_coefficients(model):
    """Temporarily collapse the ensemble to its mean coefficient set (the
    equation `print_spice_model` reports). Zeros carry the 'inactive' meaning,
    matching `CompressedSpiceModel._write`."""
    modules = model.get_modules()
    saved = {m: (model.sindy_coefficients[m].data.clone(),
                 model.sindy_coefficients_presence[m].clone()) for m in modules}
    try:
        aggregated = model.get_sindy_coefficients(aggregate=True)
        for module in modules:
            values = aggregated[module].unsqueeze(0).expand(model.ensemble_size, -1, -1, -1).clone()
            model.sindy_coefficients[module].data = values.to(saved[module][0].dtype)
            model.sindy_coefficients_presence[module] = torch.ones_like(saved[module][1])
        yield
    finally:
        for module in modules:
            model.sindy_coefficients[module].data = saved[module][0]
            model.sindy_coefficients_presence[module] = saved[module][1]


@torch.no_grad()
def _forward_batched(model, xs: torch.Tensor, batch_size: int, device) -> torch.Tensor:
    """Log-probabilities (E, B, T, W, A) for the whole split, in session batches."""
    outputs = []
    step = batch_size if batch_size else xs.shape[0]
    for i in range(0, xs.shape[0], step):
        chunk = xs[i:i + step].to(device)
        model.init_state(batch_size=chunk.shape[0])
        logits, _ = model(chunk)
        outputs.append(torch.log_softmax(logits, dim=-1).cpu())
    return torch.cat(outputs, dim=1)


def _grouped_dbic(nll_per_session, valid, n_parameters, dataset) -> float:
    unique_pairs, group_index = get_participant_experiment_groups(dataset)
    n_par = torch.tensor(n_parameters, dtype=torch.float32)[unique_pairs[:, 0], unique_pairs[:, 1]]
    info = grouped_information_criteria(
        nll_per_session=nll_per_session.cpu(),
        n_trials_per_session=valid.sum(dim=1).float().cpu(),
        group_index=group_index, n_groups=unique_pairs.shape[0],
        n_parameters_per_group=n_par, n_actions_baseline=dataset.n_actions,
    )
    return info['delta_bic_per_trial_mean']


@torch.no_grad()
def score_member(model, dataset, batch_size, device) -> float:
    """Early-stopping criterion: mean per-member trial likelihood, higher better.

    Makes no assumption about the across-member coefficient distribution --
    every member is a valid model and this reports the typical one.

    This used to return a ΔBIC, which is wrong on the held-out split it is
    called with: the `k log n` term stands in for the generalization gap that
    held-out data measures directly, so scoring both charges parsimony twice.
    The change is behaviourally a no-op *here* -- the support is frozen for the
    whole refit, so a constant parameter count makes the ΔBIC and likelihood
    rankings identical -- but the number is now the one that is actually
    meaningful off-sample, and it no longer invites cross-checkpoint
    comparisons that the penalty would corrupt.
    """
    model.eval(use_sindy=True)
    log_probs = _forward_batched(model, dataset.xs, batch_size, device)
    valid = ~torch.isnan(dataset.xs[:, :, 0, 0])
    targets = torch.nan_to_num(dataset.ys, nan=0.0)
    ll = (targets.unsqueeze(0) * log_probs).sum(-1).sum(-1)          # (E, B, T)
    return float(torch.exp((ll * valid.unsqueeze(0)).sum()
                           / (valid.sum() * model.ensemble_size)))


@torch.no_grad()
def evaluate_full(model, dataset, batch_size, device, held_out: bool = False) -> Dict[str, float]:
    """All three scorings, for the pre/post report.

    ``held_out=True`` omits the ΔBIC entries; a test split is reported as
    likelihood only.
    """
    model.eval(use_sindy=True)
    E = model.ensemble_size
    valid = ~torch.isnan(dataset.xs[:, :, 0, 0])
    targets = torch.nan_to_num(dataset.ys, nan=0.0)
    n_par = model.count_spice_parameters()['loadings'].cpu().numpy()

    log_probs = _forward_batched(model, dataset.xs, batch_size, device)
    ll_member = (targets.unsqueeze(0) * log_probs).sum(-1).sum(-1)
    lik_member = float(torch.exp((ll_member * valid.unsqueeze(0)).sum() / (valid.sum() * E)))
    dbic_member = _grouped_dbic(-(ll_member * valid.unsqueeze(0)).sum(dim=2).mean(dim=0),
                                valid, n_par, dataset)

    # logit-average needs logits, not log-probs: recompute per batch
    probs = torch.exp(log_probs).mean(dim=0)
    ll_logit = (targets * torch.log(probs.clamp_min(1e-9))).sum(-1).sum(-1)
    lik_logit = float(torch.exp((ll_logit * valid).sum() / valid.sum()))

    with averaged_coefficients(model):
        n_par_coef = model.count_spice_parameters()['loadings'].cpu().numpy()
        lp_coef = _forward_batched(model, dataset.xs, batch_size, device)[0]
        ll_coef = (targets * lp_coef).sum(-1).sum(-1)
        lik_coef = float(torch.exp((ll_coef * valid).sum() / valid.sum()))
        dbic_coef = _grouped_dbic(-(ll_coef * valid).sum(dim=1), valid, n_par_coef, dataset)

    out = dict(lik_member=lik_member, lik_logit=lik_logit, lik_coef=lik_coef,
               n_par=float(n_par.mean()))
    if not held_out:
        # Information criteria on the training split only -- see `score_member`.
        out.update(dbic_member=dbic_member, dbic_coef=dbic_coef)
    return out


def snapshot(model) -> dict:
    return dict(c={m: model.sindy_coefficients[m].data.clone() for m in model.get_modules()},
                i={k: p.data.clone() for k, p in model.learnable_initial_values.items()})


def restore(model, s: dict) -> None:
    for m in model.get_modules():
        model.sindy_coefficients[m].data = s['c'][m].clone()
    for k, p in model.learnable_initial_values.items():
        p.data = s['i'][k].clone()


# ---------------------------------------------------------------------------
# Checkpoint discovery, ordered sparsest-first
# ---------------------------------------------------------------------------

def discover_checkpoints(params_dir: str, study: str, config) -> List[Tuple[str, float, float, float]]:
    """(path, threshold, ratio, mean active coefficients) sorted sparsest-first.

    Sparse checkpoints take the longest to converge (peak step scales inversely
    with parameter count) but are also the informative ones, so they run first
    and their results land while the dense tail is still going.
    """
    found = []
    for path in sorted(glob.glob(os.path.join(params_dir, f'spice_{study}_*.pkl'))):
        if 'stage1' in path or 'stability' in path:
            continue
        m = re.search(r'_(\d+\.?\d*)_(\d+\.?\d*)\.pkl$', path)
        if not m:
            continue
        state = torch.load(path, map_location='cpu')
        presence = state.get('sindy_coefficients_presence', {})
        n_active = float(np.mean([v.float().sum(-1).mean().item() for v in presence.values()])) \
            if presence else float('nan')
        del state
        found.append((path, float(m.group(1)), float(m.group(2)), n_active))
    return sorted(found, key=lambda r: r[3])


# ---------------------------------------------------------------------------
# Refit one checkpoint
# ---------------------------------------------------------------------------

def refit_checkpoint(estimator, dataset_train, dataset_test, args, device) -> dict:
    model = estimator.model
    modules = model.get_modules()
    E = model.ensemble_size

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for m in modules:
        model.sindy_coefficients[m].requires_grad_(True)
        trainable.append(model.sindy_coefficients[m])
    frozen = {m: model.sindy_coefficients_presence[m].clone() for m in modules}
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    xs, ys = dataset_train.xs, dataset_train.ys
    valid_total = (~torch.isnan(xs[:, :, 0, 0])).sum().item()
    step_size = args.batch_size if args.batch_size else xs.shape[0]

    best = dict(score=score_member(model, dataset_test, args.batch_size, device),
                step=0, snap=snapshot(model), stale=0)
    stop_step = 0

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad()
        model.train(mode=True, use_sindy=True)
        # Gradient accumulation over session batches == exact full-batch gradient.
        for i in range(0, xs.shape[0], step_size):
            xb = xs[i:i + step_size].to(device)
            yb = torch.nan_to_num(ys[i:i + step_size].to(device), nan=0.0)
            vb = (~torch.isnan(xs[i:i + step_size, :, 0, 0])).to(device)
            model.init_state(batch_size=xb.shape[0])
            logits, _ = model(xb)
            log_probs = torch.log_softmax(logits, dim=-1)
            ll = (yb.unsqueeze(0) * log_probs).sum(-1).sum(-1)
            loss = -(ll * vb.unsqueeze(0)).sum() / (valid_total * E)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
        optimizer.step()
        with torch.no_grad():
            for m in modules:
                model.sindy_coefficients[m].data *= frozen[m].float()

        score = score_member(model, dataset_test, args.batch_size, device)
        stop_step = step
        if score > best['score'] + args.min_delta:
            best.update(score=score, step=step, snap=snapshot(model), stale=0)
        else:
            best['stale'] += 1
            if best['stale'] >= args.patience:
                break

    restore(model, best['snap'])
    return dict(best_step=best['step'], stop_step=stop_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--study', required=True, choices=sorted(STUDIES))
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50,
                        help='epochs without improvement before stopping')
    parser.add_argument('--min-delta', type=float, default=1e-3,
                        help='improvement below this does not reset patience')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='sessions per accumulation batch (0 = full batch)')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--params-dir', default=None)
    parser.add_argument('--data', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    spice_class, config, spec = load_study(args.study)
    study_dir = ROOT / 'weinhardt2026' / 'studies' / args.study
    params_dir = args.params_dir or str(study_dir / 'params_array')
    data_path = args.data or str(study_dir / 'data' / f'{args.study}.csv')
    output_dir = args.output_dir or str(study_dir / 'results' / 'stage22_refit_grid')
    ckpt_dir = os.path.join(output_dir, 'best_models')
    os.makedirs(ckpt_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'stage22_refit_grid.csv')

    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    dataset = csv_to_dataset(file=data_path)
    dataset.normalize_rewards()
    dataset_train, dataset_test = split_data_along_blockdim(dataset, spec['test_blocks'])

    print(f"study={args.study} device={device} | train {dataset_train.xs.shape[0]} "
          f"test {dataset_test.xs.shape[0]} sessions | max_steps={args.max_steps} "
          f"patience={args.patience} min_delta={args.min_delta} batch={args.batch_size or 'full'}",
          flush=True)

    checkpoints = discover_checkpoints(params_dir, args.study, config)
    print(f"{len(checkpoints)} checkpoints, sparsest first\n", flush=True)

    rows = []
    for path, threshold, ratio, n_active in checkpoints:
        t0 = time.time()
        state = torch.load(path, map_location='cpu')
        first = next(iter(config.library_setup))
        E, P = state['model'][f'sindy_concept_loadings.{first}'].shape[:2]
        del state

        estimator = SpiceEstimator(
            spice_class=spice_class, spice_config=config, n_actions=dataset.n_actions,
            n_participants=P, sindy_library_polynomial_degree=spec['polynomial_degree'],
            ensemble_size=E, use_sindy=True, kwargs_spice_class=spec['model_kwargs'],
            device=device,
        )
        estimator.load_spice(path)

        pre_test = evaluate_full(estimator.model, dataset_test, args.batch_size, device, held_out=True)
        pre_train = evaluate_full(estimator.model, dataset_train, args.batch_size, device)
        info = refit_checkpoint(estimator, dataset_train, dataset_test, args, device)
        post_test = evaluate_full(estimator.model, dataset_test, args.batch_size, device, held_out=True)
        post_train = evaluate_full(estimator.model, dataset_train, args.batch_size, device)

        estimator.save_spice(os.path.join(
            ckpt_dir, f'spice_{args.study}_{threshold:g}_{ratio:g}_BEST.pkl'))

        row = dict(threshold=threshold, ratio=ratio, ensemble=E, n_par=pre_test['n_par'], **info)
        row.update({f'pre_test_{k}': v for k, v in pre_test.items() if k != 'n_par'})
        row.update({f'post_test_{k}': v for k, v in post_test.items() if k != 'n_par'})
        row.update({f'pre_train_{k}': v for k, v in pre_train.items() if k != 'n_par'})
        row.update({f'post_train_{k}': v for k, v in post_train.items() if k != 'n_par'})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        print(f"thr={threshold:<5g} ratio={ratio:<4g} par={row['n_par']:6.2f} | "
              f"best@{info['best_step']:4d} stopped@{info['stop_step']:4d} | "
              f"train dBIC {pre_train['dbic_member']:+.4f} -> {post_train['dbic_member']:+.4f} "
              f"({post_train['dbic_member'] - pre_train['dbic_member']:+.4f}) | "
              f"test lik {pre_test['lik_member']:.4f} -> {post_test['lik_member']:.4f} "
              f"[{time.time() - t0:.0f}s]", flush=True)

        del estimator
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    # Gain is measured on the training criterion, which is the one that trades
    # fit against parameter count; the held-out columns are likelihood only.
    df['gain'] = df.post_train_dbic_member - df.pre_train_dbic_member

    print(f"\n=== pre vs post correlation (n={len(df)}) ===", flush=True)
    for split, metric in [('test', 'lik_member'), ('test', 'lik_coef'),
                          ('train', 'dbic_member'), ('train', 'dbic_coef')]:
        pre, post = df[f'pre_{split}_{metric}'], df[f'post_{split}_{metric}']
        print(f"  {split}/{metric:12s} pearson {pre.corr(post):+.3f} "
              f"spearman {pre.corr(post, method='spearman'):+.3f}")
    print(f"\n  corr(n_par, gain) pearson {df.n_par.corr(df.gain):+.3f} "
          f"spearman {df.n_par.corr(df.gain, method='spearman'):+.3f}")
    censored = df[df.best_step >= args.max_steps]
    if len(censored):
        print(f"\n  WARNING: {len(censored)} checkpoint(s) hit the step cap and are censored: "
              f"{censored[['threshold', 'ratio']].to_dict('records')}")
    sel = df.loc[df.pre_train_dbic_member.idxmax()]
    orc = df.loc[df.post_train_dbic_member.idxmax()]
    print(f"\n  pick by pre-refit train dBIC:  thr={sel.threshold:g}/{sel.ratio:g} "
          f"-> post train {sel.post_train_dbic_member:.4f}, test lik {sel.post_test_lik_member:.4f}")
    print(f"  best post-refit train dBIC:    thr={orc.threshold:g}/{orc.ratio:g} "
          f"-> post train {orc.post_train_dbic_member:.4f}, test lik {orc.post_test_lik_member:.4f}")
    print("\n=== top 5 post-refit (train ΔBIC, per-member; test likelihood shown for confirmation) ===")
    print(df.nlargest(5, 'post_train_dbic_member')[
        ['threshold', 'ratio', 'n_par', 'best_step', 'pre_train_dbic_member',
         'post_train_dbic_member', 'gain', 'post_test_lik_member']
    ].round(4).to_string(index=False))
    print(f"\nSaved to {out_csv}\nBest models: {ckpt_dir}")


if __name__ == '__main__':
    main()
