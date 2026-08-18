"""Concept discovery on a fitted SPICE checkpoint, scored against behaviour.

Runs `spice.resources.sindy_concepts` on a Stage-2 checkpoint: collects the
one-step sufficient statistics, partitions each module's candidate terms into
concepts, fits gates/loadings/directions by ALS + closed-form gate tests, and
selects the structure by a Gaussian state-space MDL.

Behavioural likelihood and BIC are **measured**, never optimized -- the whole
discovery runs on the RNN's latent states. Nothing here trains the RNN
(`epochs=0`); the checkpoint's Stage-1 weights are used as-is.

Examples:

    python weinhardt2026/analysis/analysis_concepts.py --study dezfouli2019 \\
        --checkpoint weinhardt2026/studies/dezfouli2019/params/grid_refit_best_member/\\
spice_dezfouli2019_grid_0.05_0.7_BEST.pkl
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim
from spice.resources.sindy_concepts import (
    collect_sufficient_statistics, concept_string, discover_concepts,
    dof_per_participant, install_concepts, refit_concepts_shooting, summary_row,
)
from weinhardt2026.analysis.analysis_model_evaluation import (
    get_participant_experiment_groups, grouped_information_criteria,
)
from weinhardt2026.analysis.analysis_stage22_refit_grid import STUDIES, load_study


# ---------------------------------------------------------------------------
# Behavioural scoring (measurement only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score(model, dataset, dof: np.ndarray, batch_size: int, device) -> Dict[str, float]:
    """Per-member trial likelihood and grouped BIC for a given parameter count.

    Each ensemble member is scored on its own predictions and the losses
    averaged, matching `analysis_stage22_refit_grid.score_member`: the published
    equation is a member (or their coefficient mean), so members must stand on
    their own rather than hiding behind errors that cancel in an average.

    Only read `bic`/`dbic` from a **training** split. BIC is an in-sample
    criterion: the `k log n` term stands in for the generalization gap, so
    applying it to held-out data charges for parsimony twice -- once implicitly
    through the held-out likelihood, which already reflects whatever the
    parameters overfit, and once explicitly through the penalty. Doing that on
    this study ranked a 5.6-parameter model above a 10.7-parameter one that beat
    it on both the in-sample criterion and raw held-out likelihood. Select on
    training BIC; evaluate held-out data on NLL/likelihood alone.
    """
    model.eval(use_sindy=True)
    xs, ys = dataset.xs, dataset.ys
    valid = ~torch.isnan(xs[:, :, 0, 0])
    targets = torch.nan_to_num(ys, nan=0.0)
    step = batch_size if batch_size else xs.shape[0]
    chunks = []
    for i in range(0, xs.shape[0], step):
        batch = xs[i:i + step].to(device)
        model.init_state(batch_size=batch.shape[0])
        logits, _ = model(batch)
        chunks.append(torch.log_softmax(logits, dim=-1).cpu())
    log_probs = torch.cat(chunks, dim=1)                              # (E, B, T, W, A)
    ll = (targets.unsqueeze(0) * log_probs).sum(-1).sum(-1)           # (E, B, T)
    nll_per_session = -(ll * valid.unsqueeze(0)).sum(dim=2).mean(dim=0)

    unique_pairs, group_index = get_participant_experiment_groups(dataset)
    n_par = torch.tensor(dof, dtype=torch.float32)[unique_pairs[:, 0], unique_pairs[:, 1]]
    info = grouped_information_criteria(
        nll_per_session=nll_per_session, n_trials_per_session=valid.sum(dim=1).float(),
        group_index=group_index, n_groups=unique_pairs.shape[0],
        n_parameters_per_group=n_par, n_actions_baseline=dataset.n_actions,
    )
    n_trials = valid.sum().item()
    nll_total = float(nll_per_session.sum())
    return dict(lik=float(np.exp(-nll_total / n_trials)), nll=nll_total,
                bic=info['bic_mean'], dbic=info['delta_bic_per_trial_mean'],
                n_par=float(np.mean(dof)))


@torch.no_grad()
def refit_baseline_support_onestep(estimator, stats) -> None:
    """Refit coefficients on the checkpoint's own support, one-step, in closed form.

    The control this pipeline needs to be interpretable. The checkpoint's
    coefficients came out of Stage 2.2's K-step shooting refit, while concept
    discovery runs one-step, so a raw baseline-vs-concepts comparison confounds
    "cost of imposing concepts" with "cost of the one-step objective". This
    refits the *unchanged* support against the same one-step statistics the
    concepts are fit to, so the two differences separate:

        baseline -> this row      = one-step vs shooting
        this row -> concept rows  = the cost of the concept structure

    Restricted least squares per unit: c_S = (G_SS)^-1 b_S, zeros elsewhere.
    """
    from spice.resources.sindy_concepts import _ridged, _solve

    model = getattr(estimator, 'model', estimator)
    for module, st in stats.items():
        presence = model.sindy_coefficients_presence[module].reshape(st.U, -1)
        mask = presence[:, torch.tensor(st.term_indices, dtype=torch.long)].double().cpu()  # (U, T)
        # Zero the off-support rows/cols and put 1 on their diagonal so the batched
        # solve stays non-singular and returns exactly 0 for pruned terms.
        G = st.G * mask[:, :, None] * mask[:, None, :]
        G = G + torch.diag_embed(1.0 - mask)
        c = _solve(_ridged(G, rel=1e-10), st.b * mask) * mask
        full = torch.zeros(st.U, model.sindy_coefficients[module].shape[-1], dtype=torch.float64)
        full[:, torch.tensor(st.term_indices, dtype=torch.long)] = c
        target = model.sindy_coefficients[module]
        target.data = full.reshape(st.E, st.P, st.X, -1).to(target.dtype).to(target.device)


def baseline_structure_nats(model) -> float:
    """Sum over participants of log C(T, k_p): the cost of naming an arbitrary
    support, which an ordinary BIC never charges and which a shared partition
    pays only once."""
    from scipy.special import gammaln

    total = 0.0
    coefficients = model.get_sindy_coefficients(aggregate=True)
    for m in model.get_modules():
        prior = model.sindy_coefficients_prior_mask[m]
        T = int(prior.reshape(-1, prior.shape[-1]).any(dim=0).sum())
        k = (coefficients[m] != 0).sum(dim=-1).cpu().numpy().reshape(-1)
        k = np.clip(k, 0, T)
        total += float(np.sum(gammaln(T + 1) - gammaln(k + 1) - gammaln(T - k + 1)))
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--study', required=True, choices=sorted(STUDIES))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sweeps', type=int, default=3, help='partition refinement sweeps')
    parser.add_argument('--weights', type=float, nargs='+',
                        default=[1, 10, 50, 200, 800, 3000, 12000],
                        help='complexity weights to sweep (see ConceptFit.mdl)')
    parser.add_argument('--refit-shooting', action='store_true',
                        help='after selecting a structure, re-estimate loadings and '
                             'directions against Stage 2.2\'s K-step shooting objective')
    parser.add_argument('--shooting-steps', type=int, default=20)
    parser.add_argument('--refit-epochs', type=int, default=300)
    parser.add_argument('--refit-lr', type=float, default=1e-2)
    parser.add_argument('--refit-patience', type=int, default=30)
    parser.add_argument('--refit-batch-sessions', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--data', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    spice_class, config, spec = load_study(args.study)
    study_dir = ROOT / 'weinhardt2026' / 'studies' / args.study
    data_path = args.data or str(study_dir / 'data' / f'{args.study}.csv')
    output_dir = args.output_dir or str(study_dir / 'results' / 'concepts')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    dataset = csv_to_dataset(file=data_path)
    dataset.normalize_rewards()
    dataset_train, dataset_test = split_data_along_blockdim(dataset, spec['test_blocks'])

    state = torch.load(args.checkpoint, map_location='cpu')
    first = next(iter(config.library_setup))
    E, P = state['model'][f'sindy_coefficients.{first}'].shape[:2]
    del state

    # epochs=0: the estimator is a container for the checkpoint. Stage 1 is never
    # run -- the RNN weights come from the checkpoint and stay frozen throughout.
    estimator = SpiceEstimator(
        spice_class=spice_class, spice_config=config, n_actions=dataset.n_actions,
        n_participants=P, sindy_library_polynomial_degree=spec['polynomial_degree'],
        ensemble_size=E, use_sindy=True, kwargs_spice_class=spec['model_kwargs'],
        epochs=0, sindy_refit=False, device=device, verbose=False)
    estimator.load_spice(args.checkpoint)
    model = estimator.model

    print(f"study={args.study}  device={device}  E={E}  P={P}")
    print(f"train {dataset_train.xs.shape[0]} / test {dataset_test.xs.shape[0]} sessions"
          f"  |  checkpoint {Path(args.checkpoint).name}\n")

    # ── Baseline ─────────────────────────────────────────────────────────
    base_dof = model.count_sindy_coefficients().cpu().numpy()
    base_train = score(model, dataset_train, base_dof, args.batch_size, device)
    base_test = score(model, dataset_test, base_dof, args.batch_size, device)
    base_structure = baseline_structure_nats(model)
    _agg = model.get_sindy_coefficients(aggregate=True)
    _patterns = set()
    for p in range(P):
        _patterns.add(tuple(
            bool(v) for m in model.get_modules()
            for v in (_agg[m][p, 0] != 0).cpu().numpy().tolist()
        ))
    n_supports = len(_patterns)
    print(f"BASELINE  par/participant {base_dof.mean():.2f}"
          f"  |  train lik {base_train['lik']:.4f}  test lik {base_test['lik']:.4f}"
          f"  |  train dBIC {base_train['dbic']:+.4f}  test dBIC {base_test['dbic']:+.4f}")
    print(f"          distinct support patterns across participants: {n_supports}/{P}")
    print(f"          structure code (sum_p log C(T,k_p)): {base_structure:.0f} nats "
          f"= {2 * base_structure / max(1, (~torch.isnan(dataset_train.xs[:, :, 0, 0])).sum().item()):.4f} BIC/trial\n")

    # ── Sufficient statistics ────────────────────────────────────────────
    print("Collecting one-step sufficient statistics (unmasked library)...")
    t0 = time.time()
    xs5 = dataset_train.xs.unsqueeze(0).expand(E, -1, -1, -1, -1).contiguous()
    ys5 = dataset_train.ys.unsqueeze(0).expand(E, -1, -1, -1, -1).contiguous()
    stats = collect_sufficient_statistics(estimator, xs5, ys5, verbose=True)
    print(f"  [{time.time() - t0:.1f}s]\n")

    # ── Discovery, swept over the complexity weight ──────────────────────
    # The nominal row count overstates the state-space objective's effective
    # sample size (one row per item per trial, strongly autocorrelated within a
    # session), so a textbook Gaussian BIC on it keeps essentially every
    # parameter. Sweeping the complexity weight traces the whole reduction /
    # likelihood trade-off; selection is by *in-sample* behavioural BIC, with
    # the test split held back as a genuine out-of-sample check.
    aggregated = {m: v.clone() for m, v in model.get_sindy_coefficients(aggregate=True).items()}
    presence = {m: model.sindy_coefficients_presence[m].cpu().numpy().astype(bool)
                for m in model.get_modules()}
    original = {m: (model.sindy_coefficients[m].data.clone(),
                    model.sindy_coefficients_presence[m].clone())
                for m in model.get_modules()}

    # Control: same support, one-step objective (see refit_baseline_support_onestep).
    refit_baseline_support_onestep(estimator, stats)
    ctrl_train = score(model, dataset_train, base_dof, args.batch_size, device)
    ctrl_test = score(model, dataset_test, base_dof, args.batch_size, device)
    print(f"CONTROL (baseline support, one-step refit)  par {base_dof.mean():.2f}"
          f"  |  train lik {ctrl_train['lik']:.4f}  test lik {ctrl_test['lik']:.4f}"
          f"  |  train dBIC {ctrl_train['dbic']:+.4f}  test dBIC {ctrl_test['dbic']:+.4f}\n")
    for module in model.get_modules():
        model.sindy_coefficients[module].data = original[module][0].clone()
        model.sindy_coefficients_presence[module] = original[module][1].clone()

    seeds = {}
    for module, st in stats.items():
        coefficients = aggregated[module].reshape(P * model.n_experiments, -1).detach().cpu().numpy()
        support = presence[module].all(axis=0).reshape(P * model.n_experiments, -1)
        seeds[module] = (coefficients[:, st.term_indices], support[:, st.term_indices])

    results, all_fits = [], {}
    for weight in args.weights:
        print(f"--- complexity weight {weight:g} ---")
        fits = {}
        t0 = time.time()
        for module, st in stats.items():
            coefficients, support = seeds[module]
            fits[module] = discover_concepts(
                st, coefficients, support, n_sweeps=args.sweeps,
                verbose=False, complexity_weight=weight)
            print(f"  {module:32s} K={fits[module].K:3d}  "
                  f"loadings/unit={float(fits[module].n_loadings_per_unit().mean()):5.2f}")

        for module in model.get_modules():
            model.sindy_coefficients[module].data = original[module][0].clone()
            model.sindy_coefficients_presence[module] = original[module][1].clone()
        install_concepts(estimator, fits)

        dof_amort = dof_per_participant(fits, amortize_shared=True)
        dof_raw = dof_per_participant(fits, amortize_shared=False)
        tr = score(model, dataset_train, dof_amort, args.batch_size, device)
        te = score(model, dataset_test, dof_amort, args.batch_size, device)
        tr_raw = score(model, dataset_train, dof_raw, args.batch_size, device)
        # Selection reads dbic_train only. Held-out columns are NLL/likelihood --
        # no information criterion on the test split (see `score`).
        results.append(dict(
            weight=weight, par=dof_amort.mean(), par_loadings=dof_raw.mean(),
            lik_train=tr['lik'], bic_train=tr['bic'], dbic_train=tr['dbic'],
            dbic_train_raw=tr_raw['dbic'],
            lik_test=te['lik'], nll_test=te['nll'], gap=tr['lik'] - te['lik'],
            structure_nats=sum(f.structure_nats() for f in fits.values()),
            K_total=sum(f.K for f in fits.values())))
        all_fits[weight] = fits
        print(f"  par {dof_amort.mean():.2f}  train lik {tr['lik']:.4f} dBIC {tr['dbic']:+.4f}"
              f"  |  test lik {te['lik']:.4f} (gap {tr['lik'] - te['lik']:+.4f})"
              f"  [{time.time() - t0:.0f}s]\n")

    df = pd.DataFrame(results)
    best_weight = float(df.loc[df['dbic_train'].idxmax(), 'weight'])
    fits = all_fits[best_weight]

    # Leave the model at the selected structure.
    for module in model.get_modules():
        model.sindy_coefficients[module].data = original[module][0].clone()
        model.sindy_coefficients_presence[module] = original[module][1].clone()
    install_concepts(estimator, fits)

    # ── Concept-constrained Stage 2.2 ────────────────────────────────────
    # Discovery runs one-step because that is where the objective is quadratic.
    # This re-estimates the loadings and directions inside the frozen structure
    # against the same K-step shooting objective the pipeline's Stage 2.2 uses,
    # closing the gap the `control` row measures.
    refit_train = refit_test = None
    if args.refit_shooting:
        xs5_full = dataset_train.xs.unsqueeze(0).expand(E, -1, -1, -1, -1).contiguous()
        ys5_full = dataset_train.ys.unsqueeze(0).expand(E, -1, -1, -1, -1).contiguous()
        _, refit_history = refit_concepts_shooting(
            estimator, fits, xs5_full, ys5_full, shooting_steps=args.shooting_steps,
            epochs=args.refit_epochs, lr=args.refit_lr, patience=args.refit_patience,
            batch_size_sessions=args.refit_batch_sessions, verbose=True)
        refit_train = score(model, dataset_train, dof_per_participant(fits, True),
                            args.batch_size, device)
        refit_test = score(model, dataset_test, dof_per_participant(fits, True),
                           args.batch_size, device)

    # ── Report ───────────────────────────────────────────────────────────
    print("=" * 78)
    print("COMPLEXITY SWEEP  (select on train dBIC; held-out split reported as likelihood)")
    print("=" * 78)
    print(df.to_string(index=False, float_format='{:.4f}'.format))
    print(f"\nbaseline (shooting-fit):     par {base_dof.mean():.2f}  "
          f"train lik {base_train['lik']:.4f} dBIC {base_train['dbic']:+.4f}  |  "
          f"test lik {base_test['lik']:.4f} (gap {base_train['lik'] - base_test['lik']:+.4f})")
    print(f"control  (one-step, same support): par {base_dof.mean():.2f}  "
          f"train lik {ctrl_train['lik']:.4f} dBIC {ctrl_train['dbic']:+.4f}  |  "
          f"test lik {ctrl_test['lik']:.4f} (gap {ctrl_train['lik'] - ctrl_test['lik']:+.4f})")
    print(f"selected complexity weight: {best_weight:g}")
    if refit_train is not None:
        print(f"concepts + Stage 2.2 shooting (K={args.shooting_steps}): "
              f"par {refit_train['n_par']:.2f}  "
              f"train lik {refit_train['lik']:.4f} dBIC {refit_train['dbic']:+.4f}  |  "
              f"test lik {refit_test['lik']:.4f} "
              f"(gap {refit_train['lik'] - refit_test['lik']:+.4f})   "
              f"[shooting loss {refit_history['initial']:.6f} -> "
              f"{refit_history['best_loss']:.6f} @epoch {refit_history['best_epoch']}]")

    print("\n" + "=" * 78)
    print(f"DISCOVERED CONCEPTS  (weight {best_weight:g})")
    print("=" * 78)
    for module, fit in fits.items():
        print(f"\n{module}  (T={fit.stats.T}, K={fit.K}, "
              f"loadings/participant={float(fit.n_loadings_per_unit().mean()):.2f})")
        print(concept_string(fit))

    concept_structure = sum(f.structure_nats() for f in fits.values())
    n_train_trials = (~torch.isnan(dataset_train.xs[:, :, 0, 0])).sum().item()
    print(f"\nstructure code: baseline {base_structure:.0f} nats -> concepts "
          f"{concept_structure:.0f} nats  ({base_structure - concept_structure:+.0f} nats "
          f"= {2 * (base_structure - concept_structure) / n_train_trials:+.4f} BIC/trial)")

    summary = pd.DataFrame([summary_row(f) for f in fits.values()])
    print("\nPer-module summary:")
    print(summary.to_string(index=False, float_format='{:.3f}'.format))

    name = Path(args.checkpoint).stem
    df.to_csv(os.path.join(output_dir, f'{name}_concepts_sweep.csv'), index=False)
    summary.to_csv(os.path.join(output_dir, f'{name}_concepts_summary.csv'), index=False)
    with open(os.path.join(output_dir, f'{name}_concepts.txt'), 'w') as fh:
        fh.write(f"selected complexity weight: {best_weight:g}\n\n")
        for module, fit in fits.items():
            fh.write(f"{module}  (T={fit.stats.T}, K={fit.K})\n")
            fh.write(concept_string(fit) + "\n\n")
    estimator.save_spice(os.path.join(output_dir, f'{name}_concepts.pkl'))
    print(f"\nSaved to {output_dir}")


if __name__ == '__main__':
    main()
