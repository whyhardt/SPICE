"""Greedy discovery of coefficient ties, selected by BIC.

Model reduction that keeps the equations in their own coefficients. Instead of
factorizing the per-participant coefficient matrix into shared "mechanisms"
(see `spice/resources/sindy_compression.py` and its caveats), this searches
for redundancy *between* coefficients -- groups of terms whose per-participant
values lie on a shared low-dimensional set, so one free parameter stands in
for several coefficients:

    Q[t+1] = Q[t] + c_Q Q[t] + c_R R[t]     with   c_R = -c_Q
    -->  Q[t+1] = (1-a) Q[t] + a R[t]        one parameter a, not two

The loop is:

  1. Screen candidate ties from the fitted coefficients (`screen_candidates`).
  2. Take the tightest surviving candidate, refit *every* remaining free
     parameter -- per-participant loadings and the shared constants alike --
     against behavioural NLL, and score BIC.
  3. Accept if BIC improves; either way continue down the list.

Candidates are screened once, from the checkpoint's coefficients. Every
candidate is then evaluated by a refit starting from that same checkpoint with
the accepted ties re-imposed, rather than from the previous candidate's
end state, so the comparison against the running best is like for like and the
result does not depend on the order refits happened to leave the parameters
in. A candidate that overlaps an accepted tie is replaced by the merge of the
two (`merge_groups`) -- ``c_i = lam c_j`` and ``c_j = mu c_k`` are one
statement about three terms, not two independent constraints.

BIC is the objective throughout, in the same per-participant form the rest of
the codebase reports (`grouped_information_criteria`): NLL averaged across
ensemble members, parameters counted for one member, shared population
constants amortized across participants. A tie is worth having exactly when
the degrees of freedom it removes outweigh the likelihood it costs.

Caveats worth keeping straight when reading the output:

  * A tight tie is not automatically a cognitive finding. Some hold because
    the library directions involved are near-collinear -- on dezfouli2019,
    ``choice[t-3]`` against ``choice[t-1]*choice[t-3]`` (both binary) is a
    near-flat direction of the likelihood. Those still earn their BIC, they
    just do it by removing a parameter the data never constrained. Separating
    the two needs a curvature check, not this script.
  * Selection uses the evaluation split -- both early stopping and the
    accept/reject test read it -- so every post-tie number reported here is
    optimistic and is not a held-out estimate. Confirming a discovered tie set
    needs a third split it never touched.

Examples:

    python weinhardt2026/analysis/analysis_coefficient_ties.py \\
        --study dezfouli2019 \\
        --checkpoint weinhardt2026/studies/dezfouli2019/params/grid_refit_best_member/spice_dezfouli2019_grid_0.05_0.7_BEST.pkl

    # cheaper pass: fewer refit steps per candidate, tighter screen
    python weinhardt2026/analysis/analysis_coefficient_ties.py --study dezfouli2019 \\
        --max-candidates 40 --refit-steps 150 --max-error 0.45
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim
from spice.resources.sindy_ties import (
    TieSet, extract_coefficients_and_support, functional_coefficients, merge_groups,
    install_coefficients, screen_candidates, tied_equation_string,
)
from weinhardt2026.analysis.analysis_model_evaluation import (
    get_participant_experiment_groups, grouped_information_criteria,
)
from weinhardt2026.analysis.analysis_stage22_refit_grid import STUDIES, load_study


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _nll_per_session(model, dataset, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-session NLL averaged across ensemble members, plus the valid-trial
    mask. Mirrors `analysis_stage22_refit_grid.score_member`: every member is
    scored on its own predictions, so members cannot trade individual accuracy
    for errors that cancel in an average."""
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
    log_probs = torch.cat(chunks, dim=1)                                  # (E, B, T, W, A)
    ll = (targets.unsqueeze(0) * log_probs).sum(-1).sum(-1)               # (E, B, T)
    return -(ll * valid.unsqueeze(0)).sum(dim=2).mean(dim=0), valid


@torch.no_grad()
def score(model, dataset, dof: np.ndarray, batch_size: int, device) -> Dict[str, float]:
    """BIC / ΔBIC-per-trial for a given per-participant parameter count.

    ``dof`` replaces `count_sindy_coefficients()` so that tied parameters are
    counted once rather than per term -- everything else matches the
    conventions used across the analysis scripts.

    Read `bic`/`dbic` from a **training** split only. Every tie changes the
    parameter count, so this is precisely the case where an information
    criterion on held-out data does damage: the `k log n` penalty duplicates
    what the held-out likelihood already measures, and the accept/reject test
    would then favour ties for reasons that have nothing to do with the data
    supporting them. Held-out splits are reported via `lik`/`nll`.
    """
    model.eval(use_sindy=True)
    nll, valid = _nll_per_session(model, dataset, batch_size, device)
    unique_pairs, group_index = get_participant_experiment_groups(dataset)
    n_par = torch.tensor(dof, dtype=torch.float32)[unique_pairs[:, 0], unique_pairs[:, 1]]
    info = grouped_information_criteria(
        nll_per_session=nll, n_trials_per_session=valid.sum(dim=1).float(),
        group_index=group_index, n_groups=unique_pairs.shape[0],
        n_parameters_per_group=n_par, n_actions_baseline=dataset.n_actions,
    )
    n_trials = valid.sum().item()
    return dict(bic=info['bic_mean'], dbic=info['delta_bic_per_trial_mean'],
                nll=info['nll_total'], lik=float(np.exp(-info['nll_total'] / n_trials)),
                n_par=float(np.mean(dof)))


# ---------------------------------------------------------------------------
# Constrained refit
# ---------------------------------------------------------------------------

def refit(tie_set: TieSet, model, dataset_train, dataset_eval, dof: np.ndarray,
          steps: int, lr: float, batch_size: int, patience: int, min_delta: float,
          device) -> Tuple[Dict[str, float], int]:
    """Refit the tie set's free parameters against behavioural NLL.

    Optimizes the untied coefficients, the per-participant group loadings and
    the shared constants (``lam``, ``gamma``, constant values) jointly -- the
    shared part of a tie is a population parameter fit to behaviour, not a
    number frozen from the screening step.

    Early stopping and the returned score both use ``dataset_eval``. Step 0 is
    scored and seeded as the best, so a tie the refit only harms is reported
    at its honest starting value rather than at a manufactured later one.
    """
    model.eval(use_sindy=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam([p for p in tie_set.parameters() if p.requires_grad], lr=lr)
    xs, ys = dataset_train.xs, dataset_train.ys
    valid_total = (~torch.isnan(xs[:, :, 0, 0])).sum().item()
    step_size = batch_size if batch_size else xs.shape[0]
    E = model.ensemble_size

    def _evaluate() -> Dict[str, float]:
        """Held-out score used for early stopping -- likelihood, not ΔBIC.

        The support is fixed for the duration of one refit, so ranking steps by
        likelihood is identical to ranking them by ΔBIC would have been, minus
        the invalid penalty on held-out data.
        """
        with torch.no_grad():
            install_coefficients(model, {m: v.detach() for m, v in tie_set.build_coefficients().items()})
        return score(model, dataset_eval, dof, batch_size, device)

    best = dict(score=_evaluate(), step=0,
                state={k: v.detach().clone() for k, v in tie_set.state_dict().items()})
    stale = 0

    with functional_coefficients(model) as holder:
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            model.train(mode=True, use_sindy=True)
            coefficients = tie_set.build_coefficients()
            for module, values in coefficients.items():
                holder[module] = values
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
                # Gradient accumulation over session batches is the exact
                # full-batch gradient; retain the graph because every batch
                # shares the one `build_coefficients` call above.
                loss.backward(retain_graph=i + step_size < xs.shape[0])
            optimizer.step()

            model.eval(use_sindy=True)
            current = _evaluate()
            if current['lik'] > best['score']['lik'] + min_delta:
                best = dict(score=current, step=step,
                            state={k: v.detach().clone() for k, v in tie_set.state_dict().items()})
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    tie_set.load_state_dict(best['state'])
    with torch.no_grad():
        install_coefficients(model, tie_set.build_coefficients())
    return best['score'], best['step']


# ---------------------------------------------------------------------------
# Greedy search
# ---------------------------------------------------------------------------

def greedy_ties(estimator, dataset_train, dataset_eval, args, device) -> Tuple[TieSet, List[dict]]:
    model = estimator.model
    coefficients, presence, terms = extract_coefficients_and_support(model)
    base_count = model.count_sindy_coefficients().cpu().numpy()

    accepted = TieSet(coefficients, presence, terms, device=device)
    baseline, _ = refit(accepted, model, dataset_train, dataset_eval,
                        accepted.dof_per_participant(base_count), args.refit_steps, args.lr,
                        args.batch_size, args.patience, args.min_delta, device)
    baseline_train = score(model, dataset_train, accepted.dof_per_participant(base_count),
                           args.batch_size, device)
    print(f"baseline (no ties, refit): dBIC {baseline['dbic']:+.4f}  BIC {baseline['bic']:.1f}  "
          f"lik test {baseline['lik']:.4f} train {baseline_train['lik']:.4f}  "
          f"par {baseline['n_par']:.2f}\n", flush=True)

    candidates = screen_candidates(
        coefficients, presence, terms, min_coverage=args.min_coverage,
        max_error=args.max_error, use_offset=args.affine,
        include_geometric=not args.no_geometric, include_constant=not args.no_constant,
    )
    print(f"{len(candidates)} screened candidates; evaluating up to {args.max_candidates}\n", flush=True)

    current, current_train = baseline, baseline_train
    history: List[dict] = []
    for rank, candidate in enumerate(candidates[:args.max_candidates], start=1):
        # Overlapping ties are one statement, not several: fold the candidate
        # together with every accepted group it touches, and drop them all.
        # Each merge intersects the participant sets, so a merge that ends up
        # applying to too few people is dropped rather than forced through.
        clashes = accepted.conflicts(candidate)
        for clash in clashes:
            candidate = merge_groups(clash, candidate, coefficients[candidate.module])
            if candidate is None:
                break
        if candidate is None or candidate.dof_saved <= 0:
            continue
        if accepted.contains(candidate):
            continue
        trial = accepted.copy_with(candidate, drop=clashes)
        trial.assert_disjoint()

        dof = trial.dof_per_participant(base_count)
        t0 = time.time()
        result, best_step = refit(trial, model, dataset_train, dataset_eval, dof, args.refit_steps,
                                  args.lr, args.batch_size, args.patience, args.min_delta, device)
        # `refit` leaves the model at its best-scoring state, so the train-split
        # numbers below describe the same parameters the eval numbers do. The
        # eval split drives early stopping and acceptance, so its likelihood is
        # optimistic; the gap to the train likelihood is the honest read on
        # whether a tie is buying fit or buying selection.
        train = score(model, dataset_train, dof, args.batch_size, device)
        # Accept on the *training* criterion. A tie changes the parameter count,
        # so the trade-off it makes is exactly what BIC is for -- and exactly
        # what a held-out ΔBIC would double-charge. The eval split still drives
        # early stopping inside `refit`, on likelihood.
        gain = train['dbic'] - current_train['dbic']
        keep = gain > args.accept_delta
        label = candidate.describe().splitlines()[0]
        # Report the *net* parameters this step removes. A merge absorbs groups
        # whose savings were already banked, so the merged group's own
        # `dof_saved` reads as if all of them were newly removed.
        net_saved = int(round((current['n_par'] - result['n_par']) * dof.size))
        print(f"[{rank:3d}/{min(len(candidates), args.max_candidates)}] {label}\n"
              f"        err={candidate.fit_error:.3f} saves={net_saved:4d} "
              f"(group {candidate.dof_saved}) "
              f"par {current['n_par']:.2f}->{result['n_par']:.2f} | "
              f"train dBIC {current_train['dbic']:+.4f}->{train['dbic']:+.4f} ({gain:+.4f}) "
              f"lik test {result['lik']:.4f} train {train['lik']:.4f} | "
              f"best@{best_step}{'!' if best_step >= args.refit_steps else ''} "
              f"[{time.time() - t0:.0f}s] "
              f"{'ACCEPT' if keep else 'reject'}", flush=True)

        history.append(dict(rank=rank, module=candidate.module, kind=candidate.kind,
                            terms=' + '.join(candidate.term_names), size=candidate.size,
                            group_rank=candidate.rank, screen_error=candidate.fit_error,
                            dof_saved=candidate.dof_saved, net_saved=net_saved, n_applicable=candidate.n_applicable,
                            n_par=result['n_par'], dbic=result['dbic'], bic=result['bic'],
                            lik=result['lik'], lik_train=train['lik'], dbic_train=train['dbic'], gain=gain, accepted=keep, best_step=best_step))
        if keep:
            accepted, current, current_train = trial, result, train
        else:
            with torch.no_grad():
                install_coefficients(model, accepted.build_coefficients())

    return accepted, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--study', required=True, choices=sorted(STUDIES))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--refit-steps', type=int, default=200,
                        help='max NLL refit steps per candidate evaluation')
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--accept-delta', type=float, default=0.0,
                        help='minimum dBIC-per-trial gain to accept a tie')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--max-candidates', type=int, default=60)
    parser.add_argument('--min-coverage', type=float, default=0.25,
                        help='min fraction of participants with all the tie\'s terms present')
    parser.add_argument('--max-error', type=float, default=0.6,
                        help='max relative residual for a candidate to be screened in')
    parser.add_argument('--affine', action='store_true',
                        help='allow an additive offset in a tie (c_i = lam c_j + k) '
                             'instead of the pure-ratio form')
    parser.add_argument('--no-geometric', action='store_true')
    parser.add_argument('--no-constant', action='store_true')
    parser.add_argument('--data', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    spice_class, config, spec = load_study(args.study)
    study_dir = ROOT / 'weinhardt2026' / 'studies' / args.study
    data_path = args.data or str(study_dir / 'data' / f'{args.study}.csv')
    output_dir = args.output_dir or str(study_dir / 'results' / 'coefficient_ties')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    dataset = csv_to_dataset(file=data_path)
    dataset.normalize_rewards()
    dataset_train, dataset_eval = split_data_along_blockdim(dataset, spec['test_blocks'])

    state = torch.load(args.checkpoint, map_location='cpu')
    first = next(iter(config.library_setup))
    E, P = state['model'][f'sindy_coefficients.{first}'].shape[:2]
    del state

    estimator = SpiceEstimator(
        spice_class=spice_class, spice_config=config, n_actions=dataset.n_actions,
        n_participants=P, sindy_library_polynomial_degree=spec['polynomial_degree'],
        ensemble_size=E, use_sindy=True, kwargs_spice_class=spec['model_kwargs'], device=device)
    estimator.load_spice(args.checkpoint)

    print(f"study={args.study} device={device} | E={E} P={P} | train {dataset_train.xs.shape[0]} "
          f"eval {dataset_eval.xs.shape[0]} sessions | checkpoint {Path(args.checkpoint).name}\n",
          flush=True)

    tie_set, history = greedy_ties(estimator, dataset_train, dataset_eval, args, device)

    name = Path(args.checkpoint).stem
    pd.DataFrame(history).to_csv(os.path.join(output_dir, f'{name}_ties.csv'), index=False)
    estimator.save_spice(os.path.join(output_dir, f'{name}_tied.pkl'))
    torch.save({'state_dict': tie_set.state_dict(),
                'groups': [g.__dict__ for g in tie_set.groups]},
               os.path.join(output_dir, f'{name}_tieset.pt'))

    print("\n=== accepted ties ===")
    print(tie_set.summary())
    print("\n=== example participants ===")
    coefficients = tie_set.build_coefficients()
    for participant in range(min(3, P)):
        print(f"\n-- participant {participant} --")
        print(tied_equation_string(tie_set, coefficients, participant))
    print(f"\nSaved to {output_dir}")


if __name__ == '__main__':
    main()
