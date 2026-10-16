"""Uniform evaluation across studies: training BIC/AIC (per participant), held-out
likelihood, and paired per-participant comparison SPICE-EQ vs Benchmark.

Loads the study's final models only (no training). Usage: python eval_studies_uniform.py <study>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path('/home/daniel/repositories/SPICE')
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'weinhardt2026/results/eval_studies_uniform'
OUT.mkdir(exist_ok=True)

from weinhardt2026.analysis.analysis_model_evaluation import get_participant_experiment_groups
from weinhardt2026.figures.figure_model_evaluation import load_study, load_spice
from weinhardt2026.utils.final_models import final_checkpoints, load_best

EPS = 1e-9


def setup(study):
    """Return dataset_train, dataset_test, benchmark, gru, spice estimators list, trial_filter, n_actions_baseline.

    Data as the models were trained on it (figures/figure_model_evaluation.py); SPICE: every final seed,
    benchmark / GRU: the best final seed on the training data (weinhardt2026/utils/final_models.py).
    """
    spice_module, tr, te = load_study(study)
    spice = [load_spice(path, spice_module, tr) for path in final_checkpoints(study, 'spice')]
    benchmark = load_best(study, 'benchmark')
    gru = load_best(study, 'gru')
    benchmark.eval()
    gru.eval()
    return tr, te, benchmark, gru, spice, None, None


@torch.no_grad()
def per_participant(dataset, probs, trial_filter):
    """Per-participant NLL and trial count (on the dataset's own (participant, experiment) groups)."""
    valid = ~torch.isnan(dataset.xs[:, :, 0, 0])
    if trial_filter is not None:
        valid = valid & trial_filter(dataset.ys)
    targets = dataset.ys.clone()
    targets[~valid] = float('nan')
    nll = -(targets * torch.log(probs.clamp(EPS, 1 - EPS)))
    nll_session = nll.sum(dim=-1).nansum(dim=1)[..., 0]  # (B,)
    pairs, group_index = get_participant_experiment_groups(dataset)
    n_groups = pairs.shape[0]
    nll_group = torch.zeros(n_groups, dtype=nll_session.dtype).scatter_add_(0, group_index.cpu(), nll_session.cpu())
    n_group = torch.zeros(n_groups).scatter_add_(0, group_index.cpu(), valid.sum(1).float().cpu())
    return {tuple(p): (nll_group[i].item(), n_group[i].item()) for i, p in enumerate(pairs.tolist()) if n_group[i] > 0}


@torch.no_grad()
def predict(model, dataset, kind):
    if kind == 'spice':
        model.eval(use_sindy=True)
        logits, _ = model(dataset.xs.to(model.device))
        return torch.softmax(logits.mean(dim=0), dim=-1).cpu()
    logits, _ = model.to('cpu')(dataset.xs)
    return torch.softmax(logits, dim=-1).cpu()


def main(study):
    tr, te, benchmark, gru, spice, trial_filter, baseline = setup(study)
    n_actions_baseline = baseline or tr.n_actions
    k_benchmark = benchmark.count_parameters() if hasattr(benchmark, 'count_parameters') else len(list(benchmark.parameters()))

    result = {'study': study, 'n_spice_models': len(spice)}
    per = {}
    for split, ds in (('train', tr), ('test', te)):
        per[('Benchmark', split)] = per_participant(ds, predict(benchmark, ds, 'bench'), trial_filter)
        per[('GRU', split)] = per_participant(ds, predict(gru, ds, 'bench'), trial_filter)
        runs = [per_participant(ds, predict(est, ds, 'spice'), trial_filter) for est in spice]
        # run-averaged per-participant NLL (a single model outside dezfouli2019)
        per[('SPICE-EQ', split)] = {key: (np.mean([r[key][0] for r in runs]), runs[0][key][1]) for key in runs[0]}
        per[('SPICE-EQ-runs', split)] = runs

    # SPICE per-participant parameter counts (mean over runs)
    k_spice_runs = [est.count_sindy_coefficients().float().cpu().numpy() for est in spice]
    k_spice = np.mean(k_spice_runs, axis=0)

    def criteria(model, key_k):
        train = per[(model, 'train')]
        keys = sorted(train)
        k = np.array([key_k(key) for key in keys])
        nll = np.array([train[key][0] for key in keys])
        n = np.array([train[key][1] for key in keys])
        return keys, k, nll, n

    for model in ('Benchmark', 'SPICE-EQ', 'GRU'):
        if model == 'Benchmark':
            key_k = lambda key: k_benchmark
        elif model == 'SPICE-EQ':
            key_k = lambda key: k_spice[key[0], key[1]]
        else:
            key_k = lambda key: sum(p.numel() for p in gru.parameters())
        keys, k, nll, n = criteria(model, key_k)
        test = per[(model, 'test')]
        test_nll = sum(v[0] for v in test.values())
        test_n = sum(v[1] for v in test.values())
        result[model] = {
            'k_mean': float(k.mean()),
            'trial_lik_train': math.exp(-nll.sum() / n.sum()),
            'trial_lik_test': math.exp(-test_nll / test_n),
            'BIC_train_mean': float((2 * nll + k * np.log(n)).mean()),
            'AIC_train_mean': float((2 * nll + 2 * k).mean()),
            'n_train_trials_per_participant': float(n.mean()),
            'n_participants': len(keys),
        }
    if len(spice) > 1:
        lik_runs = [math.exp(-sum(v[0] for v in r.values()) / sum(v[1] for v in r.values())) for r in per[('SPICE-EQ-runs', 'test')]]
        result['SPICE-EQ']['trial_lik_test_sd_runs'] = float(np.std(lik_runs))

    # Paired per-participant comparisons SPICE-EQ vs Benchmark
    bench_train, spice_train = per[('Benchmark', 'train')], per[('SPICE-EQ', 'train')]
    keys = sorted(set(bench_train) & set(spice_train))
    bic_b = np.array([2 * bench_train[k][0] + k_benchmark * np.log(bench_train[k][1]) for k in keys])
    bic_s = np.array([2 * spice_train[k][0] + k_spice[k[0], k[1]] * np.log(spice_train[k][1]) for k in keys])
    aic_b = np.array([2 * bench_train[k][0] + 2 * k_benchmark for k in keys])
    aic_s = np.array([2 * spice_train[k][0] + 2 * k_spice[k[0], k[1]] for k in keys])
    bench_test, spice_test = per[('Benchmark', 'test')], per[('SPICE-EQ', 'test')]
    keys_test = sorted(set(bench_test) & set(spice_test))
    ll_b = np.array([-bench_test[k][0] / bench_test[k][1] for k in keys_test])
    ll_s = np.array([-spice_test[k][0] / spice_test[k][1] for k in keys_test])
    # AIC-style effective parameters from the train->test drop in per-trial log-lik,
    # scaled by train trials. Rough: block differences also enter the drop, so read it
    # relative to the benchmark's value, whose true k is known.
    def k_eff(model):
        train, test = per[(model, 'train')], per[(model, 'test')]
        ks = sorted(set(train) & set(test))
        drop = np.array([test[k][0] / test[k][1] - train[k][0] / train[k][1] for k in ks])
        n_train = np.array([train[k][1] for k in ks])
        return float(np.median(drop * n_train))
    result['paired'] = {
        'frac_spice_better_BIC_train': float((bic_s < bic_b).mean()),
        'frac_spice_better_AIC_train': float((aic_s < aic_b).mean()),
        'frac_spice_better_heldout': float((ll_s > ll_b).mean()),
        'heldout_ll_per_trial_diff_median': float(np.median(ll_s - ll_b)),
        'wilcoxon_p_heldout': float(wilcoxon(ll_s, ll_b).pvalue),
        'n_participants_test': len(keys_test),
        'k_eff_benchmark': k_eff('Benchmark'),
        'k_eff_spice': k_eff('SPICE-EQ'),
    }
    (OUT / f'{study}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
