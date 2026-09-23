"""Uniform evaluation across studies: training BIC/AIC (per participant), held-out
likelihood, and paired per-participant comparison SPICE-EQ vs Benchmark.

Loads saved checkpoints only (no training), with the same model setup as each
study script. Usage: python eval_studies.py <study>
"""
import json
import math
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path('/home/daniel/repositories/SPICE')
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'weinhardt2026/results/eval_studies_uniform'
OUT.mkdir(exist_ok=True)

from spice import SpiceEstimator
from weinhardt2026.analysis.analysis_model_evaluation import get_participant_experiment_groups
from weinhardt2026.utils.benchmarking_gru import GRUModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-9


def setup(study):
    """Return dataset_train, dataset_test, {name: model}, spice estimators list, trial_filter, n_actions_baseline."""
    trial_filter, baseline = None, None
    if study == 'dezfouli2019':
        from spice.precoded.workingmemory import SpiceModel, CONFIG
        from weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019 import GQLModel, get_dataset
        tr, te, info = get_dataset(path_data=str(ROOT / 'weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv'), test_blocks=(3, 6, 9))
        benchmark = GQLModel(n_participants=info['n_participants'], batch_first=True)
        benchmark.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/dezfouli2019/params/benchmark_dezfouli2019.pkl', map_location='cpu'))
        gru = GRUModel(n_actions=info['n_actions'], n_participants=info['n_participants'], additional_inputs=2,
                       dropout=0.25, embedding_size=8, hidden_size=8)
        gru.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/dezfouli2019/params/gru_dezfouli2019.pkl', map_location='cpu'))
        spice = []
        for path in sorted(glob(str(ROOT / 'weinhardt2026/studies/dezfouli2019/params/params_0.05/spice_dezfouli2019_stability_[0-9].pkl'))):
            ckpt = torch.load(path, map_location='cpu')
            E = ckpt['model'][f'sindy_coefficients.{next(iter(CONFIG.library_setup))}'].shape[0]
            del ckpt
            est = SpiceEstimator(spice_class=SpiceModel, spice_config=CONFIG, n_actions=2,
                                 n_participants=info['n_participants'], ensemble_size=E,
                                 kwargs_spice_class={'reward_binary': True}, device=device)
            est.load_spice(path)
            spice.append(est)
    elif study == 'ganesh2024a':
        from weinhardt2026.studies.ganesh2024a.spice_ganesh2024a import SpiceModel, CONFIG
        from weinhardt2026.studies.ganesh2024a.benchmarking_ganesh2024a import BayesianModel, get_dataset
        tr, te, info = get_dataset(path_data=str(ROOT / 'weinhardt2026/studies/ganesh2024a/data/ganesh2024a.csv'), test_blocks=(3, 6, 9))
        est = SpiceEstimator(spice_class=SpiceModel, spice_config=CONFIG, n_actions=tr.n_actions,
                             n_participants=tr.n_participants, device=device)
        est.load_spice(str(ROOT / 'weinhardt2026/studies/ganesh2024a/params/spice_ganesh2024a.pkl'))
        spice = [est]
        benchmark = BayesianModel(n_participants=info['n_participants'], batch_first=True)
        benchmark.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/ganesh2024a/params/bay_ganesh2024a.pkl', map_location='cpu'))
        gru = GRUModel(n_actions=info['n_actions'], n_participants=info['n_participants'], additional_inputs=2,
                       dropout=0.25, embedding_size=8, hidden_size=8)
        gru.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/ganesh2024a/params/gru_ganesh2024a.pkl', map_location='cpu'))
    elif study == 'braun2018':
        from weinhardt2026.studies.braun2018.spice_braun2018 import SpiceModel, CONFIG
        from weinhardt2026.studies.braun2018.benchmarking_braun2018 import ExpectedValueControl, get_dataset
        tr, te, info = get_dataset(path_data=str(ROOT / 'weinhardt2026/studies/braun2018/data/braun2018.csv'), test_blocks=(3, 6, 9))
        n_trials = (~torch.isnan(tr.xs[:, :, 0, 0])).sum(1).float().numpy()
        cutoff = int(n_trials.mean() + 2 * n_trials.std())
        tr.xs, tr.ys = tr.xs[:, :cutoff], tr.ys[:, :cutoff]
        te.xs, te.ys = te.xs[:, :cutoff], te.ys[:, :cutoff]
        est = SpiceEstimator(spice_class=SpiceModel, spice_config=CONFIG, n_actions=tr.n_actions,
                             n_participants=tr.n_participants, n_reward_features=0, ensemble_size=10,
                             sindy_weight=0.01, device=device)
        est.load_spice(str(ROOT / 'weinhardt2026/studies/braun2018/params/spice_braun2018.pkl'))
        est.model.preprocess_coefficients()
        spice = [est]
        benchmark = ExpectedValueControl(n_participants=tr.n_participants)
        benchmark.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/braun2018/params/benchmark_braun2018.pkl', map_location='cpu'))
        gru = GRUModel(tr.n_actions, additional_inputs=4, n_reward_features=0)
        gru.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/braun2018/params/gru_braun2018.pkl', map_location='cpu'))
    elif study == 'bustamante2023':
        from weinhardt2026.studies.bustamante2023.spice_bustamante2023 import SpiceModel, CONFIG
        from weinhardt2026.studies.bustamante2023.benchmarking_bustamante2023 import MarginalValueTheoremModel, get_dataset
        tr, te, info = get_dataset(path_data=str(ROOT / 'weinhardt2026/studies/bustamante2023/data/bustamante2023.csv'), test_blocks=(3, 6))
        est = SpiceEstimator(spice_class=SpiceModel, spice_config=CONFIG, n_actions=tr.n_actions,
                             n_participants=tr.n_participants, sindy_weight=0, ensemble_size=1, device=device)
        est.load_spice(str(ROOT / 'weinhardt2026/studies/bustamante2023/params/spice_bustamante2023.pkl'))
        spice = [est]
        benchmark = MarginalValueTheoremModel(n_participants=tr.n_participants, depletion=None, baseline_gain=None, batch_first=True)
        benchmark.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/bustamante2023/params/mvt_bustamante2023.pkl', map_location='cpu'))
        gru = GRUModel(tr.n_actions)
        gru.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/bustamante2023/params/gru_bustamante2023.pkl', map_location='cpu'))
    elif study == 'eckstein2026':
        from weinhardt2026.studies.eckstein2026.spice_eckstein2026 import SpiceModel, CONFIG
        from weinhardt2026.studies.eckstein2026.benchmarking_eckstein2026 import Castro2025Model, get_dataset
        tr, te, info = get_dataset(path_data=str(ROOT / 'weinhardt2026/studies/eckstein2026/data/eckstein2026.csv'), test_blocks=(2,))
        est = SpiceEstimator(spice_class=SpiceModel, spice_config=CONFIG, n_actions=tr.n_actions,
                             n_participants=tr.n_participants, device=device)
        est.load_spice(str(ROOT / 'weinhardt2026/studies/eckstein2026/params/spice_eckstein2026.pkl'))
        spice = [est]
        benchmark = Castro2025Model(n_participants=tr.n_participants, n_actions=tr.n_actions, batch_first=True)
        benchmark.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/eckstein2026/params/benchmark_eckstein2026.pkl', map_location='cpu'))
        gru = GRUModel(n_actions=tr.n_actions, additional_inputs=2, dropout=0.1, hidden_size=32)
        gru.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/eckstein2026/params/gru_eckstein2026.pkl', map_location='cpu'))
    elif study == 'kolff2025':
        from weinhardt2026.studies.kolff2025.spice_kolff2025 import CONFIG, SpiceModel, cross_entropy_loss_mask_waiting, filter_non_waiting
        from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import ConditionalFrequencyModel, get_dataset
        tr, te, info = get_dataset(path_data=str(ROOT / 'weinhardt2026/studies/kolff2025/data/kolff2025_original.csv'), test_blocks=(1,))
        est = SpiceEstimator(spice_config=CONFIG, spice_class=SpiceModel, n_actions=tr.n_actions,
                             n_participants=tr.n_participants, n_reward_features=tr.n_reward_features,
                             embedding_size=4, loss_fn=cross_entropy_loss_mask_waiting, ensemble_size=10,
                             sindy_weight=0., device=device)
        est.load_spice(str(ROOT / 'weinhardt2026/studies/kolff2025/params/spice_kolff2025.pkl'))
        spice = [est]
        gru = GRUModel(n_actions=tr.n_actions, additional_inputs=tr.n_additional_inputs, n_reward_features=0)
        gru.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/kolff2025/params/gru_kolff2025.pkl', map_location='cpu'))
        benchmark = ConditionalFrequencyModel(n_actions=tr.n_actions, n_participants=info['n_participants'])
        benchmark.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/kolff2025/params/benchmark_kolff2025.pkl', map_location='cpu'))
        trial_filter, baseline = filter_non_waiting, tr.n_actions - 1
    elif study == 'kolff2025_groom':
        from weinhardt2026.studies.kolff2025.spice_kolff2025_groom import CONFIG, SpiceModel
        from weinhardt2026.studies.kolff2025.benchmarking_kolff2025_groom import get_dataset, ConditionalGroomingModel
        tr, te, info = get_dataset(path_data=str(ROOT / 'weinhardt2026/studies/kolff2025/data/kolff2025_original.csv'), test_fraction=0.2)
        est = SpiceEstimator(spice_config=CONFIG, spice_class=SpiceModel, n_actions=tr.n_actions,
                             n_participants=tr.n_participants, n_reward_features=tr.n_reward_features,
                             embedding_size=4, ensemble_size=10, sindy_weight=1e-1, sindy_alpha=1e-3,
                             sindy_threshold_pruning=5e-2, device=device)
        est.load_spice(str(ROOT / 'weinhardt2026/studies/kolff2025/params/spice_kolff2025_groom.pkl'))
        spice = [est]
        benchmark = ConditionalGroomingModel(n_participants=tr.n_participants)
        benchmark.fit(tr)
        gru = GRUModel(n_actions=tr.n_actions, n_participants=tr.n_participants,
                       additional_inputs=tr.n_additional_inputs, n_reward_features=0,
                       embedding_size=4, hidden_size=8)
        gru.load_state_dict(torch.load(ROOT / 'weinhardt2026/studies/kolff2025/params/gru_kolff2025_groom.pkl', map_location='cpu'))
    else:
        raise ValueError(study)
    benchmark.eval()
    gru.eval()
    return tr, te, benchmark, gru, spice, trial_filter, baseline


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
