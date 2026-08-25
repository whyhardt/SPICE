"""Analyze hyperparameter scan over pruning_threshold × pruning_test.

For each checkpoint, computes (SINDy autoregressive):
  - In-sample trial likelihood/NLL/BIC/AIC on training data
  - Hold-out trial likelihood and NLL on test blocks (suffixed `_test`), plus
    the generalization gap between them -- **no information criterion on the
    hold-out**, since BIC's `k log n` term stands in for exactly the gap the
    hold-out already measures, and applying both charges parsimony twice. On a
    scan whose whole axis is sparsity, that bias points straight down the axis.
  - Mean number of active SINDy coefficients per participant

Usage:
    python weinhardt2026/analysis/analysis_sparsity_hpscan.py
"""

import os
import re
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim
from weinhardt2026.analysis.analysis_model_evaluation import (
    get_participant_experiment_groups,
    grouped_information_criteria,
)


@torch.no_grad()
def analysis_sparsity_hpscan(
    pkl_pattern,
    spice_class,
    spice_config,
    n_actions,
    data_path,
    test_blocks,
    polynomial_degree=2,
    model_kwargs=None,
    device=None,
):
    """Evaluate all HP scan checkpoints and return a summary DataFrame.

    Parameters
    ----------
    pkl_pattern : str
        Glob pattern matching HP scan .pkl files (e.g.
        'params_array/spice_dezfouli2019_*_*.pkl').
    spice_class : type
        Model class (e.g. workingmemory.SpiceModel).
    spice_config : SpiceConfig
        Model configuration.
    n_actions : int
        Number of actions.
    data_path : str
        Path to the dataset CSV.
    test_blocks : tuple[int]
        Block IDs for hold-out evaluation.
    polynomial_degree : int
        SINDy library polynomial degree.
    model_kwargs : dict, optional
        Extra kwargs for the model constructor.
    device : torch.device, optional
        Compute device (default: auto-detect).

    Returns
    -------
    pd.DataFrame
        Rows = HP configurations, columns include threshold, test,
        trial_likelihood, NLL, BIC(_std), AIC(_std), delta_bic_per_trial(_std)
        (all computed in-sample on training data), plus trial_likelihood_test,
        NLL_test, generalization_gap on test_blocks, and n_params_mean,
        n_params_std. BIC/AIC are computed per (participant, experiment) group
        and reported as mean ± std across groups. Select on the training
        criteria; read the hold-out columns as confirmation only.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load and split data ───────────────────────────────────────────
    dataset = csv_to_dataset(file=data_path)
    dataset.normalize_rewards()
    dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)

    xs_test = dataset_test.xs.to(device)
    ys_test = dataset_test.ys.cpu()
    valid = ~torch.isnan(dataset_test.xs[:, :, 0, 0])
    n_valid = valid.sum().item()
    n_trials_per_session = valid.sum(dim=1).float()

    unique_pairs, group_index = get_participant_experiment_groups(dataset_test)
    n_groups = unique_pairs.shape[0]

    # ── Training data (for BIC/AIC, which should reflect in-sample fit) ─
    xs_train = dataset_train.xs.to(device)
    ys_train = dataset_train.ys.cpu()
    valid_train = ~torch.isnan(dataset_train.xs[:, :, 0, 0])
    n_valid_train = valid_train.sum().item()
    n_trials_per_session_train = valid_train.sum(dim=1).float()

    unique_pairs_train, group_index_train = get_participant_experiment_groups(dataset_train)
    n_groups_train = unique_pairs_train.shape[0]

    # ── Find and parse checkpoint files ───────────────────────────────
    pkl_paths = sorted(glob(pkl_pattern))
    # Filter out stage1 and stability files
    pkl_paths = [p for p in pkl_paths
                 if 'stage1' not in p
                 and 'stability' not in p
                 and not p.endswith('spice_dezfouli2019.pkl')
                 and not p.endswith('spice_dezfouli2019_test.pkl')
                 and not p.endswith('spice_dezfouli2019_2.pkl')]

    if not pkl_paths:
        print(f"No HP scan checkpoints found matching {pkl_pattern}")
        return pd.DataFrame()

    # Parse threshold and test from filename
    # Pattern: spice_dezfouli2019_{threshold}_{test}.pkl
    def parse_hp(path):
        basename = os.path.basename(path).replace('.pkl', '')
        # Match the last two numeric segments: {threshold}_{test}
        match = re.search(r'_(\d+\.?\d*)_(\d+\.?\d*)$', basename)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None

    # ── Evaluate each checkpoint ──────────────────────────────────────
    rows = []
    for path in pkl_paths:
        threshold, test_val = parse_hp(path)
        if threshold is None:
            print(f"  Skipping (cannot parse HP from filename): {path}")
            continue

        print(f"  Evaluating threshold={threshold}, test={test_val} ...")

        # Load checkpoint to get ensemble size and n_participants
        ckpt = torch.load(path, map_location='cpu')
        first_mod = next(iter(spice_config.library_setup))
        ensemble_size = ckpt['model'][f'sindy_concept_loadings.{first_mod}'].shape[0]
        n_participants = ckpt['model'][f'sindy_concept_loadings.{first_mod}'].shape[1]
        del ckpt

        estimator = SpiceEstimator(
            spice_class=spice_class,
            spice_config=spice_config,
            n_actions=n_actions,
            n_participants=n_participants,
            sindy_library_polynomial_degree=polynomial_degree,
            ensemble_size=ensemble_size,
            use_sindy=True,
            kwargs_spice_class=model_kwargs or {},
            device=device,
        )
        estimator.load_spice(path)

        # ── Coefficient count (per participant/experiment group actually in the test set) ──
        n_params = estimator.count_spice_parameters()['loadings']  # (P, X)
        n_params_per_group = n_params[unique_pairs[:, 0], unique_pairs[:, 1]].float()
        n_params_mean = n_params_per_group.mean().item()
        n_params_std = n_params_per_group.std().item() if n_params_per_group.numel() > 1 else 0.0
        n_params_per_group_train = n_params[unique_pairs_train[:, 0], unique_pairs_train[:, 1]].float()

        # ── Hold-out trial likelihood (SINDy autoregressive) ──────────
        estimator.model.eval()
        estimator.use_sindy(True)
        logits, _ = estimator.model(xs_test)
        # (E, B, T, W, A) → ensemble mean → softmax
        probs = torch.softmax(logits.mean(dim=0), dim=-1).cpu()

        eps = 1e-9
        probs = probs.clamp(eps, 1 - eps)
        ll = (ys_test * torch.log(probs)).sum(dim=-1).sum(dim=-1)  # (B, T)
        ll = ll.where(valid, torch.tensor(float('nan')))

        nll = -torch.nansum(ll).item()
        trial_lik = np.exp(-nll / n_valid)

        # BIC/AIC computed per (participant, experiment) group -- using that
        # group's own trial count and own coefficient count -- then averaged
        # across groups. Pooling NLL across the whole test set while scaling
        # k with the number of participants (as is correct here, since each
        # participant has independently active SINDy coefficients) would make
        # the penalty grow with dataset size regardless of fit quality; see
        # grouped_information_criteria for the full rationale.
        nll_per_session = (-ll).nansum(dim=1)  # (B,)
        info = grouped_information_criteria(
            nll_per_session=nll_per_session,
            n_trials_per_session=n_trials_per_session,
            group_index=group_index,
            n_groups=n_groups,
            n_parameters_per_group=n_params_per_group,
            n_actions_baseline=n_actions,
        )

        # ── In-sample likelihood on training data (for BIC/AIC) ────────
        logits_train, _ = estimator.model(xs_train)
        probs_train = torch.softmax(logits_train.mean(dim=0), dim=-1).cpu()
        probs_train = probs_train.clamp(eps, 1 - eps)
        ll_train = (ys_train * torch.log(probs_train)).sum(dim=-1).sum(dim=-1)  # (B, T)
        ll_train = ll_train.where(valid_train, torch.tensor(float('nan')))
        nll_per_session_train = (-ll_train).nansum(dim=1)  # (B,)

        nll_train = -torch.nansum(ll_train).item()
        trial_lik_train = np.exp(-nll_train / n_valid_train)

        info_train = grouped_information_criteria(
            nll_per_session=nll_per_session_train,
            n_trials_per_session=n_trials_per_session_train,
            group_index=group_index_train,
            n_groups=n_groups_train,
            n_parameters_per_group=n_params_per_group_train,
            n_actions_baseline=n_actions,
        )

        # Information criteria on the training split only; the hold-out is
        # reported as likelihood/NLL. A BIC on held-out data double-charges
        # parsimony (see `analysis_model_evaluation`), which on a sparsity scan
        # is exactly the wrong thumb on the scale -- it rewards the very axis
        # the scan is varying.
        rows.append({
            'threshold': threshold,
            'test': test_val,
            'n_params_mean': n_params_mean,
            'n_params_std': n_params_std,
            'trial_likelihood': trial_lik_train,
            'NLL': nll_train,
            'BIC': info_train['bic_mean'],
            'BIC_std': info_train['bic_std'],
            'AIC': info_train['aic_mean'],
            'AIC_std': info_train['aic_std'],
            'delta_bic_per_trial': info_train['delta_bic_per_trial_mean'],
            'delta_bic_per_trial_std': info_train['delta_bic_per_trial_std'],
            'trial_likelihood_test': trial_lik,
            'NLL_test': nll,
            'generalization_gap': trial_lik_train - trial_lik,
            'path': os.path.basename(path),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(['threshold', 'test']).reset_index(drop=True)
    return df


def plot_hpscan_heatmaps(df, output_path):
    """Save a 2×3 grid of heatmaps over the threshold × test HP grid.

    Top row is the training split, where the selection criteria live: parameter
    count, trial likelihood, and BIC. Bottom row is the hold-out, where only
    likelihood-based quantities are meaningful -- parameter count (identical, so
    repeated as a reference), trial likelihood, and the generalization gap. The
    bottom-right panel deliberately shows the gap rather than a hold-out BIC;
    see this module's docstring.
    """
    panels = [
        [('n_params_mean', 'Parameter Count'), ('trial_likelihood', 'Trial Likelihood'), ('BIC', 'BIC')],
        [('n_params_mean', 'Parameter Count'), ('trial_likelihood_test', 'Trial Likelihood'),
         ('generalization_gap', 'Generalization Gap')],
    ]
    split_labels = ['Training', 'Hold-out']

    thresholds = sorted(df['threshold'].unique())
    test_vals = sorted(df['test'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, split_label in enumerate(split_labels):
        for col, (column, metric_label) in enumerate(panels[row]):
            ax = axes[row, col]
            pivot = df.pivot(index='threshold', columns='test', values=column)
            pivot = pivot.reindex(index=thresholds, columns=test_vals)

            im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
            ax.set_xticks(range(len(test_vals)))
            ax.set_xticklabels(test_vals)
            ax.set_yticks(range(len(thresholds)))
            ax.set_yticklabels(thresholds)
            ax.set_xlabel('test')
            ax.set_ylabel('threshold')
            ax.set_title(f'{split_label}: {metric_label}')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.values[i, j]
                    if not np.isnan(value):
                        ax.text(j, i, f'{value:.3g}', ha='center', va='center',
                                 color='white', fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Standalone execution ─────────────────────────────────────────────

if __name__ == '__main__':

    study = 'eckstein2026'
    from weinhardt2026.studies.eckstein2026.spice_eckstein2026 import SpiceModel, CONFIG
    n_actions = 4
    test_blocks = (2,)

    # study = 'dezfouli2019'
    # from spice.precoded.workingmemory import SpiceModel, CONFIG
    # n_actions = 2
    # test_blocks = (3,6,9)
    
    model_kwargs = {'reward_binary': True}
    # Dataset
    data_path = f'weinhardt2026/studies/{study}/data/{study}.csv'
    params_dir = f'weinhardt2026/studies/{study}/params_array'
    pkl_pattern = os.path.join(params_dir, f'spice_{study}_*.pkl')

    print(f"Hyperparameter scan analysis: {study}")
    print("=" * 60)

    df = analysis_sparsity_hpscan(
        pkl_pattern=pkl_pattern,
        spice_class=SpiceModel,
        spice_config=CONFIG,
        n_actions=n_actions,
        data_path=data_path,
        test_blocks=test_blocks,
        polynomial_degree=2,
        model_kwargs=model_kwargs,
    )

    if len(df) > 0:
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(df.to_string(index=False, float_format='{:.4f}'.format))

        # Save results
        results_dir = f'weinhardt2026/studies/{study}/results'
        output_path = os.path.join(results_dir, 'hpscan_results.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")

        # Save heatmap figure
        figure_path = os.path.join(results_dir, 'hpscan_heatmaps.png')
        plot_hpscan_heatmaps(df, figure_path)
        print(f"Saved heatmaps to {figure_path}")
