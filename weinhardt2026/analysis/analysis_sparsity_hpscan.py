"""Analyze hyperparameter scan over pruning_threshold × pruning_test.

For each checkpoint, computes:
  - Hold-out trial likelihood (SINDy autoregressive on test blocks)
  - Mean number of active SINDy coefficients per participant

Usage:
    python weinhardt2026/analysis/analysis_sparsity_hpscan.py
"""

import os
import re
import sys
from glob import glob
from pathlib import Path

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
        trial_likelihood, NLL, BIC(_std), AIC(_std), delta_bic_per_trial(_std),
        n_params_mean, n_params_std. BIC/AIC are computed per (participant,
        experiment) group and reported as mean ± std across groups.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load and split data ───────────────────────────────────────────
    dataset = csv_to_dataset(file=data_path)
    dataset.normalize_rewards()
    _, dataset_test = split_data_along_blockdim(dataset, test_blocks)

    xs_test = dataset_test.xs.to(device)
    ys_test = dataset_test.ys.cpu()
    valid = ~torch.isnan(dataset_test.xs[:, :, 0, 0])
    n_valid = valid.sum().item()
    n_trials_per_session = valid.sum(dim=1).float()

    unique_pairs, group_index = get_participant_experiment_groups(dataset_test)
    n_groups = unique_pairs.shape[0]

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
        ensemble_size = ckpt['model'][f'sindy_coefficients.{first_mod}'].shape[0]
        n_participants = ckpt['model'][f'sindy_coefficients.{first_mod}'].shape[1]
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
        n_params = estimator.count_sindy_coefficients()  # (P, X)
        n_params_per_group = n_params[unique_pairs[:, 0], unique_pairs[:, 1]].float()
        n_params_mean = n_params_per_group.mean().item()
        n_params_std = n_params_per_group.std().item() if n_params_per_group.numel() > 1 else 0.0

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

        rows.append({
            'threshold': threshold,
            'test': test_val,
            'n_params_mean': n_params_mean,
            'n_params_std': n_params_std,
            'trial_likelihood': trial_lik,
            'NLL': nll,
            'BIC': info['bic_mean'],
            'BIC_std': info['bic_std'],
            'AIC': info['aic_mean'],
            'AIC_std': info['aic_std'],
            'delta_bic_per_trial': info['delta_bic_per_trial_mean'],
            'delta_bic_per_trial_std': info['delta_bic_per_trial_std'],
            'path': os.path.basename(path),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(['threshold', 'test']).reset_index(drop=True)
    return df


# ── Standalone execution ─────────────────────────────────────────────

if __name__ == '__main__':
    
    # SPICE
    from spice.precoded.workingmemory import SpiceModel, CONFIG
    # from weinhardt2026.studies.dezfouli2019.spice_dezfouli2019 import SpiceModel, CONFIG
    model_kwargs = {'reward_binary': True}
    # Dataset
    data_path = 'weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv'
    n_actions = 2
    test_blocks = (3,6,9)
    
    params_dir = 'weinhardt2026/studies/dezfouli2019/params_array'
    pkl_pattern = os.path.join(params_dir, 'spice_dezfouli2019_*.pkl')

    print("Hyperparameter scan analysis: dezfouli2019")
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
        output_path = os.path.join(
            'weinhardt2026/studies/dezfouli2019/results',
            'hpscan_results.csv',
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
