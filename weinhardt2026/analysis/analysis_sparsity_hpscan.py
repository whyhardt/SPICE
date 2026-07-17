"""Analyze hyperparameter scan over pruning_threshold × pruning_test.

For each checkpoint, computes:
  - Hold-out trial likelihood (SINDy autoregressive on test blocks)
  - Mean number of active SINDy coefficients per participant

Usage:
    python -m weinhardt2026.analysis.analysis_sparsity_hpscan
"""

import os
import re
from glob import glob

import numpy as np
import pandas as pd
import torch

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim


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
        trial_likelihood, NLL, n_params_mean, n_params_std.
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

        # ── Coefficient count ─────────────────────────────────────────
        n_params = estimator.count_sindy_coefficients()  # (P, X)
        n_params_mean = n_params.float().mean().item()
        n_params_std = n_params.float().std().item()

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

        # BIC = 2*NLL + k*ln(n)
        bic = 2 * nll + n_params_mean * np.log(n_valid)

        rows.append({
            'threshold': threshold,
            'test': test_val,
            'trial_likelihood': trial_lik,
            'NLL': nll,
            'BIC': bic,
            'n_params_mean': n_params_mean,
            'n_params_std': n_params_std,
            'path': os.path.basename(path),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(['threshold', 'test']).reset_index(drop=True)
    return df


# ── Standalone execution ─────────────────────────────────────────────

if __name__ == '__main__':
    from spice.precoded import workingmemory

    params_dir = 'weinhardt2026/studies/dezfouli2019/params_array'
    pkl_pattern = os.path.join(params_dir, 'spice_dezfouli2019_*.pkl')

    print("Hyperparameter scan analysis: dezfouli2019")
    print("=" * 60)

    df = analysis_sparsity_hpscan(
        pkl_pattern=pkl_pattern,
        spice_class=workingmemory.SpiceModel,
        spice_config=workingmemory.CONFIG,
        n_actions=2,
        data_path='weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv',
        test_blocks=(3, 6, 9),
        polynomial_degree=2,
        model_kwargs={'reward_binary': True},
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
