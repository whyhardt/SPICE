"""Run morphing analysis for dezfouli2019 study.

Morphs participant embeddings along the avg_reward direction independently for
each ensemble member, refits SINDy equations, generates behavior to validate
the morphing direction, and reports whether avg_reward increases monotonically
along the morphing axis.

Usage:
    python -m weinhardt2026.studies.dezfouli2019.run_morphing
    python -m weinhardt2026.studies.dezfouli2019.run_morphing --n_steps 20 --range_sd 1.5
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from spice import SpiceEstimator, csv_to_dataset
from spice.precoded import workingmemory

from weinhardt2026.analysis.analysis_morphing import run_morphing, get_morphed_coefficients
from weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019 import (
    EnvironmentDezfouli2019,
    get_dataset,
)
from weinhardt2026.utils.task import generate_behavior


# ── Paths ──
STUDY_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(STUDY_DIR, 'data', 'dezfouli2019.csv')
PARAMS_PATH = os.path.join(STUDY_DIR, 'params', 'spice_dezfouli2019.pkl')
METRICS_PATH = os.path.join(STUDY_DIR, 'results', 'behavioral_metrics_real.csv')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')


def _load_estimator():
    """Load fitted SpiceEstimator."""
    ckpt = torch.load(PARAMS_PATH, map_location='cpu')
    first_mod = next(iter(workingmemory.CONFIG.library_setup))
    ensemble_size = ckpt['model'][f'sindy_coefficients.{first_mod}'].shape[0]
    n_participants = ckpt['model'][f'sindy_coefficients.{first_mod}'].shape[1]
    del ckpt

    estimator = SpiceEstimator(
        spice_class=workingmemory.SpiceModel,
        spice_config=workingmemory.CONFIG,
        n_actions=2,
        n_participants=n_participants,
        sindy_library_polynomial_degree=2,
        ensemble_size=ensemble_size,
        use_sindy=True,
    )
    estimator.load_spice(PARAMS_PATH)
    return estimator


def _compute_avg_reward_from_dataset(dataset, n_participants):
    """Compute average reward per participant from a SpiceDataset."""
    xs = dataset.xs  # (B, T, W, F)
    n_actions = dataset.n_actions

    avg_rewards = []
    for pid in range(n_participants):
        pid_mask = xs[:, 0, 0, -1].long() == pid
        sessions = xs[pid_mask]  # (n_blocks, T, W, F)

        # Rewards are in columns n_actions:2*n_actions; take chosen reward
        rewards = sessions[:, :, 0, n_actions:2 * n_actions]  # (n_blocks, T, n_actions)
        # Chosen reward = nanmean of all non-NaN reward entries
        valid = ~torch.isnan(rewards)
        if valid.any():
            avg_rewards.append(rewards[valid].mean().item())
        else:
            avg_rewards.append(float('nan'))

    return np.array(avg_rewards)


def _validate_morphing(morphed_estimator, morphed_dataset, n_participants, n_steps):
    """Generate behavior with morphed model and check avg_reward per morphing step.

    Returns:
        avg_reward_per_step: (n_steps,) mean avg_reward across participants at each step.
        avg_reward_per_step_std: (n_steps,) std across participants at each step.
    """
    n_virtual = n_participants * n_steps

    # Generate behavior using the morphed model + morphed dataset
    environment = EnvironmentDezfouli2019(
        n_actions=2,
        n_participants=n_virtual,
        n_blocks=12,
    )

    # Use RNN mode for generation (more stable than SINDy for generative)
    morphed_estimator.model.eval(use_sindy=False)

    dataset_gen = generate_behavior(
        dataset=morphed_dataset,
        model=morphed_estimator,
        environment=environment,
    )

    # Compute avg_reward per virtual participant
    avg_rewards_virtual = _compute_avg_reward_from_dataset(dataset_gen, n_virtual)

    # Reshape to (P, M) and average across participants
    avg_rewards_matrix = avg_rewards_virtual.reshape(n_participants, n_steps)
    avg_reward_per_step = np.nanmean(avg_rewards_matrix, axis=0)
    avg_reward_per_step_std = np.nanstd(avg_rewards_matrix, axis=0)

    return avg_reward_per_step, avg_reward_per_step_std


def main():
    parser = argparse.ArgumentParser(description='Run morphing analysis for dezfouli2019')
    parser.add_argument('--n_steps', type=int, default=20, help='Number of morphing steps')
    parser.add_argument('--range_sd', type=float, default=1.0,
                        help='Morphing range in SD of embedding projections')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip behavior generation validation')
    parser.add_argument('--n_validation_runs', type=int, default=5,
                        help='Number of validation runs (averaged for stability)')
    args = parser.parse_args()

    # ── Load data ──
    print("Loading estimator and data...")
    estimator = _load_estimator()

    dataset = csv_to_dataset(file=DATA_PATH)
    dataset.normalize_rewards()

    df_metrics = pd.read_csv(METRICS_PATH)
    metric_values = df_metrics['avg_reward'].values

    print(f"  Participants: {estimator.n_participants}")
    print(f"  Ensemble size: {estimator.model.ensemble_size}")
    print(f"  Metric (avg_reward) range: [{metric_values.min():.4f}, {metric_values.max():.4f}]")

    # ── Run morphing (all ensemble members) ──
    save_dir = os.path.join(STUDY_DIR, 'params', 'morphing')

    result = run_morphing(
        estimator=estimator,
        dataset=dataset,
        metric_values=metric_values,
        n_steps=args.n_steps,
        morphing_range_sd=args.range_sd,
        save_dir=save_dir,
        verbose=True,
    )

    # ── Extract and aggregate morphed coefficients ──
    print("\nExtracting morphed coefficients (aggregated across ensemble members)...")
    morph_coeffs = get_morphed_coefficients(result)

    E = result['n_ensemble_members']
    for module, data in morph_coeffs.items():
        n_active = (data['inclusion_probability'] > 0).sum(axis=1)
        print(f"  {module}: {len(data['term_names'])} terms, "
              f"active per step: [{n_active.min():.0f}, {n_active.max():.0f}]")

    # Save aggregated morphed coefficients for plotting
    coeffs_path = os.path.join(RESULTS_DIR, 'morphing_coefficients.npz')
    step_values = morph_coeffs[list(morph_coeffs.keys())[0]]['step_values']
    np.savez(
        coeffs_path,
        step_values=step_values,
        n_steps=result['n_steps'],
        n_participants=result['n_participants'],
        n_ensemble_members=E,
        metric_values=result['metric_values'],
        **{f'{m}_mean_coefficients': d['mean_coefficients'] for m, d in morph_coeffs.items()},
        **{f'{m}_se_coefficients': d['se_coefficients'] for m, d in morph_coeffs.items()},
        **{f'{m}_inclusion_probability': d['inclusion_probability'] for m, d in morph_coeffs.items()},
        **{f'{m}_se_inclusion_probability': d['se_inclusion_probability'] for m, d in morph_coeffs.items()},
        **{f'{m}_term_names': d['term_names'] for m, d in morph_coeffs.items()},
        **{f'{m}_all_coefficients': d['all_coefficients'] for m, d in morph_coeffs.items()},
        **{f'{m}_all_inclusion_probability': d['all_inclusion_probability'] for m, d in morph_coeffs.items()},
    )
    print(f"  Saved morphed coefficients to {coeffs_path}")

    # ── Validate: generate behavior and check avg_reward monotonicity ──
    if not args.skip_validation:
        print(f"\nValidation: generating behavior ({args.n_validation_runs} runs)...")
        print("  Using ensemble member 0 for validation.")

        # Use member 0's morphed estimator for validation
        member_0 = result['member_results'][0]
        morphed_estimator = member_0['estimator']

        from weinhardt2026.analysis.analysis_morphing import _create_morphed_dataset
        morphed_dataset = _create_morphed_dataset(dataset, estimator.n_participants, args.n_steps)

        all_avg_rewards = []
        for run_i in range(args.n_validation_runs):
            print(f"  Run {run_i + 1}/{args.n_validation_runs}...")
            avg_per_step, _ = _validate_morphing(
                morphed_estimator, morphed_dataset,
                result['n_participants'], result['n_steps'],
            )
            all_avg_rewards.append(avg_per_step)

        avg_rewards_runs = np.stack(all_avg_rewards)  # (n_runs, n_steps)
        avg_reward_mean = avg_rewards_runs.mean(axis=0)
        avg_reward_se = avg_rewards_runs.std(axis=0) / np.sqrt(args.n_validation_runs)

        # Check monotonicity
        diffs = np.diff(avg_reward_mean)
        n_increasing = (diffs > 0).sum()
        n_total = len(diffs)
        spearman_r = np.corrcoef(np.arange(args.n_steps), avg_reward_mean)[0, 1]

        print(f"\n  Validation results:")
        print(f"    Avg reward range: [{avg_reward_mean.min():.4f}, {avg_reward_mean.max():.4f}]")
        print(f"    Monotonicity: {n_increasing}/{n_total} steps increasing "
              f"({100 * n_increasing / n_total:.0f}%)")
        print(f"    Correlation (step vs avg_reward): r = {spearman_r:.3f}")

        print(f"\n    Step-by-step avg_reward (mean ± SE across {args.n_validation_runs} runs):")
        for i, (sv, ar, se) in enumerate(zip(step_values, avg_reward_mean, avg_reward_se)):
            bar = '█' * int(ar * 200)
            print(f"      step {i+1:2d} (proj={sv:+.2f}): {ar:.4f} ± {se:.4f}  {bar}")

        # Save validation results
        val_path = os.path.join(RESULTS_DIR, 'morphing_validation.npz')
        np.savez(
            val_path,
            step_values=step_values,
            avg_reward_mean=avg_reward_mean,
            avg_reward_se=avg_reward_se,
            avg_rewards_all_runs=avg_rewards_runs,
        )
        print(f"\n  Saved validation results to {val_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
