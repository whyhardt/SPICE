"""Fit a SPICE model to dezfouli2019 with the RNN bypassed entirely.

SINDy coefficients are optimized straight against the behavioural
cross-entropy (Stage 1, with L1 + pruning to discover the sparsity pattern),
then refitted on the frozen support with no penalty and no pruning (Stage 2).
See `spice.resources.spice_direct_training` for the method.

Uses `spice.precoded.workingmemory` -- the same architecture as the
`params_array/` hyperparameter scan, so the numbers here are directly
comparable to the RNN-mediated checkpoints evaluated by
`weinhardt2026/analysis/analysis_sparsity_hpscan.py`. (Note that
`dezfouli2019.py` itself currently imports `spice.precoded.choice`, a
different and smaller model.)

Pruning strength is set from the command line so several settings can coexist;
outputs are keyed by (threshold, ensemble ratio) and never overwrite each other.

Usage:
    python weinhardt2026/studies/dezfouli2019/dezfouli2019_direct.py
    python weinhardt2026/studies/dezfouli2019/dezfouli2019_direct.py --threshold 0.01 --ensemble-ratio 0.5
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim
from spice.precoded.workingmemory import SpiceModel, CONFIG
from spice.resources.spice_direct_training import fit_spice_direct
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STUDY = 'dezfouli2019'
TEST_BLOCKS = (3, 6, 9)
ENSEMBLE_SIZE = 8
POLYNOMIAL_DEGREE = 2

PATH_DATA = f'weinhardt2026/studies/{STUDY}/data/{STUDY}.csv'

TRAINING_KWARGS = dict(
    epochs_stage1=300,
    epochs_stage2=150,
    n_warmup_epochs=50,
    learning_rate=0.01,
    learning_rate_stage2=0.003,
    n_steps=50,               # BPTT truncation over 202-trial sessions
    sindy_alpha=1e-3,
    sindy_pruning_frequency=10,
    sindy_pruning_patience=2,
    check_every=10,
)


def plot_training_history(history: pd.DataFrame, output_path: str) -> None:
    """Four panels tracking whether the fit converged and settled:
    likelihood, sparsity, coefficient drift, and ensemble agreement.
    The Stage 1 -> Stage 2 boundary is marked in each.
    """
    history = history.reset_index(drop=True)
    x = history.index.values
    boundary = (history['stage'] == 'stage1').sum() - 0.5

    panels = [
        (['train_trial_lik', 'test_per_member', 'test_logit_avg', 'test_coef_avg'],
         'trial likelihood', 'Fit (three scorings)'),
        (['n_active_mean'], 'active coefficients / participant', 'Sparsity'),
        (['coef_drift'], 'mean |Δ coefficient|', 'Stability (drift)'),
        (['ensemble_cv', 'support_agree'], 'ratio', 'Ensemble consistency'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (columns, ylabel, title) in zip(axes.ravel(), panels):
        for column in columns:
            ax.plot(x, history[column].values, marker='o', markersize=3, label=column)
        ax.axvline(boundary, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        if history['is_best'].any():
            ax.axvline(history.index[history['is_best']][-1], color='tab:green',
                       linestyle=':', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('check index (stage 1 -> stage 2)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if 'coef_drift' in history:
        axes.ravel()[2].set_yscale('log')

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='|coefficient| below which a term counts as inactive at a pruning event')
    parser.add_argument('--ensemble-ratio', type=float, default=0.7,
                        help='fraction of ensemble members that must support a term for it to survive')
    args = parser.parse_args()

    tag = f'{args.threshold:g}_{args.ensemble_ratio:g}'
    path_model = f'weinhardt2026/studies/{STUDY}/params/spice_{STUDY}_direct_{tag}.pkl'
    output_dir = f'weinhardt2026/studies/{STUDY}/results/direct_{tag}'

    TRAINING_KWARGS['sindy_threshold_pruning'] = args.threshold
    TRAINING_KWARGS['sindy_ensemble_pruning'] = args.ensemble_ratio

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(path_model), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- data ---------------------------------------------------------------
    dataset = csv_to_dataset(file=PATH_DATA)
    dataset.normalize_rewards()
    dataset_train, dataset_test = split_data_along_blockdim(dataset, TEST_BLOCKS)
    n_participants = int(dataset.xs[:, 0, 0, -1].max().item()) + 1

    print(f"Dataset: {tuple(dataset.xs.shape)} | participants: {n_participants} | "
          f"train sessions: {dataset_train.xs.shape[0]} | test sessions: {dataset_test.xs.shape[0]}")

    # -- estimator ----------------------------------------------------------
    estimator = SpiceEstimator(
        spice_class=SpiceModel,
        spice_config=CONFIG,
        n_actions=dataset.n_actions,
        n_participants=n_participants,
        sindy_library_polynomial_degree=POLYNOMIAL_DEGREE,
        ensemble_size=ENSEMBLE_SIZE,
        use_sindy=True,
        kwargs_spice_class={'reward_binary': True},
        device=device,
        verbose=True,
    )

    # -- fit ----------------------------------------------------------------
    model, history = fit_spice_direct(
        model=estimator.model,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        verbose=True,
        **TRAINING_KWARGS,
    )

    history.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))

    estimator.save_spice(path_model)
    print(f"\nModel saved to {path_model}")

    # -- example equations --------------------------------------------------
    estimator.eval()
    equations_path = os.path.join(output_dir, 'equations_examples.txt')
    with open(equations_path, 'w') as f:
        for pid in range(3):
            f.write(f"--- participant {pid} ---\n")
            f.write(estimator.model.get_spice_model_string(participant_id=pid, experiment_id=0))
            f.write("\n\n")
    for pid in range(3):
        print(f"\nExample SPICE model (participant {pid}):")
        estimator.print_spice_model(participant_id=pid)

    # -- evaluation ---------------------------------------------------------
    print("\n--- Model evaluation (train) ---")
    results_train = analysis_model_evaluation(dataset=dataset_train, spice_model=estimator)
    print(results_train)

    print("\n--- Model evaluation (test) ---")
    results_test = analysis_model_evaluation(dataset=dataset_test, spice_model=estimator)
    print(results_test)

    results_train.to_csv(os.path.join(output_dir, 'model_evaluation_train.csv'), index=False)
    results_test.to_csv(os.path.join(output_dir, 'model_evaluation_test.csv'), index=False)
    print(f"\nAll results saved to {output_dir}")
