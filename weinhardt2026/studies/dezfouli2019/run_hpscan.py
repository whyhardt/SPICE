"""Rerun the sparsity hyperparameter scan over dezfouli2019/params_array/.

Writes results/hpscan_results.csv and results/hpscan_heatmaps.png, which
analysis_bestbic.py reads to pick the lowest-BIC checkpoint.

Usage:
    python -m weinhardt2026.studies.dezfouli2019.run_hpscan
"""

import os

from spice.precoded.workingmemory import SpiceModel, CONFIG
from weinhardt2026.analysis.analysis_sparsity_hpscan import (
    analysis_sparsity_hpscan,
    plot_hpscan_heatmaps,
)

STUDY = 'dezfouli2019'
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
N_ACTIONS = 2
TEST_BLOCKS = (3, 6, 9)
MODEL_KWARGS = {'reward_binary': True}


def main():
    results_dir = os.path.join(STUDY_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print(f"Hyperparameter scan analysis: {STUDY}")
    print("=" * 60)

    df = analysis_sparsity_hpscan(
        pkl_pattern=os.path.join(STUDY_DIR, 'params_array', f'spice_{STUDY}_*.pkl'),
        spice_class=SpiceModel,
        spice_config=CONFIG,
        n_actions=N_ACTIONS,
        data_path=os.path.join(STUDY_DIR, 'data', f'{STUDY}.csv'),
        test_blocks=TEST_BLOCKS,
        polynomial_degree=2,
        model_kwargs=MODEL_KWARGS,
    )

    if len(df) == 0:
        print("No checkpoints matched.")
        return

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(df.to_string(index=False, float_format='{:.4f}'.format))

    output_path = os.path.join(results_dir, 'hpscan_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    figure_path = os.path.join(results_dir, 'hpscan_heatmaps.png')
    plot_hpscan_heatmaps(df, figure_path)
    print(f"Saved heatmaps to {figure_path}")


if __name__ == '__main__':
    main()
