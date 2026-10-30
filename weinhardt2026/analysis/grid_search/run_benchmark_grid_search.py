"""Aggregate the benchmark/GRU grid search (benchmark_grid_search.py) over seeds and pick the best grid point.

For every study x model kind x grid point (all hyperparameters except the seed), the test loss
(NLL or MSE, depending on the study) is aggregated over seeds by its median. The grid point with the
lowest median test loss is selected; for it, the seed with the lowest test loss and the median seed
are reported (the checkpoints to use for that study).

Writes (overwriting):
    <out_dir>/grid_search_configs.csv   one row per grid point
    <out_dir>/grid_search_best.csv      one row per study x model kind

Usage (from the repository root):
    python weinhardt2026/benchmark_grid_search/aggregate_grid_search.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_RUNS_DIR = 'weinhardt2026/benchmark_grid_search/runs'
DEFAULT_OUT_DIR = 'weinhardt2026/benchmark_grid_search/results'
CONFIG_COLUMNS = ['study', 'kind', 'lr', 'epochs', 'scheduler', 'hidden_size', 'dropout']


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = [json.loads(path.read_text()) | {'name': path.stem} for path in sorted(runs_dir.glob('*/*.json'))]
    if not rows:
        raise FileNotFoundError(f"No run results found in {runs_dir}")
    return pd.DataFrame(rows)


def median_run(group: pd.DataFrame) -> str:
    """Run at the (lower) median test loss of a grid point."""
    ordered = group.sort_values('test_loss')
    return ordered['name'].iloc[(len(ordered) - 1) // 2]


def aggregate_configs(runs: pd.DataFrame) -> pd.DataFrame:
    # benchmarks have no hidden_size/dropout; groupby drops NaN keys, so group on a string key
    runs = runs.assign(config=runs[CONFIG_COLUMNS].astype(str).agg('|'.join, axis=1))
    rows = []
    for _, group in runs.groupby('config'):
        rows.append({
            **group[CONFIG_COLUMNS].iloc[0].to_dict(),
            'test_loss_median': group['test_loss'].median(),
            'test_loss_mean': group['test_loss'].mean(),
            'test_loss_std': group['test_loss'].std(),
            'train_loss_median': group['train_loss'].median(),
            'n_seeds': group['seed'].nunique(),
            'n_parameters': group['n_parameters'].iloc[0],
            'duration_sec': group['duration_sec'].mean(),
            'best_run': group.sort_values('test_loss')['name'].iloc[0],
            'median_run': median_run(group),
        })
    return pd.DataFrame(rows).sort_values(['study', 'kind', 'test_loss_median'])


def select_best(configs: pd.DataFrame) -> pd.DataFrame:
    return configs.loc[configs.groupby(['study', 'kind'])['test_loss_median'].idxmin()].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Aggregate the benchmark/GRU grid search over seeds.')
    parser.add_argument('--runs_dir', default=DEFAULT_RUNS_DIR)
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = aggregate_configs(load_runs(Path(args.runs_dir)))
    best = select_best(configs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs.to_csv(out_dir / 'grid_search_configs.csv', index=False)
    best.to_csv(out_dir / 'grid_search_best.csv', index=False)

    incomplete = configs[configs['n_seeds'] < configs['n_seeds'].max()]
    if len(incomplete):
        print(f"WARNING: {len(incomplete)} grid points have fewer than {configs['n_seeds'].max()} seeds")

    with pd.option_context('display.max_columns', None, 'display.width', 250):
        print(best[CONFIG_COLUMNS + ['test_loss_median', 'test_loss_std', 'n_seeds', 'best_run', 'median_run']])
    print(f"\nWritten to {out_dir}")


if __name__ == '__main__':
    main()
