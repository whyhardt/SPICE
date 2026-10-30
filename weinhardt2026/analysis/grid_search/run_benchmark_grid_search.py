"""Aggregate the benchmark/GRU grid search (benchmark_grid_search.py) over seeds and pick the best grid point.

For every study x model kind x grid point (all hyperparameters except the seed), the test loss
(NLL or MSE, depending on the study) is aggregated over seeds by its median. The grid point with the
lowest median test loss is selected; for it, the seed with the lowest test loss and the median seed
are reported (the checkpoints to use for that study).

Writes (overwriting):
    <out_dir>/grid_search_configs.csv   one row per grid point
    <out_dir>/grid_search_best.csv      one row per study x model kind
    <out_dir>/figure_benchmark_grid_search[_clean].pdf/.png
                                        per study: every grid point's median test loss (± SD over seeds) per
                                        model kind; the selected grid point in red

Usage (from the repository root):
    python weinhardt2026/analysis/grid_search/run_benchmark_grid_search.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUNS_DIR = 'weinhardt2026/analysis/grid_search/runs'
DEFAULT_OUT_DIR = 'weinhardt2026/results/benchmark_grid_search'
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


KIND_COLORS = {'benchmark': '#2f6f9f', 'gru': '#4d4d4d'}


def plot_grid_search(configs: pd.DataFrame, best: pd.DataFrame, output_path: Path, clean: bool = False) -> None:
    """Per study (one panel): every grid point's median test loss with its SD over seeds, per model kind; the
    selected grid point is red. With `clean`, also without titles, labels and legend (<output_path>_clean)."""
    studies = list(dict.fromkeys(configs['study']))
    for annotated in ((True, False) if clean else (True,)):
        fig, axes = plt.subplots(1, len(studies), figsize=(1.9 * len(studies) + 0.6, 2.6), squeeze=False)
        for ax, study in zip(axes[0], studies):
            for position, kind in enumerate(['benchmark', 'gru']):
                points = configs[(configs['study'] == study) & (configs['kind'] == kind)]
                if points.empty:
                    continue
                chosen = best[(best['study'] == study) & (best['kind'] == kind)]['best_run'].iloc[0]
                # grid points spread over the kind's slot, in order of test loss; the selected one on top
                offsets = (pd.Series(range(len(points))) - (len(points) - 1) / 2) * 0.8 / max(len(points), 1)
                for offset, (_, point) in zip(offsets, points.iterrows()):
                    selected = point['best_run'] == chosen
                    ax.errorbar(position + offset, point['test_loss_median'], yerr=point['test_loss_std'],
                                fmt='o', markersize=5 if selected else 3.5, color=KIND_COLORS[kind], elinewidth=0.6,
                                markerfacecolor='#c44e52' if selected else 'white', zorder=3 if selected else 2)
            ax.set_xticks([0, 1], ['Benchmark', 'GRU'], fontsize=7.5)
            ax.set_xlim(-0.6, 1.6)
            ax.tick_params(axis='y', labelsize=7)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if annotated:
                ax.set_title(study, fontsize=8.5)
        if annotated:
            axes[0, 0].set_ylabel('test loss (median ± SD over seeds)', fontsize=8)
            fig.suptitle('Benchmark / GRU grid search: one point per grid point, red = selected', fontsize=9)
        fig.tight_layout()
        path = output_path if annotated else output_path.with_name(f'{output_path.name}_clean')
        for extension in ('pdf', 'png'):
            fig.savefig(f'{path}.{extension}', bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f'Saved {path}.pdf/.png')


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
    plot_grid_search(configs, best, out_dir / 'figure_benchmark_grid_search', clean=True)

    incomplete = configs[configs['n_seeds'] < configs['n_seeds'].max()]
    if len(incomplete):
        print(f"WARNING: {len(incomplete)} grid points have fewer than {configs['n_seeds'].max()} seeds")

    with pd.option_context('display.max_columns', None, 'display.width', 250):
        print(best[CONFIG_COLUMNS + ['test_loss_median', 'test_loss_std', 'n_seeds', 'best_run', 'median_run']])
    print(f"\nWritten to {out_dir}")


if __name__ == '__main__':
    main()
