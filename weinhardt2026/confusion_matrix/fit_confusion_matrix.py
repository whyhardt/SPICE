"""Fit one model (SPICE or the benchmark) with one seed to one simulated dataset of generate_datasets.py.

The simulated CSVs are in the study's format, so every fit runs exactly as on the real data, with the
configuration stored for the dataset in <data_dir>/generators.csv (split, final hyperparameters):
    --fitter spice      weinhardt2026/run.py, as slurm_jobs/grid_search.sh calls it
    --fitter benchmark  benchmark_grid_search.fit_one on the study's setup loaded from the simulated CSV:
                        random start around the default parameters (N(0, init_std)), lr scheduler
                        stepping on the test loss; kolff2025: the closed-form logistic regression on the
                        training fold (identical for every seed, so one seed suffices)

SPICE is fit on the cluster (slurm_jobs/confusion_matrix.sh); the benchmarks are fast enough to fit locally:

    for study in dezfouli2019 ganesh2024a eckstein2026 bustamante2023 braun2018 bruckner2025 weber2024; do
      for generator in spice benchmark; do for seed in $(seq 10 19); do
        python weinhardt2026/confusion_matrix/fit_confusion_matrix.py --study $study --generator $generator --fitter benchmark --seed $seed
      done; done
    done
    for generator in spice benchmark; do
      python weinhardt2026/confusion_matrix/fit_confusion_matrix.py --study kolff2025 --generator $generator --fitter benchmark --seed 10
    done

Saved to <out_dir>/<study>/<fitter>_<study>_sim<generator>_rs<seed>.{pkl,json} (default out_dir:
weinhardt2026/confusion_matrix/runs). The json is written once the fit is complete; fits with an
existing json are skipped, so a timed-out array task can simply be resubmitted.

Usage (from the repository root):
    python weinhardt2026/confusion_matrix/fit_confusion_matrix.py --study dezfouli2019 \
        --generator benchmark --fitter spice --seed 10
"""

import sys
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from weinhardt2026.confusion_matrix.generate_datasets import DEFAULT_OUT_DIR as DEFAULT_DATA_DIR, GENERATORS, STUDIES


DEFAULT_OUT_DIR = 'weinhardt2026/confusion_matrix/runs'
FITTERS = ('spice', 'benchmark')
SPICE_ARGUMENTS = ('pruning_threshold', 'sindy_weight', 'sindy_alpha', 'pruning_test', 'gate_penalty', 'feature_penalty')


def run_name(fitter: str, study: str, generator: str, seed: int) -> str:
    return f'{fitter}_{study}_sim{generator}_rs{seed}'


def load_generator_row(data_dir: str, study: str, generator: str) -> pd.Series:
    generators = pd.read_csv(Path(data_dir) / 'generators.csv')
    row = generators[(generators['study'] == study) & (generators['generator'] == generator)]
    if len(row) != 1:
        raise ValueError(f'{study}: no dataset simulated by {generator} in {data_dir}/generators.csv')
    return row.iloc[0]


def fit_spice(config: pd.Series, model: Path, seed: int) -> dict:
    """run.py with the study module, data kwargs, split and final SPICE hyperparameters."""
    command = [
        sys.executable, str(ROOT / 'weinhardt2026' / 'run.py'),
        '--module', config['module'],
        '--data', str(ROOT / config['data']),
        '--model', str(model),
        '--data_kwargs', config['data_kwargs'],
        '--test_blocks', str(config['test_blocks']),
        *[item for name in SPICE_ARGUMENTS for item in (f'--{name}', repr(float(config[f'spice_{name}'])))],
        *str(config['spice_extra_arguments'] if pd.notna(config['spice_extra_arguments']) else '').split(),
        '--seed', str(seed),
    ]
    print(' '.join(command), flush=True)
    start = time.time()
    subprocess.run(command, check=True, cwd=ROOT)
    return dict(duration_sec=time.time() - start)


def fit_benchmark(config: pd.Series, model: Path, seed: int) -> dict:
    """benchmark_grid_search.fit_one on the study's setup, loaded from the simulated CSV (kolff2025: closed form)."""
    from weinhardt2026.benchmark_grid_search.benchmark_grid_search import STUDIES as SETUPS, fit_one

    if config['study'] == 'kolff2025':
        from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import LogisticRegressionModel, get_dataset
        start = time.time()
        fitted = LogisticRegressionModel().fit(get_dataset(path_data=str(ROOT / config['data']), test_fold=0)[0])
        torch.save(fitted.state_dict(), model)
        return dict(n_parameters=fitted.count_parameters(), duration_sec=time.time() - start)

    setup = SETUPS[config['study']](path_data=str(ROOT / config['data']))
    args = argparse.Namespace(
        kind='benchmark', lr=float(config['benchmark_lr']), scheduler=int(config['benchmark_scheduler']),
        epochs=int(config['benchmark_epochs']), init_std=float(config['benchmark_init_std']), hidden_size=None, dropout=None,
    )
    fitted, metrics = fit_one(setup, args, seed)
    torch.save(fitted.state_dict(), model)
    return metrics


def main() -> None:
    args = parse_args()
    config = load_generator_row(args.data_dir, args.study, args.generator)

    out_dir = Path(args.out_dir) / args.study
    out_dir.mkdir(parents=True, exist_ok=True)
    name = run_name(args.fitter, args.study, args.generator, args.seed)
    if (out_dir / f'{name}.json').exists():
        print(f'[{args.study}] {name} exists, nothing to do')
        return

    print(f'[{args.study}] fitting {name}')
    fit = fit_spice if args.fitter == 'spice' else fit_benchmark
    metrics = fit(config, out_dir / f'{name}.pkl', args.seed)

    hyperparameters = {key: value for key, value in config.items() if key.startswith(f'{args.fitter}_') and pd.notna(value)}
    result = dict(
        study=args.study, generator=args.generator, fitter=args.fitter, seed=args.seed, data=config['data'],
        generator_checkpoint=config['checkpoint'], test_blocks=config['test_blocks'], **hyperparameters,
    )
    (out_dir / f'{name}.json').write_text(json.dumps(result | metrics, indent=2, default=float))
    print(f'[{args.study}] {name} done ({metrics["duration_sec"]:.0f}s)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fit one model with one seed to one simulated confusion-matrix dataset.')
    parser.add_argument('--study', required=True, choices=STUDIES)
    parser.add_argument('--generator', required=True, choices=GENERATORS, help='model the dataset was simulated from')
    parser.add_argument('--fitter', required=True, choices=FITTERS, help='model fit to the dataset')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == '__main__':
    main()
