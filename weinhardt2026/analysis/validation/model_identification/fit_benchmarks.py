"""Fit the benchmark to every simulated dataset with every seed, locally (SPICE: slurm_jobs/confusion_matrix.sh).

Runs fit_confusion_matrix.fit_run for STUDIES x GENERATORS x SEEDS with --fitter benchmark. kolff2025's
benchmark is fit in closed form, identical for every seed, so it is fit with the first seed only.
Existing fits (json) are skipped, so an interrupted run can simply be restarted; a failed fit is
reported at the end and does not stop the others.

Usage (from the repository root):
    python weinhardt2026/analysis/validation/model_identification/fit_benchmarks.py
    python weinhardt2026/analysis/validation/model_identification/fit_benchmarks.py --studies dezfouli2019 weber2024 --seeds 10 11
"""

import sys
import argparse
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from weinhardt2026.analysis.validation.model_identification.fit_confusion_matrix import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, fit_run, run_name
from weinhardt2026.analysis.validation.model_identification.generate_datasets import GENERATORS, STUDIES


CLOSED_FORM_STUDIES = ('kolff2025',)


def main() -> None:
    args = parse_args()
    failed = []
    for study in args.studies:
        seeds = args.seeds[:1] if study in CLOSED_FORM_STUDIES else args.seeds
        for generator in args.generators:
            for seed in seeds:
                try:
                    fit_run(study, generator, 'benchmark', seed, args.data_dir, args.out_dir)
                except Exception:
                    traceback.print_exc()
                    failed.append(run_name('benchmark', study, generator, seed))

    if failed:
        print(f'FAILED ({len(failed)}): {failed}', file=sys.stderr)
        sys.exit(1)
    print('All benchmark fits done.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fit the benchmark to every simulated confusion-matrix dataset with every seed.')
    parser.add_argument('--studies', nargs='+', default=list(STUDIES), choices=STUDIES)
    parser.add_argument('--generators', nargs='+', default=list(GENERATORS), choices=GENERATORS)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10, 20)))
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == '__main__':
    main()
