"""Generate paper figures for a given study.

Usage:
    python -m weinhardt2026.figures.run --study dezfouli2019
    python -m weinhardt2026.figures.run --study dezfouli2019 --figures 2 4
    python -m weinhardt2026.figures.run --study dezfouli2019 --figures 3
    python -m weinhardt2026.figures.run --list
"""

import argparse
import os

import torch

from spice import SpiceEstimator, csv_to_dataset
from spice.precoded import workingmemory
from weinhardt2026.studies.eckstein2026 import spice_eckstein2026
from weinhardt2026.figures.figure2 import plot_figure2
from weinhardt2026.figures.figure3 import plot_figure3
from weinhardt2026.figures.figure4 import plot_figure4
from weinhardt2026.figures.figure5 import plot_figure5


# ── Study registry ────────────────────────────────────────────────────
# Each study defines paths and config needed for figure generation.
# Add new studies by adding an entry here.

STUDY_REGISTRY = {
    'dezfouli2019': {
        'spice_class': workingmemory.SpiceModel,
        'spice_config': workingmemory.CONFIG,
        'n_actions': 2,
        'polynomial_degree': 2,
        'data': 'weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv',
        'params': 'weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019.pkl',
        'results_dir': 'weinhardt2026/studies/dezfouli2019/results',
        'figures_dir': 'weinhardt2026/studies/dezfouli2019/figures',
        'metrics_csv': 'behavioral_metrics_real.csv',
        'beta_csv': 'discrete_odds_ratio_results.csv',
        'n_clusters': 3,
        'n_trials_show': 100,
        'stability_pattern': 'weinhardt2026/studies/dezfouli2019/params_array/spice_dezfouli2019_stability_[0-9].pkl',
        'hpscan_csv': 'weinhardt2026/studies/dezfouli2019/results/hpscan_results.csv',
        'model_kwargs': {'reward_binary': True},
    },
    'eckstein2026': {
        'spice_class': spice_eckstein2026.SpiceModel,
        'spice_config': spice_eckstein2026.CONFIG,
        'n_actions': 4,
        'polynomial_degree': 2,
        'data': 'weinhardt2026/studies/eckstein2026/data/eckstein2026.csv',
        'params': 'weinhardt2026/studies/eckstein2026/params/spice_eckstein2026.pkl',
        'results_dir': 'weinhardt2026/studies/eckstein2026/results',
        'figures_dir': 'weinhardt2026/studies/eckstein2026/figures',
        'metrics_csv': 'behavioral_metrics_real.csv',
        'beta_csv': 'continuous_effect_results_all.csv',
        'n_clusters': 3,
        'n_trials_show': 100,
        # 'stability_pattern': 'weinhardt2026/studies/eckstein2026/params_array/spice_dezfouli2019_stability_[0-9].pkl',
        # 'hpscan_csv': 'weinhardt2026/studies/dezfouli2019/results/hpscan_results.csv',
        # 'model_kwargs': {'reward_binary': True},
    },
}


def _load_estimator(study_cfg):
    """Load a fitted SpiceEstimator from a study config."""
    ckpt = torch.load(study_cfg['params'], map_location='cpu')
    first_mod = next(iter(study_cfg['spice_config'].library_setup))
    ensemble_size = ckpt['model'][f'sindy_coefficients.{first_mod}'].shape[0]
    n_participants = ckpt['model'][f'sindy_coefficients.{first_mod}'].shape[1]
    del ckpt

    estimator = SpiceEstimator(
        spice_class=study_cfg['spice_class'],
        spice_config=study_cfg['spice_config'],
        n_actions=study_cfg['n_actions'],
        n_participants=n_participants,
        sindy_library_polynomial_degree=study_cfg['polynomial_degree'],
        ensemble_size=ensemble_size,
        use_sindy=True,
    )
    estimator.load_spice(study_cfg['params'])
    return estimator


def generate_figure2(study_cfg, estimator):
    """Figure 2: Equation showcase for structurally distinctive participants."""
    figures_dir = study_cfg['figures_dir']

    dataset = csv_to_dataset(file=study_cfg['data'])
    dataset.normalize_rewards()

    plot_figure2(
        estimator=estimator,
        dataset=dataset,
        participant_ids=None,  # auto-select via structural distance
        session_idx=None,
        n_trials_show=study_cfg.get('n_trials_show'),
        output_dir=os.path.join(figures_dir, 'figure2'),
    )


def generate_figure3(study_cfg):
    """Figure 3: Performance + Structural variance (cross-study)."""
    plot_figure3(
        output_dir=os.path.join(study_cfg['figures_dir'], 'figure3'),
    )


def generate_figure4(study_cfg, estimator):
    """Figure 4: Individual differences."""
    results_dir = study_cfg['results_dir']
    figures_dir = study_cfg['figures_dir']

    plot_figure4(
        estimator=estimator,
        path_behavioral_metrics=os.path.join(results_dir, study_cfg['metrics_csv']),
        results_csv_path=os.path.join(results_dir, study_cfg['beta_csv']),
        n_clusters=study_cfg['n_clusters'],
        output_dir=os.path.join(figures_dir, 'figure4'),
        path_raw_data=study_cfg.get('data'),
    )


def generate_figure5(study_cfg):
    """Figure 5: Across-run stability analysis."""
    from glob import glob
    pattern = study_cfg.get('stability_pattern')
    if not pattern:
        print("  No stability_pattern configured for this study, skipping.")
        return
    pkl_paths = sorted(glob(pattern))
    if not pkl_paths:
        print(f"  No stability runs found matching {pattern}")
        return

    plot_figure5(
        stability_pkl_paths=pkl_paths,
        spice_class=study_cfg['spice_class'],
        spice_config=study_cfg['spice_config'],
        n_actions=study_cfg['n_actions'],
        output_dir=os.path.join(study_cfg['figures_dir'], 'figure5'),
        polynomial_degree=study_cfg.get('polynomial_degree', 2),
        model_kwargs=study_cfg.get('model_kwargs'),
        hpscan_csv=study_cfg.get('hpscan_csv'),
    )


FIGURE_GENERATORS = {
    '2': generate_figure2,
    '3': generate_figure3,
    '4': generate_figure4,
    '5': generate_figure5,
}


def main():
    parser = argparse.ArgumentParser(
        description='Generate paper figures for a study.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--study', type=str, help='Study name (e.g. dezfouli2019)')
    parser.add_argument('--figures', nargs='+', default=['2', '3', '4', '5'],
                        help='Which figures to generate (default: all). E.g. --figures 2 4')
    parser.add_argument('--params', type=str, default=None,
                        help='Override params .pkl path (default: use study registry)')
    parser.add_argument('--list', action='store_true', help='List available studies')
    args = parser.parse_args()

    if args.list:
        print("Available studies:")
        for name in STUDY_REGISTRY:
            print(f"  {name}")
        return

    if not args.study:
        parser.error("--study is required (use --list to see available studies)")

    if args.study not in STUDY_REGISTRY:
        parser.error(f"Unknown study '{args.study}'. Available: {', '.join(STUDY_REGISTRY)}")

    study_cfg = dict(STUDY_REGISTRY[args.study])  # copy so we can override
    if args.params is not None:
        study_cfg['params'] = args.params
    print(f"Generating figures for: {args.study}")
    print(f"Params: {study_cfg['params']}")
    print(f"Figures: {', '.join(args.figures)}")
    print(f"Output: {study_cfg['figures_dir']}/\n")

    # Only load the estimator if we need it (figures 2 and 4)
    estimator = None
    needs_estimator = any(f in ('2', '4') for f in args.figures)
    no_estimator_figures = {'3', '5'}
    if needs_estimator:
        estimator = _load_estimator(study_cfg)

    for fig_id in args.figures:
        if fig_id not in FIGURE_GENERATORS:
            print(f"Warning: unknown figure '{fig_id}', skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Figure {fig_id}")
        print(f"{'='*60}")

        gen_fn = FIGURE_GENERATORS[fig_id]
        if fig_id in no_estimator_figures:
            gen_fn(study_cfg)
        else:
            gen_fn(study_cfg, estimator)

    print(f"\nDone. All figures saved to {study_cfg['figures_dir']}/")


if __name__ == '__main__':
    main()
