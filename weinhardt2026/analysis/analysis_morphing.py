"""
Model morphing: how the discovered equations change along a behavioral axis.

For any study whose SPICE model has participant embeddings (all but kolff2025's group model), from the best
final seed (utils/final_models.best_checkpoint):
  1. direction, per ensemble member (each has its own embedding space): linear regression of the participants'
     embeddings onto the metric (default: the participant's reward rate, utils/participant_metrics.reward_rate; or
     any column of the study's data averaged over the participant's trials)
  2. every participant's embedding is moved to N_STEPS positions along the member's direction (projection
     mean ± RANGE_SD SD), giving P x N_STEPS virtual participants that keep their real (training) data; all members
     in one estimator, the RNN frozen
  3. the SINDy equations of the virtual participants are refit with the Stage 2 of training
     (spice_training._run_sindy_training: sparsity discovery + coefficient estimation with multi-step shooting,
     which optimizes the equations for autonomous rollouts) and the module cleanup, with the hyperparameters of the
     final models (pruning threshold, ensemble ratio and alpha from the checkpoint name; run.py defaults otherwise)
  4. per step and member: mean coefficient over participants and inclusion probability (share of participants with
     the term); aggregated over members (mean ± SE)

Validation (discrete studies with rewards): the morphed SPICE-EQ (ensemble mean, as in generate_behavior) runs the
task (studies/<study>/simulation_<study>.py) for every virtual participant (the environment keeps the real
participant IDs, see VirtualParticipants); the mean reward per step shows whether behavior follows the axis.

Written to weinhardt2026/studies/<study>/results/morphing/ (<metric>: reward_rate or the column):
    morphing_<metric>.csv              per (module, term, step): coefficient (mean, SE), inclusion (mean, SE)
    morphing_<metric>.png/.pdf         per module: coefficient and inclusion curves of the terms included anywhere
    morphing_<metric>_validation.csv   per step: mean reward of the simulated virtual participants (mean, SE over runs)

Usage:
    python weinhardt2026/analysis/analysis_morphing.py dezfouli2019                   # metric: reward rate
    python weinhardt2026/analysis/analysis_morphing.py braun2018 --metric rt --participant_col subject   # another column
    python weinhardt2026/analysis/analysis_morphing.py dezfouli2019 --n_steps 3 --epochs 20             # quick test
    sbatch --export=ALL,STUDY=dezfouli2019 slurm_jobs/morphing.sh
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'weinhardt2026', 'figures'))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from spice import SpiceDataset, SpiceEstimator
from spice.resources.spice_training import _run_sindy_training, cross_entropy_loss

import figure_model_evaluation as evaluation
from weinhardt2026.analysis.analysis_coefficient_betas import get_criterion
from weinhardt2026.analysis.analysis_grid_search import parse_grid_tags
from weinhardt2026.utils.final_models import best_checkpoint
from weinhardt2026.utils.participant_metrics import reward_rate, study_data
from weinhardt2026.utils.study_models import choices_and_rewards, load_study_module

N_STEPS = 20
RANGE_SD = 1.0
N_VALIDATION_RUNS = 5
MIN_INCLUSION = 0.05   # terms plotted: included in at least this share of participants at some step

# Stage 2 settings of the final models: run.py defaults, overridden by the checkpoint name's grid tags (TAG_ARGUMENTS)
STAGE2_DEFAULTS = dict(epochs=1000, n_warmup_steps=100, sindy_alpha=5e-4, sindy_pruning_frequency=100,
                       sindy_ensemble_pruning=0.5, sindy_threshold_pruning=0.05, sindy_pruning_terms=None,
                       shooting_steps=100, sindy_ridge=True)
TAG_ARGUMENTS = {'th': 'sindy_threshold_pruning', 'al': 'sindy_alpha', 'er': 'sindy_ensemble_pruning'}


# ── Morphing ────────────────────────────────────────────────────────

def morphing_direction(estimator: SpiceEstimator, metric: np.ndarray, member: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit direction in the member's embedding space that best predicts the metric, and every participant's projection."""
    embedding = estimator.model.participant_embedding.weight.data[member].detach().cpu().numpy()   # (P, D)
    direction = LinearRegression().fit(embedding, metric).coef_
    direction = direction / np.linalg.norm(direction)
    return direction, embedding @ direction


def morphed_dataset(dataset: SpiceDataset, n_steps: int) -> SpiceDataset:
    """n_steps copies of every session, step-major (so per-session replays line up), participant p of copy s
    renamed to the virtual participant p * n_steps + s."""
    participant = dataset.xs[:, 0, 0, -1].long()
    xs_parts = []
    for step in range(n_steps):
        xs = dataset.xs.clone()
        xs[..., -1] = (participant * n_steps + step)[:, None, None].float().expand_as(xs[..., -1])
        xs_parts.append(xs)
    return SpiceDataset(torch.cat(xs_parts), dataset.ys.repeat(n_steps, 1, 1, 1),
                        n_reward_features=dataset.n_reward_features, continuous_action=dataset.continuous_action)


def morphed_estimator(estimator: SpiceEstimator, build, directions: np.ndarray, step_values: np.ndarray) -> SpiceEstimator:
    """All members with P x n_steps virtual participants: member e moves participant p's embedding to every step value
    of step_values[e] along directions[e]; the RNN, the participant tensors (indexed by the real participant) and the
    SINDy masks are copied."""
    model = estimator.model
    n_members, n_steps = step_values.shape
    n_participants = estimator.n_participants
    n_virtual = n_participants * n_steps
    real = torch.arange(n_virtual) // n_steps   # virtual participant v = p * n_steps + s
    new = build(n_participants=n_virtual, ensemble_size=n_members)
    new_model = new.model

    source, target = model.state_dict(), new_model.state_dict()
    state = {}
    for key, value in target.items():
        if key not in source:
            continue
        if source[key].shape == value.shape:
            state[key] = source[key].clone()
        elif source[key].dim() >= 2 and source[key].shape[1] == n_participants and value.shape[1] == n_virtual:
            state[key] = source[key][:, real.to(source[key].device)].clone()   # (E, P, ...) -> (E, P x n_steps, ...)
    new_model.load_state_dict(state, strict=False)

    embedding = model.participant_embedding.weight.data.detach().cpu()   # (E, P, D)
    moved = torch.empty(n_members, n_virtual, embedding.shape[-1])
    for member in range(n_members):
        direction = torch.tensor(directions[member], dtype=torch.float32)
        shift = torch.tensor(step_values[member], dtype=torch.float32).repeat(n_participants) - (embedding[member] @ direction)[real]
        moved[member] = embedding[member][real] + shift[:, None] * direction
    new_model.participant_embedding.weight.data = moved.to(new_model.participant_embedding.weight.device)

    for module in model.get_modules():
        device = new_model.sindy_coefficients[module].device
        new_model.sindy_coefficients_presence[module] = model.sindy_coefficients_presence[module][:, real].clone().to(device)
        new_model.sindy_coefficients_prior_mask[module] = model.sindy_coefficients_prior_mask[module][:, real].clone().to(device)
        new_model.sindy_pruning_patience_counters[module] = torch.zeros_like(new_model.sindy_coefficients[module].data)
    return new


def stage2_settings(checkpoint: str) -> dict:
    tags = parse_grid_tags(checkpoint)
    return STAGE2_DEFAULTS | {argument: tags[tag] for tag, argument in TAG_ARGUMENTS.items() if tag in tags}


def refit_sindy(estimator: SpiceEstimator, dataset: SpiceDataset, settings: dict, softmax_outputs: bool):
    """Stage 2 of training on the frozen RNN (every member sees the full data, as in training without bootstrap),
    then the module cleanup."""
    model = estimator.model
    n_members = model.ensemble_size
    xs = dataset.xs.unsqueeze(0).expand(n_members, -1, -1, -1, -1).contiguous()
    ys = dataset.ys.unsqueeze(0).expand(n_members, -1, -1, -1, -1).contiguous()
    _run_sindy_training(model=model, xs_train=xs, ys_train=ys, verbose=True, **settings)
    model.module_cleanup(xs=dataset.xs, ys=dataset.ys, softmax_outputs=softmax_outputs)


def step_curves(estimator: SpiceEstimator, n_participants: int, n_steps: int) -> dict:
    """Per module: (E, n_steps, n_terms) mean coefficient over participants and inclusion probability (experiment 0)."""
    coefficients = estimator.get_sindy_coefficients()
    curves = {}
    for module in estimator.get_modules():
        values = coefficients[module][:, :, 0]
        values = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
        presence = estimator.model.sindy_coefficients_presence[module][:, :, 0].float().cpu().numpy()
        shape = (values.shape[0], n_participants, n_steps, -1)
        curves[module] = (values.reshape(shape).mean(axis=1), presence.reshape(shape).mean(axis=1))
    return curves


def participant_metric(study: str, metric_col: str = None, participant_col: str = 'participant') -> np.ndarray:
    """Reward rate per participant, or the mean of a data column over the participant's trials (model order)."""
    if metric_col is None:
        return reward_rate(study)
    return get_criterion(study_data(study), metric_col, participant_col)[metric_col].to_numpy()


def run_morphing(study: str, metric: np.ndarray, metric_name: str, n_steps: int = N_STEPS,
                 epochs: int = None) -> tuple[pd.DataFrame, SpiceEstimator]:
    spice_module, dataset_train, _ = evaluation.load_study(study)
    build = lambda **overrides: evaluation.spice_estimator(spice_module, dataset_train, **overrides)
    checkpoint = best_checkpoint(study, 'spice')
    estimator = evaluation.load_spice(checkpoint, spice_module, dataset_train)
    if len(metric) != estimator.n_participants:
        raise ValueError(f'{study}: {len(metric)} participants in the data, {estimator.n_participants} in the model')

    n_members = estimator.model.ensemble_size
    directions, step_values = [], []
    for member in range(n_members):
        direction, projection = morphing_direction(estimator, metric, member)
        directions.append(direction)
        step_values.append(np.linspace(projection.mean() - RANGE_SD * projection.std(),
                                       projection.mean() + RANGE_SD * projection.std(), n_steps))
        r2 = LinearRegression().fit(projection[:, None], metric).score(projection[:, None], metric)
        print(f'{study}: member {member}, R² of the direction with {metric_name}: {r2:.3f}', flush=True)

    morphed = morphed_estimator(estimator, build, np.array(directions), np.array(step_values))
    settings = stage2_settings(checkpoint) | ({'epochs': epochs, 'n_warmup_steps': min(100, epochs // 10)} if epochs else {})
    print(f'{study}: Stage 2 refit of {morphed.n_participants} virtual participants, {settings}', flush=True)
    refit_sindy(morphed, morphed_dataset(dataset_train, n_steps), settings,
                softmax_outputs=getattr(spice_module, 'LOSS_FN', cross_entropy_loss) is cross_entropy_loss)

    rows = []
    terms = estimator.get_candidate_terms()
    for module, (coefficient, inclusion) in step_curves(morphed, estimator.n_participants, n_steps).items():
        for member in range(n_members):
            for step in range(n_steps):
                rows += [dict(member=member, module=module, term=term, step=step, coefficient=coefficient[member, step, index],
                              inclusion=inclusion[member, step, index]) for index, term in enumerate(terms[module])]
    table = pd.DataFrame(rows).groupby(['module', 'term', 'step'], sort=False).agg(
        coefficient=('coefficient', 'mean'), coefficient_se=('coefficient', 'sem'),
        inclusion=('inclusion', 'mean'), inclusion_se=('inclusion', 'sem')).reset_index()
    return table, morphed


class VirtualParticipants:
    """The morphed estimator called on sessions of the real participants: the task environment keeps the real
    participant IDs (it looks up per-participant data), the model sees participant p of morphing step k as the
    virtual participant p * n_steps + k."""

    def __init__(self, estimator: SpiceEstimator, steps: torch.Tensor, n_steps: int):
        self.estimator, self.steps, self.n_steps = estimator, steps, n_steps   # steps: (n_sessions,)

    @property
    def device(self):
        return self.estimator.device

    def eval(self, use_sindy: bool = None) -> 'VirtualParticipants':
        if use_sindy is not None:
            self.estimator.eval(use_sindy=use_sindy)
        return self

    def __call__(self, xs: torch.Tensor, state=None):
        xs = xs.clone()
        xs[..., -1] = xs[..., -1] * self.n_steps + self.steps.to(xs.device).view(-1, *([1] * (xs.dim() - 2)))
        return self.estimator(xs, state)


def validate(study: str, estimator: SpiceEstimator, n_steps: int) -> pd.DataFrame:
    """Mean reward per morphing step of the morphed SPICE-EQ simulated in the task (all blocks, N_VALIDATION_RUNS runs)."""
    simulation = load_study_module(study)
    real = simulation.load_dataset()
    dataset = SpiceDataset(real.xs.repeat(n_steps, 1, 1, 1), real.ys.repeat(n_steps, 1, 1, 1),
                           n_reward_features=real.n_reward_features)   # step-major copies, real participant IDs
    steps = torch.arange(n_steps).repeat_interleave(real.xs.shape[0])
    estimator.eval(use_sindy=True)
    model = VirtualParticipants(estimator, steps, n_steps)
    runs = []
    for run in range(N_VALIDATION_RUNS):
        torch.manual_seed(run)
        _, rewards = choices_and_rewards(simulation.generate(model, dataset))
        session_reward = np.nanmean(rewards, axis=1)
        runs.append([np.nanmean(session_reward[steps.numpy() == step]) for step in range(n_steps)])
    runs = np.array(runs)
    return pd.DataFrame(dict(step=range(n_steps), reward=runs.mean(axis=0), reward_se=runs.std(axis=0) / np.sqrt(len(runs))))


# ── Plot ────────────────────────────────────────────────────────────

def plot_morphing(table: pd.DataFrame, study: str, metric_col: str, output_path: str):
    included = table.groupby(['module', 'term'], sort=False)['inclusion'].max()
    modules = list(dict.fromkeys(table['module']))
    fig, axes = plt.subplots(2, len(modules), figsize=(3.6 * len(modules), 6), squeeze=False, sharex=True)
    for col, module in enumerate(modules):
        terms = [term for (m, term), value in included.items() if m == module and value >= MIN_INCLUSION]
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(terms), 1)))
        for term, color in zip(terms, colors):
            rows = table[(table['module'] == module) & (table['term'] == term)]
            for row, (value, se) in enumerate((('coefficient', 'coefficient_se'), ('inclusion', 'inclusion_se'))):
                ax = axes[row, col]
                ax.plot(rows['step'], rows[value], color=color, linewidth=1.2, label=term if row == 0 else None)
                ax.fill_between(rows['step'], rows[value] - rows[se].fillna(0), rows[value] + rows[se].fillna(0),
                                color=color, alpha=0.2, linewidth=0)
        axes[0, col].set_title(module, fontsize=9, fontweight='bold')
        axes[0, col].axhline(0, color='grey', linewidth=0.5)
        axes[0, col].legend(fontsize=6, frameon=False)
        axes[1, col].set_ylim(-0.02, 1.02)
        axes[1, col].set_xlabel(f'morphing step (low → high {metric_col})', fontsize=8)
        for ax in axes[:, col]:
            ax.tick_params(labelsize=7)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel('coefficient (mean over participants)', fontsize=8)
    axes[1, 0].set_ylabel('inclusion probability', fontsize=8)
    fig.suptitle(f'{study}: equations along {metric_col} (mean ± SE over ensemble members)', fontsize=10)
    fig.tight_layout()
    for extension in ('pdf', 'png'):
        fig.savefig(f'{output_path}.{extension}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {output_path}.pdf/.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('study')
    parser.add_argument('--metric', default=None, help='data column, averaged per participant (default: reward rate)')
    parser.add_argument('--participant_col', default='participant', help='participant column of the data (with --metric)')
    parser.add_argument('--n_steps', type=int, default=N_STEPS, help='positions along the morphing axis')
    parser.add_argument('--epochs', type=int, default=None, help='Stage 2 epochs per phase (default: as in training, 1000)')
    args = parser.parse_args()
    metric_name = args.metric or 'reward_rate'

    output_dir = evaluation._study_path(args.study, 'results', 'morphing')
    os.makedirs(output_dir, exist_ok=True)
    metric = participant_metric(args.study, args.metric, args.participant_col)
    table, morphed = run_morphing(args.study, metric, metric_name, n_steps=args.n_steps, epochs=args.epochs)
    table.to_csv(os.path.join(output_dir, f'morphing_{metric_name}.csv'), index=False)
    plot_morphing(table, args.study, metric_name, os.path.join(output_dir, f'morphing_{metric_name}'))

    if morphed.model.n_reward_features > 0 and not evaluation.load_study(args.study)[1].continuous_action:
        validation = validate(args.study, morphed, args.n_steps)
        validation.to_csv(os.path.join(output_dir, f'morphing_{metric_name}_validation.csv'), index=False)
        print(validation.round(4).to_string(index=False))
