"""Simulated datasets for the model confusion matrix (SPICE vs. benchmark).

Per study and generator (SPICE, benchmark), one dataset is simulated from the generator's median seed:
all final seeds (weinhardt2026/utils/final_models.py) are ranked by their training log-likelihood per
trial and the (lower) median one is taken, as run_benchmark_grid_search.py picks its median run.
For SPICE, the seeds are scored by their best ensemble member (highest training likelihood), and the
dataset is simulated from that member's SPICE-EQ.

The simulation replays the real experiment (all blocks and participants) with the study's generator
(studies/<study>/simulation_<study>.py: load_dataset, generate). Continuous studies (bruckner2025,
weber2024) simulate from the Gaussian likelihood their models are scored with: the response is the
model's prediction plus N(0, sigma^2), sigma the generator's MLE residual s.d. on the training data
per (participant, experiment) group, as in the AIC/BIC of analysis/evaluation/run_model_evaluation.py.

Every dataset is a copy of the study's CSV with the simulated columns overwritten in raw units
(SIMULATED_COLUMNS), so it is fit exactly like the real data (run.py with the study's data_kwargs and
hooks, the benchmark's own get_dataset). Simulated cells without a simulated value (e.g. the outcome of
a session's last trial, which never enters the dataset) are NaN; columns the fit does not read keep the
real values. Each CSV is checked by loading it through that pipeline and comparing with the simulation.

Written to <out_dir>/ (default weinhardt2026/analysis/validation/model_identification/data/):
    <study>_sim<generator>.csv   the simulated dataset
    generators.csv               one row per (study, generator): the generator (checkpoint, member,
                                 training likelihood, response noise) and the configuration every model
                                 is refit with (split, final hyperparameters); read by fit_confusion_matrix.py
    selection.csv                training likelihood of every seed of both generators

kolff2025's benchmark is a logistic regression fit in closed form: there is one fit (on the training fold)
instead of seeds, and it is the generator.

Usage (from the repository root):
    python weinhardt2026/analysis/validation/model_identification/generate_datasets.py --studies dezfouli2019 weber2024
"""

import sys
import argparse
import importlib
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'weinhardt2026'))  # study modules import 'studies.<study>...' (as in run.py)

import numpy as np
import pandas as pd
import torch

from spice import SpiceDataset, csv_to_dataset


STUDIES = ('dezfouli2019', 'ganesh2024a', 'eckstein2026', 'bustamante2023', 'braun2018', 'bruckner2025', 'weber2024', 'kolff2025')
GENERATORS = ('spice', 'benchmark')
DEFAULT_OUT_DIR = 'weinhardt2026/analysis/validation/model_identification/data'
BENCHMARK_RUNS_DIR = ROOT / 'weinhardt2026' / 'analysis' / 'grid_search' / 'runs'
ROW = '_row'   # raw row index, carried through csv_to_dataset as an extra additional input
CLOSED_FORM = 'closed-form fit'   # checkpoint of kolff2025's benchmark

# run.py arguments slurm_jobs/grid_search.sh adds per study, besides the grid
SPICE_EXTRA_ARGUMENTS = {'kolff2025': '--polynomial_degree 1 --n_participants 1'}

SPICE_TAGS = re.compile(r'_th(?P<pruning_threshold>[^_]+)_sw(?P<sindy_weight>[^_]+)_al(?P<sindy_alpha>[^_]+)'
                        r'_er(?P<pruning_test>[^_]+)_gp(?P<gate_penalty>[^_]+)_fp(?P<feature_penalty>[^_]+)_rs\d+\.pkl$')
BENCHMARK_TAGS = re.compile(r'_lr(?P<lr>[^_]+)_sc(?P<scheduler>[01])_rs\d+\.pkl$')


def study_csv(study: str) -> Path:
    return ROOT / 'weinhardt2026' / 'studies' / study / 'data' / f'{study}.csv'


def dataset_path(out_dir: str, study: str, generator: str) -> Path:
    return Path(out_dir) / f'{study}_sim{generator}.csv'


def study_data(study: str) -> tuple[dict, list[int]]:
    """csv_to_dataset kwargs and test blocks of the study, as slurm_jobs/grid_search.sh passes them to run.py."""
    if study == 'kolff2025':
        from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import DATA_KWARGS, get_dataset
        return dict(DATA_KWARGS), list(get_dataset(test_fold=0)[2]['test_blocks'])
    from weinhardt2026.utils.studies import STUDIES as REGISTRY
    return dict(REGISTRY[study]['data_kwargs']), list(REGISTRY[study]['test_blocks'])


def load_csv(study: str, path: Path) -> SpiceDataset:
    """A CSV in the study's format, loaded as run.py loads it (study hooks, data kwargs, reward normalization)."""
    spice_module = importlib.import_module(f'studies.{study}.spice_{study}')
    data = pd.read_csv(path)
    if hasattr(spice_module, 'prepare_dataframe'):
        data = spice_module.prepare_dataframe(data)
    dataset = csv_to_dataset(file=data, **study_data(study)[0])
    if getattr(spice_module, 'NORMALIZE_REWARDS', True):
        dataset.normalize_rewards()
    if hasattr(spice_module, 'prepare_dataset'):
        dataset = spice_module.prepare_dataset(dataset)
    return dataset


def kolff2025_benchmark():
    """kolff2025's benchmark fit on the training fold (as studies/kolff2025/simulation_kolff2025.load_models), and that fold."""
    from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import LogisticRegressionModel, evaluation_dataset, get_dataset
    dataset_train = get_dataset(test_fold=0)[0]
    return LogisticRegressionModel().fit(dataset_train), evaluation_dataset(dataset_train)


def upsert(path: Path, table: pd.DataFrame, study: str) -> None:
    """Replace the study's rows of a table shared across studies."""
    if path.exists():
        existing = pd.read_csv(path)
        table = pd.concat([existing[existing['study'] != study], table], ignore_index=True)
    table.to_csv(path, index=False)


# -------------------------------------------------------------------------------------------
# GENERATOR SELECTION: THE MEDIAN SEED BY TRAINING LIKELIHOOD
# -------------------------------------------------------------------------------------------

def score_seeds(study: str, generator: str) -> pd.DataFrame:
    """Training log-likelihood per trial of every final seed (SPICE: of its best member's SPICE-EQ)."""
    from weinhardt2026.utils.study_models import member_train_loglik, model_train_loglik
    from weinhardt2026.utils.final_models import model_setup, checkpoint_seed, final_checkpoints

    if study == 'kolff2025' and generator == 'benchmark':
        model, dataset_train = kolff2025_benchmark()
        return pd.DataFrame([dict(study=study, generator=generator, checkpoint=CLOSED_FORM, seed=None, member=None,
                                  loglik_train=model_train_loglik(study, model, dataset_train))])

    load, dataset_train = model_setup(study, generator)
    rows = []
    for path in final_checkpoints(study, generator):
        model = load(path)
        if generator == 'spice':
            loglik = member_train_loglik(study, model, dataset_train)
            member, score = int(np.argmax(loglik)), float(loglik.max())
        else:
            member, score = None, model_train_loglik(study, model, dataset_train)
        rows.append(dict(study=study, generator=generator, checkpoint=os.path.relpath(path, ROOT),
                         seed=checkpoint_seed(path), member=member, loglik_train=score))
        del model
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def median_seed(scores: pd.DataFrame) -> pd.Series:
    """Seed at the (lower) median training loss, i.e. ranked by descending likelihood."""
    ordered = scores.sort_values('loglik_train', ascending=False).reset_index(drop=True)
    return ordered.iloc[(len(ordered) - 1) // 2]


def load_generator(study: str, generator: str, selected: pd.Series):
    """The selected seed as the generative analysis simulates it (SPICE: its member's SPICE-EQ)."""
    from weinhardt2026.utils.study_models import EnsembleMember
    from weinhardt2026.utils.final_models import model_setup

    if selected['checkpoint'] == CLOSED_FORM:
        return kolff2025_benchmark()[0].eval()
    load, _ = model_setup(study, generator)
    model = load(str(ROOT / selected['checkpoint']))
    if generator == 'spice':
        return EnsembleMember(model, int(selected['member'])).eval(use_sindy=True)
    return model.eval()


# -------------------------------------------------------------------------------------------
# RESPONSE NOISE (CONTINUOUS STUDIES): MLE RESIDUAL S.D. PER (PARTICIPANT, EXPERIMENT)
# -------------------------------------------------------------------------------------------

@torch.no_grad()
def predict(model, dataset: SpiceDataset) -> torch.Tensor:
    """Outputs (B, T, W, A) on CPU of a benchmark or an EnsembleMember."""
    from weinhardt2026.utils.study_models import EnsembleMember
    from weinhardt2026.utils.studies import predict as predict_registered
    if not isinstance(model, EnsembleMember):
        return predict_registered(model, dataset)
    model.estimator.model.init_state(batch_size=dataset.xs.shape[0])
    outputs, _ = model(dataset.xs.to(model.device))
    return outputs[0].detach().cpu()


def response_noise_std(study: str, generator: str, model, dataset: SpiceDataset) -> torch.Tensor:
    """(n_sessions,) noise s.d. of every session of `dataset`: sqrt(SSE / (n * dim)) of the generator on the
    training data of the session's (participant, experiment) group."""
    from weinhardt2026.utils.studies import continuous_session_sse
    from weinhardt2026.utils.final_models import model_setup

    _, dataset_train = model_setup(study, generator)
    sse, n_trials, dim = continuous_session_sse(predict(model, dataset_train), dataset_train, study)
    groups = pd.DataFrame({
        'participant': dataset_train.xs[:, 0, 0, -1].long().numpy(),
        'experiment': dataset_train.xs[:, 0, 0, -2].long().numpy(),
        'sse': sse.numpy(), 'n': n_trials.numpy(),
    }).groupby(['participant', 'experiment'])[['sse', 'n']].sum()
    std = np.sqrt(groups['sse'] / (groups['n'] * dim))

    keys = zip(dataset.xs[:, 0, 0, -1].long().tolist(), dataset.xs[:, 0, 0, -2].long().tolist())
    return torch.tensor([std[key] for key in keys], dtype=torch.float32)


# -------------------------------------------------------------------------------------------
# SIMULATED DATASET -> CSV IN THE STUDY'S RAW FORMAT
# -------------------------------------------------------------------------------------------

def trial_rows(study: str, df: pd.DataFrame) -> np.ndarray:
    """(n_sessions, n_positions) raw row index of every trial position of every session, -1 where none.

    Sessions and positions are those of the study's SpiceDataset: positions 0 .. n-2 are its trial slots,
    position n-1 is the last trial (only its choice enters the dataset, as the target of slot n-2).
    """
    spice_module = importlib.import_module(f'studies.{study}.spice_{study}')
    kwargs = study_data(study)[0]

    prepared = df.copy()
    if hasattr(spice_module, 'prepare_dataframe'):
        prepared = spice_module.prepare_dataframe(prepared)
    prepared[ROW] = np.arange(len(prepared), dtype=float)
    kwargs['additional_inputs'] = list(kwargs.get('additional_inputs') or []) + [ROW]
    if kwargs.get('timeshift_additional_inputs') is not None:
        kwargs['timeshift_additional_inputs'] = list(kwargs['timeshift_additional_inputs']) + [0]
    dataset = csv_to_dataset(file=prepared, **kwargs)
    rows_slots = dataset.xs[:, :, 0, -6].numpy()

    # session columns and their metadata column in xs (csv_to_dataset: ids by first appearance, raw block)
    choice = kwargs.get('df_choice', 'choice')
    choice = choice if isinstance(choice, str) else choice[0]
    session_columns = {kwargs.get('df_participant_id', 'participant'): -1, kwargs.get('df_experiment_id', 'experiment'): -2,
                       kwargs.get('df_block', 'block'): -3}
    session_columns = {column: index for column, index in session_columns.items() if column in prepared.columns}
    keys = list(session_columns)
    valid = prepared[prepared[choice].notna()]

    # the last trial of a session: the session's next row that csv_to_dataset keeps (valid choice)
    next_row = valid.groupby(keys, sort=False)[ROW].shift(-1)
    next_row.index = valid[ROW].astype(int)

    # the only trial of a single-trial session (no slot): found by the session's metadata
    session_ids = pd.DataFrame({
        column: valid[column] if index == -3 else valid[column].map({value: i for i, value in enumerate(prepared[column].unique())})
        for column, index in session_columns.items()
    }).astype(float)
    first_row = dict(zip(map(tuple, session_ids.to_numpy()), valid[ROW].astype(int)))   # last occurrence; single-row sessions only
    metadata = dataset.xs[:, 0, 0, list(session_columns.values())].numpy()

    n_sessions, n_slots = rows_slots.shape
    rows = np.full((n_sessions, n_slots + 1), -1, dtype=int)
    for session in range(n_sessions):
        slots = rows_slots[session][~np.isnan(rows_slots[session])].astype(int)
        rows[session, :len(slots)] = slots
        if len(slots):
            rows[session, len(slots)] = int(next_row[slots[-1]])
        else:
            rows[session, 0] = first_row[tuple(metadata[session].astype(float))]
    return rows


def n_actions_of(study: str, dataset: SpiceDataset) -> int:
    """Action columns of xs; studies whose ys carry loss metadata declare them (ESTIMATOR_KWARGS, as in run.py)."""
    spice_module = importlib.import_module(f'studies.{study}.spice_{study}')
    return getattr(spice_module, 'ESTIMATOR_KWARGS', {}).get('n_actions', dataset.ys.shape[-1])


def simulated_positions(dataset_sim: SpiceDataset, n_actions: int, n_positions: int) -> tuple[np.ndarray, np.ndarray]:
    """Per session and trial position: the simulated action (n_sessions, n_positions, n_actions) and the
    slot features (n_sessions, n_positions, n_features); NaN where the simulation sets none.

    Position k < m (m = the session's simulated slots) is slot k; the action of position m is the target
    of slot m - 1.
    """
    xs, ys = dataset_sim.xs[:, :, 0].numpy(), dataset_sim.ys[:, :, 0].numpy()
    n_sessions, _, n_features = xs.shape
    actions = np.full((n_sessions, n_positions, n_actions), np.nan)
    features = np.full((n_sessions, n_positions, n_features), np.nan)
    for session in range(n_sessions):
        m = int((~np.isnan(xs[session, :, 0])).sum())
        if m == 0:
            continue
        actions[session, :m] = xs[session, :m, :n_actions]
        actions[session, m] = ys[session, m - 1, :n_actions]
        features[session, :m] = xs[session, :m]
    return actions, features


def shifted(values: np.ndarray, shift: int) -> np.ndarray:
    """Slot values -> raw position values for an input loaded with csv_to_dataset's timeshift (-1: slot k = row k+1)."""
    if shift == -1:
        out = np.full_like(values, np.nan)
        out[:, 1:] = values[:, :-1]
        return out
    if shift == 0:
        return values
    raise ValueError(f'timeshift {shift} not supported')


def denormalize(values: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Inverse of SpiceDataset.normalize_column with the real data's raw range `raw`."""
    raw = raw[~np.isnan(raw)]
    low, high = raw.min(), raw.max()
    if high == low:
        return values
    target_low, target_high = (-1 if low < 0 else 0), (1 if high > 0 else 0)
    return (values - target_low) / (target_high - target_low) * (high - low) + low


def discrete_columns(study: str, df: pd.DataFrame, rows: np.ndarray, actions: np.ndarray, features: np.ndarray,
                     inputs: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Choice, feedback (single column, partial feedback) and the given additional inputs, in raw units."""
    spice_module = importlib.import_module(f'studies.{study}.spice_{study}')
    kwargs = study_data(study)[0]
    choice_column, feedback_column = kwargs.get('df_choice', 'choice'), kwargs.get('df_feedback', 'reward')
    n_actions = actions.shape[-1]

    # choice indices are the sorted raw values (csv_to_dataset: numeric minus the minimum, strings alphabetical)
    action = np.where(np.isnan(actions).any(axis=-1), -1, np.nan_to_num(actions).argmax(axis=-1))
    codes = np.sort(df[choice_column].dropna().unique())
    if not isinstance(codes[0], str):
        codes = codes.min() + np.arange(n_actions)
    columns = {choice_column: np.where(action >= 0, codes[np.maximum(action, 0)], np.nan)}

    if feedback_column is not None:
        reward = np.take_along_axis(features[..., n_actions:2 * n_actions], np.maximum(action, 0)[..., None], axis=-1)[..., 0]
        reward[action < 0] = np.nan
        if getattr(spice_module, 'NORMALIZE_REWARDS', True):
            slots = rows[:, :-1][rows[:, :-1] >= 0]
            reward = denormalize(reward, df[feedback_column].to_numpy(dtype=float)[slots])
        columns[feedback_column] = reward

    names = list(kwargs.get('additional_inputs') or [])
    shifts = kwargs.get('timeshift_additional_inputs') or [0] * len(names)
    n_rewards = n_actions if feedback_column is not None else 0
    for name in inputs:
        index = names.index(name)
        columns[name] = shifted(features[..., n_actions + n_rewards + index], shifts[index])
    return columns


def bruckner2025_columns(study, df, rows, actions, features) -> dict[str, np.ndarray]:
    """Bucket position b_t and the next trial's start position z_t (prepare_dataframe: z_next = z_t of the next row)."""
    from weinhardt2026.studies.bruckner2025.benchmarking_bruckner2025 import POSITION_SCALE
    from weinhardt2026.studies.bruckner2025.spice_bruckner2025 import CONFIG
    z_next = features[..., 1 + 1 + CONFIG.additional_inputs.index('z_next')]
    return {'b_t': actions[..., 0] * POSITION_SCALE, 'z_t': shifted(z_next, -1) * POSITION_SCALE}


def weber2024_columns(study, df, rows, actions, features) -> dict[str, np.ndarray]:
    """Shield angle (degrees, from sin/cos) and whether the laser was caught."""
    from weinhardt2026.studies.weber2024.benchmarking_weber2024 import _AI_CAUGHT, _AI_START
    return {
        'shieldRotation': np.degrees(np.arctan2(actions[..., 0], actions[..., 1])) % 360,
        'laser_caught': features[..., _AI_START + _AI_CAUGHT],
    }


# raw columns a simulation changes (all others are task settings the generators replay)
SIMULATED_COLUMNS = {
    'dezfouli2019': discrete_columns,
    'ganesh2024a': discrete_columns,
    'eckstein2026': discrete_columns,
    'bustamante2023': discrete_columns,
    'braun2018': lambda *args: discrete_columns(*args, inputs=('difference', 'current', 'other')),
    'bruckner2025': bruckner2025_columns,
    'weber2024': weber2024_columns,
    'kolff2025': lambda *args: discrete_columns(*args, inputs=kolff2025_self_columns()),
}


def kolff2025_self_columns() -> tuple[str, ...]:
    """The participant's own acts (its grooming is simulated; the partner's acts and the timing are replayed)."""
    from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import SELF_COLUMNS
    return SELF_COLUMNS


# raw action column, where it is not the df_choice of the data kwargs
ACTION_COLUMNS = {'bruckner2025': 'b_t', 'weber2024': 'shieldRotation'}


def simulated_dataframe(study: str, df: pd.DataFrame, rows: np.ndarray, dataset_sim: SpiceDataset) -> pd.DataFrame:
    """The real data with the simulated columns overwritten.

    Cells the simulation sets no value for are NaN, except actions: there the real action is kept, since
    csv_to_dataset drops rows without one (and sessions without any). These actions never enter a dataset:
    they are the last trial of a session that prepare_dataset cuts (ganesh2024a) or the only trial of a
    single-trial session (kolff2025).
    """
    actions, features = simulated_positions(dataset_sim, n_actions_of(study, dataset_sim), rows.shape[1])
    action_column = ACTION_COLUMNS.get(study, study_data(study)[0].get('df_choice', 'choice'))
    df_sim = df.copy()
    keep = rows >= 0
    for column, values in SIMULATED_COLUMNS[study](study, df, rows, actions, features).items():
        cells = df[column].to_numpy(dtype=float).copy() if column == action_column else np.full(len(df), np.nan)
        simulated = values[keep]
        cells[rows[keep][~np.isnan(simulated)]] = simulated[~np.isnan(simulated)]
        df_sim[column] = cells
    return df_sim


def check_roundtrip(study: str, path: Path, dataset_sim: SpiceDataset) -> None:
    """The CSV loaded as the fit loads it must reproduce the simulated choices; prints the largest deviation per block."""
    dataset = load_csv(study, path)
    n_slots = min(dataset.xs.shape[1], dataset_sim.xs.shape[1])
    xs, xs_sim = dataset.xs[:, :n_slots, 0].numpy(), dataset_sim.xs[:, :n_slots, 0].numpy()
    ys, ys_sim = dataset.ys[:, :n_slots, 0].numpy(), dataset_sim.ys[:, :n_slots, 0].numpy()
    both = ~np.isnan(xs[..., 0]) & ~np.isnan(xs_sim[..., 0])
    n_actions, n_rewards = n_actions_of(study, dataset), dataset.n_reward_features

    def deviation(a, b):
        difference = np.abs(a[both] - b[both])
        return float(np.nanmax(difference)) if difference.size else 0.

    report = {
        'actions': deviation(xs[..., :n_actions], xs_sim[..., :n_actions]),
        'targets': deviation(ys, ys_sim),
        'rewards': deviation(xs[..., n_actions:n_actions + n_rewards], xs_sim[..., n_actions:n_actions + n_rewards]),
        'inputs': deviation(xs[..., n_actions + n_rewards:-5], xs_sim[..., n_actions + n_rewards:-5]),
    }
    print(f'{path.name}: round trip, largest deviation per block ' + ', '.join(f'{k} {v:.2e}' for k, v in report.items())
          + f'; slots {int(both.sum())} of {int((~np.isnan(xs_sim[..., 0])).sum())} simulated')
    if report['actions'] > 1e-4 or report['targets'] > 1e-4:
        raise RuntimeError(f'{path.name}: the CSV does not reproduce the simulated behavior')


# -------------------------------------------------------------------------------------------
# REFIT CONFIGURATION: THE FINAL HYPERPARAMETERS OF BOTH MODELS
# -------------------------------------------------------------------------------------------

def fit_config(study: str) -> dict:
    """Split and final hyperparameters, read off the final checkpoints (and the benchmark's grid search run)."""
    from weinhardt2026.utils.final_models import final_checkpoints

    data_kwargs, test_blocks = study_data(study)
    config = dict(module=f'studies.{study}.spice_{study}', data_kwargs=json.dumps(data_kwargs),
                  test_blocks=','.join(str(block) for block in test_blocks))

    path_spice = final_checkpoints(study, 'spice')[0]
    config.update({f'spice_{name}': float(value) for name, value in SPICE_TAGS.search(os.path.basename(path_spice)).groupdict().items()})
    config['spice_extra_arguments'] = SPICE_EXTRA_ARGUMENTS.get(study, '')

    if study == 'kolff2025':   # closed-form benchmark: no training hyperparameters
        return config
    path_benchmark = final_checkpoints(study, 'benchmark')[0]
    tags = BENCHMARK_TAGS.search(os.path.basename(path_benchmark)).groupdict()
    benchmark = dict(benchmark_lr=float(tags['lr']), benchmark_scheduler=int(tags['scheduler']), benchmark_epochs=1000, benchmark_init_std=0.5)
    run = BENCHMARK_RUNS_DIR / study / Path(path_benchmark).with_suffix('.json').name
    if run.exists():
        benchmark.update({f'benchmark_{key}': json.loads(run.read_text())[key] for key in ('epochs', 'init_std')})
    else:
        print(f'{study}: {run.name} not found, assuming the grid search defaults epochs=1000, init_std=0.5')

    return config | benchmark


# -------------------------------------------------------------------------------------------
# PIPELINE
# -------------------------------------------------------------------------------------------

def generate_study(study: str, out_dir: str, simulation_seed: int) -> None:
    from weinhardt2026.utils.study_models import load_study_module

    study_module = load_study_module(study)
    dataset = study_module.load_dataset()
    df = pd.read_csv(study_csv(study))
    rows = trial_rows(study, df)
    if not (dataset.xs.shape[0] == rows.shape[0] and dataset.xs.shape[1] <= rows.shape[1]):
        raise RuntimeError(f'{study}: the raw rows do not match the sessions of the dataset')
    config = fit_config(study)

    scores, generators = [], []
    for generator in GENERATORS:
        scores_generator = score_seeds(study, generator)
        selected = median_seed(scores_generator)
        scores_generator['selected'] = scores_generator['checkpoint'] == selected['checkpoint']
        scores.append(scores_generator)
        print(f"{study} {generator}: " + (f"median seed {selected['seed']} of {len(scores_generator)}" if selected['checkpoint'] != CLOSED_FORM else CLOSED_FORM)
              + (f", member {selected['member']}" if generator == 'spice' else '')
              + f" (training log-lik. / trial {selected['loglik_train']:.4f})")

        model = load_generator(study, generator, selected)
        kwargs, noise = {}, np.nan
        if dataset.continuous_action:
            kwargs['noise_std'] = response_noise_std(study, generator, model, dataset)
            noise = float(kwargs['noise_std'].mean())
            print(f'{study} {generator}: response noise s.d. {noise:.4f} (mean over sessions)')

        torch.manual_seed(simulation_seed)
        dataset_sim = study_module.generate(model, dataset, **kwargs)
        path = dataset_path(out_dir, study, generator)
        simulated_dataframe(study, df, rows, dataset_sim).to_csv(path, index=False)
        check_roundtrip(study, path, dataset_sim)

        generators.append(dict(
            study=study, generator=generator, data=os.path.relpath(path.resolve(), ROOT),
            checkpoint=selected['checkpoint'], seed=None if pd.isna(selected['seed']) else int(selected['seed']),
            member=None if generator == 'benchmark' else int(selected['member']),
            loglik_train=float(selected['loglik_train']), noise_std_mean=noise,
            simulation_seed=simulation_seed, **config,
        ))
        del model
        torch.cuda.empty_cache()

    upsert(Path(out_dir) / 'generators.csv', pd.DataFrame(generators), study)
    upsert(Path(out_dir) / 'selection.csv', pd.concat(scores, ignore_index=True), study)
    print(f'{study}: saved to {out_dir}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simulate one dataset per study from the median seed of SPICE and the benchmark.')
    parser.add_argument('--studies', nargs='+', default=list(STUDIES), choices=STUDIES)
    parser.add_argument('--simulation_seed', type=int, default=0, help='seed of the sampled actions, outcomes and response noise')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for study in args.studies:
        generate_study(study, args.out_dir, args.simulation_seed)
