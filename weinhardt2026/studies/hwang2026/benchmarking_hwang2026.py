import os
import torch
import pandas as pd
import numpy as np
from typing import Union
from tqdm import tqdm

from spice import SpiceEstimator, SpiceDataset, BaseModel, csv_to_dataset, split_data_along_blockdim, dataset_to_csv

from weinhardt2026.studies.hwang2026.spice_hwang2026 import CONFIG


# --- Constants ---

DEFAULT_DATA_PATH = 'weinhardt2026/studies/hwang2026/data/hwang2025_processed.csv'
N_ACTIONS = 4  # TODO: set to 5 when working with scratching data
N_TRIALS_DEFAULT = 20
WAITING_ACTION = 3
GROOMING_ACTION = 1
LOWER_STATUS_HAS_LARGER_RANK_NUMBER = True
# TODO: replace when working with scratching data
# ACTION_NAMES = {0: 'action', 1: 'grooming', 2: 'gesture', 3: 'scratching', 4: 'waiting'}
ACTION_NAMES = {0: 'action', 1: 'grooming', 2: 'gesture', 3: 'waiting'}


# --- Data Loading ---

def get_dataset(
    path_data: str = None,
    test_blocks: tuple[int] = None,
) -> tuple[SpiceDataset, SpiceDataset, dict]:

    if path_data is None:
        path_data = DEFAULT_DATA_PATH

    dataset = csv_to_dataset(
        file=path_data,
        df_participant_id='interaction_id',
        df_choice='SigAct_ID1',
        df_feedback=None,
        additional_inputs=CONFIG.additional_inputs,
    )

    n_actions = dataset.n_actions

    # Remap participant_id -> ID1, experiment_id -> ID2
    dataset.xs[..., -1] = dataset.xs[..., n_actions + 1].nan_to_num(0)
    dataset.xs[..., -2] = dataset.xs[..., n_actions + 2].nan_to_num(0)

    n_participants = dataset.xs[..., [-1, -2]].nan_to_num(0).max().int().item() + 1

    if test_blocks is None:
        test_blocks = (1,)
    if test_blocks:
        dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)
    else:
        dataset_train = dataset_test = dataset

    info_dataset = {
        'n_participants': n_participants,
        'n_actions': n_actions,
    }

    return dataset_train, dataset_test, info_dataset


def get_pair_table(dataset: SpiceDataset) -> pd.DataFrame:
    """Extract unique (ID1, ID2) pairs and their dominance ranks from the dataset.

    Returns a DataFrame with columns: id1, id2, rank1, rank2, sender_lower_rank.
    """
    # Read raw CSV to get dominance ranks
    df = pd.read_csv(DEFAULT_DATA_PATH)
    pairs = df.groupby(['ID1', 'ID2']).agg({
        'Dominance rank_ID1': 'first',
        'Dominance rank_ID2': 'first',
    }).reset_index()
    pairs.columns = ['id1', 'id2', 'rank1', 'rank2']
    if LOWER_STATUS_HAS_LARGER_RANK_NUMBER:
        pairs['sender_lower_rank'] = pairs['rank1'] > pairs['rank2']
    else:
        pairs['sender_lower_rank'] = pairs['rank1'] < pairs['rank2']
    return pairs


# --- Dyadic Behavior Generation ---

def _load_empirical_metadata(path_data: str = DEFAULT_DATA_PATH) -> tuple[dict[int, int], dict[tuple[int, int], dict]]:
    """Load rank and pair-level metadata used when writing generated CSV files."""
    if path_data is None or not os.path.exists(path_data):
        return {}, {}

    df = pd.read_csv(path_data)
    rank_map = {}
    pair_metadata = {}

    for id_col, rank_col in (
        ('ID1', 'Dominance rank_ID1'),
        ('ID2', 'Dominance rank_ID2'),
    ):
        if id_col not in df.columns or rank_col not in df.columns:
            continue
        for _, row in df[[id_col, rank_col]].dropna().drop_duplicates().iterrows():
            rank_map[int(row[id_col])] = int(row[rank_col])

    required_pair_cols = {'ID1', 'ID2'}
    if required_pair_cols.issubset(df.columns):
        for _, row in df.drop_duplicates(['ID1', 'ID2']).iterrows():
            pair = (int(row['ID1']), int(row['ID2']))
            pair_metadata[pair] = {
                'community_id': int(row['community_id']) if 'community_id' in df.columns else 0,
            }

    return rank_map, pair_metadata


def _extract_session_pairs(dataset: SpiceDataset) -> torch.Tensor:
    """Return one directed sender/receiver pair per empirical interaction session."""
    xs = dataset.xs[:, :, 0, :]
    valid = ~torch.isnan(xs[..., 0])
    id1 = xs[..., -1].nan_to_num(0).long()
    id2 = xs[..., -2].nan_to_num(0).long()

    pairs = []
    for session_idx in range(xs.shape[0]):
        valid_trials = torch.where(valid[session_idx])[0]
        if len(valid_trials) == 0:
            continue
        first_trial = valid_trials[0]
        pairs.append(torch.stack([id1[session_idx, first_trial], id2[session_idx, first_trial]]))

    if not pairs:
        raise ValueError("Cannot generate behavior from an empty dataset.")
    return torch.stack(pairs, dim=0)


@torch.no_grad()
def generate_behavior(
    model: Union[SpiceEstimator, torch.nn.Module],
    dataset: SpiceDataset,
    n_trials: int = N_TRIALS_DEFAULT,
    save_csv: str = None,
    source_csv: str = DEFAULT_DATA_PATH,
    waiting_action: int = WAITING_ACTION,
    initial_sender_action: int = None,
) -> SpiceDataset:
    """Generate synthetic behavior by letting two model instances communicate.

    For each empirical interaction session, two "chimps" with that session
    directed ID1/ID2 pair alternate turns:
      - ape1 starts with a random seed action
      - ape2 observes that seed action and responds
      - ape1 responds to ape2
      - this repeats for n_trials generated ape1 responses

    The same model is used for both chimps; the participant/experiment
    embeddings distinguish them. The returned dataset records the focal ape1
    sequence only, so each generated session has fixed ID1/ID2 columns like the
    empirical interaction_id groups.

    Args:
        model: Fitted SPICE model (SpiceEstimator or nn.Module).
        dataset: Original dataset (used to extract ID pairs and metadata).
        n_trials: Number of generated focal-ape responses per interaction session.
        save_csv: Optional path to save generated behavior as CSV.
        source_csv: Empirical CSV path used for rank/community metadata.
        waiting_action: Action index used as the receiver's initial previous action.
        initial_sender_action: Optional fixed seed action for ape1; if None, sampled uniformly.

    Returns:
        SpiceDataset with generated dyadic behavior.
    """
    print("Generating dyadic behavior...")

    if dataset.n_reward_features != 0 or dataset.n_additional_inputs < 3:
        raise ValueError("Hwang generation expects no rewards and additional inputs: SigAct_ID2, ID1, ID2.")

    # Unwrap SpiceEstimator
    if isinstance(model, SpiceEstimator):
        inner_model = model.model
    else:
        inner_model = model
    inner_model.eval()

    if hasattr(inner_model, 'device'):
        device = inner_model.device
    elif hasattr(inner_model, 'parameters'):
        device = next(inner_model.parameters()).device
    else:
        device = torch.device('cpu')

    n_actions = dataset.n_actions
    n_features = dataset.xs.shape[-1]

    pairs = _extract_session_pairs(dataset).to(device)
    n_pairs = pairs.shape[0]

    id1s = pairs[:, 0]
    id2s = pairs[:, 1]

    xs_gen = torch.full((n_pairs, n_trials, 1, n_features), float('nan'), device=device)
    ys_gen = torch.full((n_pairs, n_trials, 1, n_actions), float('nan'), device=device)

    state_ape1 = None
    state_ape2 = None

    # Initial conditions from the proposal: ape1 has a random seed action;
    # ape2's own previous action is waiting.
    if initial_sender_action is None:
        action_ape1 = torch.randint(0, n_actions, (n_pairs,), device=device)
    else:
        action_ape1 = torch.full((n_pairs,), initial_sender_action, dtype=torch.long, device=device)
    action_ape2 = torch.full((n_pairs,), waiting_action, dtype=torch.long, device=device)

    for t in tqdm(range(n_trials)):
        # Ape2 responds to ape1's previous action. This closes the dyadic loop
        # but is not written as a separate empirical-style focal sequence row.
        obs_ape2 = _build_observation(
            own_action=action_ape2,
            partner_action=action_ape1,
            sender_id=id2s,
            receiver_id=id1s,
            trial_idx=t,
            n_actions=n_actions,
            n_features=n_features,
            device=device,
        )
        logits_ape2, state_ape2 = inner_model(obs_ape2, state_ape2)
        action_ape2 = _sample_action(logits_ape2, n_actions)

        # Ape1 responds to ape2. This is the generated focal behavior that is
        # compared with empirical ID1 behavior.
        own_action_ape1 = action_ape1
        obs_ape1 = _build_observation(
            own_action=own_action_ape1,
            partner_action=action_ape2,
            sender_id=id1s,
            receiver_id=id2s,
            trial_idx=t,
            n_actions=n_actions,
            n_features=n_features,
            device=device,
        )
        logits_ape1, state_ape1 = inner_model(obs_ape1, state_ape1)
        action_ape1 = _sample_action(logits_ape1, n_actions)

        _write_generated_step(
            xs_gen=xs_gen,
            ys_gen=ys_gen,
            trial_idx=t,
            own_action=own_action_ape1,
            generated_action=action_ape1,
            partner_action=action_ape2,
            sender_id=id1s,
            receiver_id=id2s,
            n_actions=n_actions,
        )

    xs_gen = xs_gen.cpu()
    ys_gen = ys_gen.cpu()

    dataset_gen = SpiceDataset(xs_gen, ys_gen, n_reward_features=0)

    if save_csv is not None:
        _save_generated_csv(dataset_gen, save_csv, source_csv=source_csv)

    print("Done generating dyadic behavior.")
    return dataset_gen


def _write_generated_step(
    xs_gen: torch.Tensor,
    ys_gen: torch.Tensor,
    trial_idx: int,
    own_action: torch.Tensor,
    generated_action: torch.Tensor,
    partner_action: torch.Tensor,
    sender_id: torch.Tensor,
    receiver_id: torch.Tensor,
    n_actions: int,
) -> None:
    """Write one generated response and the model input that produced it."""
    xs_gen[:, trial_idx, 0, :n_actions] = torch.nn.functional.one_hot(
        own_action, n_actions,
    ).float()
    xs_gen[:, trial_idx, 0, n_actions] = partner_action.float()  # SigAct_ID2
    xs_gen[:, trial_idx, 0, n_actions + 1] = sender_id.float()   # ID1
    xs_gen[:, trial_idx, 0, n_actions + 2] = receiver_id.float() # ID2

    xs_gen[:, trial_idx, 0, -5] = 0
    xs_gen[:, trial_idx, 0, -4] = float(trial_idx)
    xs_gen[:, trial_idx, 0, -3] = 0
    xs_gen[:, trial_idx, 0, -2] = receiver_id.float()
    xs_gen[:, trial_idx, 0, -1] = sender_id.float()

    ys_gen[:, trial_idx, 0, :] = torch.nn.functional.one_hot(
        generated_action, n_actions,
    ).float()


def _build_observation(
    own_action: torch.Tensor,
    partner_action: torch.Tensor,
    sender_id: torch.Tensor,
    receiver_id: torch.Tensor,
    trial_idx: int,
    n_actions: int,
    n_features: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a single-trial observation tensor for the model.

    Args:
        own_action: (n_pairs,) int - focal ape's previous action.
        partner_action: (n_pairs,) int - partner ape's previous action.
        sender_id: (n_pairs,) int - focal ape ID (= participant_id).
        receiver_id: (n_pairs,) int - partner ape ID (= experiment_id).
        trial_idx: int - current trial index.
        n_actions: int - number of action categories.
        n_features: int - total feature dimension.
        device: torch device.

    Returns:
        obs: (n_pairs, 1, 1, n_features) tensor ready for model forward pass.
    """
    n_pairs = own_action.shape[0]
    obs = torch.zeros(n_pairs, 1, 1, n_features, device=device)

    obs[:, 0, 0, :n_actions] = torch.nn.functional.one_hot(own_action, n_actions).float()
    obs[:, 0, 0, n_actions] = partner_action.float()
    obs[:, 0, 0, n_actions + 1] = sender_id.float()
    obs[:, 0, 0, n_actions + 2] = receiver_id.float()
    obs[:, 0, 0, -5] = 0
    obs[:, 0, 0, -4] = float(trial_idx)
    obs[:, 0, 0, -3] = 0
    obs[:, 0, 0, -2] = receiver_id.float()
    obs[:, 0, 0, -1] = sender_id.float()

    return obs


def _sample_action(logits: torch.Tensor, n_actions: int) -> torch.Tensor:
    """Sample an action from model logits.

    Args:
        logits: Model output, possibly 5D (E, B, T, W, A) or 4D (B, T, W, A).
        n_actions: Number of actions.

    Returns:
        action_idx: (B,) sampled action indices.
    """
    if logits.dim() == 5:  # BaseModel: (E, B, T, W, A)
        logits = logits.mean(dim=0)
    if logits.dim() == 4:  # (B, T, W, A)
        logits_t = logits[:, -1, 0, :n_actions]
    else:  # (B, T, A)
        logits_t = logits[:, -1, :n_actions]

    probs = torch.softmax(logits_t, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def _save_generated_csv(
    dataset: SpiceDataset,
    path: str,
    source_csv: str = DEFAULT_DATA_PATH,
) -> None:
    """Save generated dataset to CSV matching the original data format."""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rank_map, pair_metadata = _load_empirical_metadata(source_csv)
    xs = dataset.xs.numpy()
    ys = dataset.ys.numpy()
    n_actions = dataset.n_actions
    n_sessions = xs.shape[0]

    def append_row(
        session_idx: int,
        row_idx: int,
        id1: int,
        id2: int,
        sigact_id1: int,
        sigact_id2: int,
        block: int,
    ) -> None:
        meta = (
            pair_metadata.get((id1, id2))
            or pair_metadata.get((id2, id1))
            or {}
        )
        rows.append({
            'start': row_idx,
            'stop': row_idx + 1,
            'ID1': id1,
            'ID2': id2,
            'Dominance rank_ID1': rank_map.get(id1, np.nan),
            'Dominance rank_ID2': rank_map.get(id2, np.nan),
            'SigAct_ID1': sigact_id1,
            'SigAct_ID2': sigact_id2,
            'interaction_id': session_idx,
            'community_id': meta.get('community_id', 0),
            'Grooming_ID1': int(sigact_id1 == GROOMING_ACTION),
            'Grooming_ID2': int(sigact_id2 == GROOMING_ACTION),
            'block': block,
        })

    rows = []
    for s in range(n_sessions):
        n_valid = int((~np.isnan(xs[s, :, 0, 0])).sum())
        if n_valid == 0:
            continue

        for t in range(n_valid):
            append_row(
                session_idx=s,
                row_idx=t,
                id1=int(xs[s, t, 0, -1]),
                id2=int(xs[s, t, 0, -2]),
                sigact_id1=int(xs[s, t, 0, :n_actions].argmax()),
                sigact_id2=int(xs[s, t, 0, n_actions]),
                block=int(xs[s, t, 0, -3]),
            )

        last_t = n_valid - 1
        append_row(
            session_idx=s,
            row_idx=n_valid,
            id1=int(xs[s, last_t, 0, -1]),
            id2=int(xs[s, last_t, 0, -2]),
            sigact_id1=int(ys[s, last_t, 0, :n_actions].argmax()),
            sigact_id2=int(xs[s, last_t, 0, n_actions]),
            block=int(xs[s, last_t, 0, -3]),
        )

    columns = [
        'start', 'stop', 'ID1', 'ID2',
        'Dominance rank_ID1', 'Dominance rank_ID2',
        'SigAct_ID1', 'SigAct_ID2', 'interaction_id', 'community_id',
        'Grooming_ID1', 'Grooming_ID2', 'block',
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
