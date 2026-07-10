import os
import torch
import pandas as pd
import numpy as np
from typing import Union
from tqdm import tqdm

from spice import SpiceEstimator, SpiceDataset, csv_to_dataset, split_data_along_blockdim, dataset_to_csv

from weinhardt2026.studies.kolff2025.spice_kolff2025 import CONFIG


# --- Constants ---

DEFAULT_DATA_PATH = 'weinhardt2026/studies/kolff2025/data/kolff2025.csv'
N_ACTIONS = 5  # action, grooming, gesture, scratching, waiting
N_TRIALS_DEFAULT = 20
WAITING_ACTION = 4
GROOMING_ACTION = 1
SCRATCHING_ACTION = 3
MAX_RANK = 26
LOWER_STATUS_HAS_LARGER_RANK_NUMBER = True
ACTION_NAMES = {0: 'action', 1: 'grooming', 2: 'gesture', 3: 'scratching', 4: 'waiting'}


# --- Benchmark Model ---

class ConditionalFrequencyModel(torch.nn.Module):
    """Baseline: P(action_id1 | action_id2, sender_id) from empirical frequencies.

    A per-chimp lookup table of action probabilities conditioned on the
    partner's current action. No temporal dynamics, no dyad-level effects.
    """

    def __init__(self, n_actions: int, n_participants: int, smoothing: float = 1.0):
        super().__init__()
        self.n_actions = n_actions
        self.n_participants = n_participants
        self.smoothing = smoothing
        # log P(my_action | partner_action, my_id)
        # shape: (n_participants, n_partner_actions, n_my_actions)
        self.register_buffer('log_probs', torch.zeros(n_participants, n_actions, n_actions))

    def fit(self, dataset: SpiceDataset, exclude_action: int = WAITING_ACTION) -> 'ConditionalFrequencyModel':
        xs, ys = dataset.xs, dataset.ys

        counts = torch.zeros(self.n_participants, self.n_actions, self.n_actions)

        valid = ~torch.isnan(xs[:, :, 0, 0]) & ~torch.isnan(ys[:, :, 0, 0])
        participant_ids = xs[:, 0, 0, -1].nan_to_num(0).long()
        partner_actions = xs[:, :, 0, self.n_actions].nan_to_num(0).long()
        target_actions = ys[:, :, 0, :].argmax(dim=-1)

        for s in range(xs.shape[0]):
            pid = participant_ids[s]
            for t in range(xs.shape[1]):
                if not valid[s, t]:
                    continue
                target = target_actions[s, t].item()
                if exclude_action is not None and target == exclude_action:
                    continue
                counts[pid, partner_actions[s, t], target] += 1

        counts += self.smoothing
        self.log_probs = torch.log(counts / counts.sum(dim=-1, keepdim=True))
        return self

    def forward(self, xs: torch.Tensor, prev_state=None) -> tuple[torch.Tensor, None]:
        B, T, W, F = xs.shape

        participant_ids = xs[:, 0, 0, -1].nan_to_num(0).long()
        partner_actions = xs[:, :, 0, self.n_actions].nan_to_num(0).long()

        logits = self.log_probs[
            participant_ids.unsqueeze(1).expand(-1, T),
            partner_actions,
        ]  # (B, T, n_actions)

        valid = ~torch.isnan(xs[:, :, 0, 0])
        logits[~valid] = float('nan')

        return logits.unsqueeze(2), None  # (B, T, 1, n_actions)

    def count_parameters(self) -> int:
        return self.n_actions * (self.n_actions - 1)


# --- Data Loading ---

def get_dataset(
    path_data: str = None,
    test_blocks: tuple[int] = None,
) -> tuple[SpiceDataset, SpiceDataset, dict]:

    if path_data is None:
        path_data = DEFAULT_DATA_PATH

    dataset = csv_to_dataset(
        file=path_data,
        df_participant_id='ID1',
        df_choice='SigAct_ID1',
        df_feedback=None,
        additional_inputs=CONFIG.additional_inputs,
    )

    n_actions = dataset.n_actions

    # Remap participant_id -> ID1 (sender)
    # dataset.xs[..., -1] = dataset.xs[..., n_actions + 1].nan_to_num(0)

    # n_participants = dataset.xs[..., -1].nan_to_num(0).max().int().item() + 1

    # Normalize rank columns to [0, 1] (rank_diff in model will be in [-1, 1])
    rank_col_start = n_actions + 3  # after SigAct_ID2, ID1, ID2
    dataset.xs[..., rank_col_start] /= MAX_RANK
    dataset.xs[..., rank_col_start + 1] /= MAX_RANK

    if test_blocks is None:
        test_blocks = (1,)
    if test_blocks:
        dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)
    else:
        dataset_train = dataset_test = dataset

    info_dataset = {
        'n_participants': dataset.n_participants,
        'n_actions': dataset.n_actions,
    }

    return dataset_train, dataset_test, info_dataset


def get_pair_table(dataset: SpiceDataset) -> pd.DataFrame:
    """Extract unique (ID1, ID2) pairs and their dominance ranks from the dataset.

    Returns a DataFrame with columns: id1, id2, rank1, rank2, sender_lower_rank.
    """
    # Read raw CSV to get dominance ranks
    df = pd.read_csv(DEFAULT_DATA_PATH)
    pairs = df.groupby(['ID1', 'ID2']).agg({
        'Dominance_rank_ID1': 'first',
        'Dominance_rank_ID2': 'first',
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
        ('ID1', 'Dominance_rank_ID1'),
        ('ID2', 'Dominance_rank_ID2'),
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


def _extract_session_ranks(dataset: SpiceDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized rank1 and rank2 per session (already normalized in get_dataset)."""
    xs = dataset.xs[:, :, 0, :]
    valid = ~torch.isnan(xs[..., 0])
    n_actions = dataset.n_actions
    rank_col_start = n_actions + 3

    rank1_list = []
    rank2_list = []
    for session_idx in range(xs.shape[0]):
        valid_trials = torch.where(valid[session_idx])[0]
        if len(valid_trials) == 0:
            continue
        first_trial = valid_trials[0]
        rank1_list.append(xs[session_idx, first_trial, rank_col_start])
        rank2_list.append(xs[session_idx, first_trial, rank_col_start + 1])

    return torch.stack(rank1_list), torch.stack(rank2_list)


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

    # Unwrap SpiceEstimator
    if isinstance(model, SpiceEstimator):
        inner_model = model.model
    else:
        inner_model = model
    inner_model.eval()

    if hasattr(inner_model, 'device'):
        device = inner_model.device
    else:
        try:
            device = next(inner_model.parameters()).device
        except StopIteration:
            try:
                device = next(inner_model.buffers()).device
            except StopIteration:
                device = torch.device('cpu')

    n_actions = dataset.n_actions
    n_features = dataset.xs.shape[-1]

    pairs = _extract_session_pairs(dataset).to(device)
    rank1s, rank2s = _extract_session_ranks(dataset)
    rank1s = rank1s.to(device)
    rank2s = rank2s.to(device)
    n_pairs = pairs.shape[0]

    id1s = pairs[:, 0]
    id2s = pairs[:, 1]

    xs_gen = torch.full((n_pairs, n_trials, 1, n_features), float('nan'), device=device)
    ys_gen = torch.full((n_pairs, n_trials, 1, n_actions), float('nan'), device=device)

    state_ape1 = None
    state_ape2 = None

    # Initial conditions: ape1 has a random seed action;
    # ape2's own previous action is waiting.
    if initial_sender_action is None:
        action_ape1 = torch.randint(0, n_actions, (n_pairs,), device=device)
    else:
        action_ape1 = torch.full((n_pairs,), initial_sender_action, dtype=torch.long, device=device)
    action_ape2 = torch.full((n_pairs,), waiting_action, dtype=torch.long, device=device)

    for t in tqdm(range(n_trials)):
        # Ape2 responds to ape1's previous action.
        obs_ape2 = _build_observation(
            own_action=action_ape2,
            partner_action=action_ape1,
            sender_id=id2s,
            receiver_id=id1s,
            sender_rank=rank2s,
            receiver_rank=rank1s,
            trial_idx=t,
            n_actions=n_actions,
            n_features=n_features,
            device=device,
        )
        logits_ape2, state_ape2 = inner_model(obs_ape2, state_ape2)
        action_ape2 = _sample_action(logits_ape2, n_actions)

        # Ape1 responds to ape2.
        own_action_ape1 = action_ape1
        obs_ape1 = _build_observation(
            own_action=own_action_ape1,
            partner_action=action_ape2,
            sender_id=id1s,
            receiver_id=id2s,
            sender_rank=rank1s,
            receiver_rank=rank2s,
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
            sender_rank=rank1s,
            receiver_rank=rank2s,
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
    sender_rank: torch.Tensor,
    receiver_rank: torch.Tensor,
    n_actions: int,
) -> None:
    """Write one generated response and the model input that produced it."""
    xs_gen[:, trial_idx, 0, :n_actions] = torch.nn.functional.one_hot(
        own_action, n_actions,
    ).float()
    xs_gen[:, trial_idx, 0, n_actions] = partner_action.float()       # SigAct_ID2
    xs_gen[:, trial_idx, 0, n_actions + 1] = sender_id.float()        # ID1
    xs_gen[:, trial_idx, 0, n_actions + 2] = receiver_id.float()      # ID2
    xs_gen[:, trial_idx, 0, n_actions + 3] = sender_rank.float()      # Dominance_rank_ID1 (normalized)
    xs_gen[:, trial_idx, 0, n_actions + 4] = receiver_rank.float()    # Dominance_rank_ID2 (normalized)

    xs_gen[:, trial_idx, 0, -5] = 0
    xs_gen[:, trial_idx, 0, -4] = float(trial_idx)
    xs_gen[:, trial_idx, 0, -3] = 0
    xs_gen[:, trial_idx, 0, -2] = 0                                    # experiment_id (unused)
    xs_gen[:, trial_idx, 0, -1] = sender_id.float()

    ys_gen[:, trial_idx, 0, :] = torch.nn.functional.one_hot(
        generated_action, n_actions,
    ).float()


def _build_observation(
    own_action: torch.Tensor,
    partner_action: torch.Tensor,
    sender_id: torch.Tensor,
    receiver_id: torch.Tensor,
    sender_rank: torch.Tensor,
    receiver_rank: torch.Tensor,
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
        receiver_id: (n_pairs,) int - partner ape ID.
        sender_rank: (n_pairs,) float - focal ape normalized rank.
        receiver_rank: (n_pairs,) float - partner ape normalized rank.
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
    obs[:, 0, 0, n_actions + 3] = sender_rank.float()
    obs[:, 0, 0, n_actions + 4] = receiver_rank.float()
    obs[:, 0, 0, -5] = 0
    obs[:, 0, 0, -4] = float(trial_idx)
    obs[:, 0, 0, -3] = 0
    obs[:, 0, 0, -2] = 0               # experiment_id (unused)
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
            'Dominance_rank_ID1': rank_map.get(id1, np.nan),
            'Dominance_rank_ID2': rank_map.get(id2, np.nan),
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
                id2=int(xs[s, t, 0, n_actions + 2]),
                sigact_id1=int(xs[s, t, 0, :n_actions].argmax()),
                sigact_id2=int(xs[s, t, 0, n_actions]),
                block=int(xs[s, t, 0, -3]),
            )

        last_t = n_valid - 1
        append_row(
            session_idx=s,
            row_idx=n_valid,
            id1=int(xs[s, last_t, 0, -1]),
            id2=int(xs[s, last_t, 0, n_actions + 2]),
            sigact_id1=int(ys[s, last_t, 0, :n_actions].argmax()),
            sigact_id2=int(xs[s, last_t, 0, n_actions]),
            block=int(xs[s, last_t, 0, -3]),
        )

    columns = [
        'start', 'stop', 'ID1', 'ID2',
        'Dominance_rank_ID1', 'Dominance_rank_ID2',
        'SigAct_ID1', 'SigAct_ID2', 'interaction_id', 'community_id',
        'Grooming_ID1', 'Grooming_ID2', 'block',
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


@torch.no_grad()
def generate_behavior_replay(
    model: Union[SpiceEstimator, torch.nn.Module],
    dataset: SpiceDataset,
    n_repeats: int = 100,
    save_csv: str = None,
    source_csv: str = DEFAULT_DATA_PATH,
    waiting_action: int = WAITING_ACTION,
) -> SpiceDataset:
    """Generate synthetic behavior by replaying empirical turn-taking structure.

    Preserves the empirical sequence of who sends when and only generates
    ID1's action content autoregressively. ID2's actions are taken directly
    from the empirical data (the model is only trained to predict ID1
    behavior, so generating ID2 actions would be unreliable). Each empirical
    interaction is repeated ``n_repeats`` times to yield stable per-session
    statistics.

    Args:
        model: Fitted SPICE model (SpiceEstimator or nn.Module).
        dataset: Original empirical dataset (turn structure is read from here).
        n_repeats: Number of independent stochastic repetitions per session.
        save_csv: Optional path to save generated behavior as CSV.
        source_csv: Empirical CSV path used for rank/community metadata.
        waiting_action: Action index that indicates "not sending".

    Returns:
        SpiceDataset with generated dyadic behavior
        (n_sessions * n_repeats sessions).
    """
    print(f"Generating replay behavior ({n_repeats} repeats per session)...")

    # Unwrap SpiceEstimator
    if isinstance(model, SpiceEstimator):
        inner_model = model.model
    else:
        inner_model = model
    inner_model.eval()

    if hasattr(inner_model, 'device'):
        device = inner_model.device
    else:
        try:
            device = next(inner_model.parameters()).device
        except StopIteration:
            try:
                device = next(inner_model.buffers()).device
            except StopIteration:
                device = torch.device('cpu')

    n_actions = dataset.n_actions
    n_features = dataset.xs.shape[-1]
    xs_emp = dataset.xs[:, :, 0, :]  # (S, T, F)
    n_sessions = xs_emp.shape[0]
    rank_col_start = n_actions + 3

    # Per-session metadata
    valid_mask = ~torch.isnan(xs_emp[..., 0])  # (S, T)
    n_valid_per_session = valid_mask.sum(dim=1).long()  # (S,)
    max_trials = int(n_valid_per_session.max().item())

    # Allocate output
    n_out = n_sessions * n_repeats
    xs_gen = torch.full((n_out, max_trials, 1, n_features), float('nan'))
    ys_gen = torch.full((n_out, max_trials, 1, n_actions), float('nan'))

    for s in tqdm(range(n_sessions)):
        n_valid = int(n_valid_per_session[s].item())
        if n_valid == 0:
            continue

        # Extract session metadata from first valid trial
        first_t = int(torch.where(valid_mask[s])[0][0].item())
        id1 = xs_emp[s, first_t, -1].nan_to_num(0).long().to(device)
        id2 = xs_emp[s, first_t, n_actions + 2].nan_to_num(0).long().to(device)
        rank1 = xs_emp[s, first_t, rank_col_start].to(device)
        rank2 = xs_emp[s, first_t, rank_col_start + 1].to(device)

        # Replicate scalars to (n_repeats,) for _build_observation
        id1_batch = id1.expand(n_repeats)
        id2_batch = id2.expand(n_repeats)
        rank1_batch = rank1.expand(n_repeats)
        rank2_batch = rank2.expand(n_repeats)

        # Turn structure and empirical ID2 actions from data
        is_id1_sender = (
            xs_emp[s, :n_valid, :n_actions].argmax(dim=-1) != waiting_action
        )  # (n_valid,)
        emp_id2_actions = xs_emp[s, :n_valid, n_actions].nan_to_num(0).long().to(device)

        # Initialize — only track ID1 state (ID2 actions come from data)
        state_id1 = None
        prev_id1 = torch.full((n_repeats,), waiting_action, dtype=torch.long, device=device)

        actions_id1 = []  # list of (n_repeats,) tensors

        for t in range(n_valid):
            emp_id2 = emp_id2_actions[t].expand(n_repeats)

            # Build observation for ID1 using empirical ID2 action
            obs_id1 = _build_observation(
                own_action=prev_id1, partner_action=emp_id2,
                sender_id=id1_batch, receiver_id=id2_batch,
                sender_rank=rank1_batch, receiver_rank=rank2_batch,
                trial_idx=t, n_actions=n_actions, n_features=n_features,
                device=device,
            )
            logits_id1, state_id1 = inner_model(obs_id1, state_id1)

            if is_id1_sender[t]:
                curr_id1 = _sample_action(logits_id1, n_actions)
            else:
                curr_id1 = torch.full((n_repeats,), waiting_action, dtype=torch.long, device=device)

            actions_id1.append(curr_id1.cpu())
            prev_id1 = curr_id1

        # Write xs_gen / ys_gen for this session's repeats
        out_start = s * n_repeats
        out_end = out_start + n_repeats

        for t in range(n_valid):
            xs_gen[out_start:out_end, t, 0, :n_actions] = torch.nn.functional.one_hot(
                actions_id1[t], n_actions,
            ).float()
            xs_gen[out_start:out_end, t, 0, n_actions] = emp_id2_actions[t].cpu().float()
            xs_gen[out_start:out_end, t, 0, n_actions + 1] = id1.cpu().float()
            xs_gen[out_start:out_end, t, 0, n_actions + 2] = id2.cpu().float()
            xs_gen[out_start:out_end, t, 0, n_actions + 3] = rank1.cpu().float()
            xs_gen[out_start:out_end, t, 0, n_actions + 4] = rank2.cpu().float()
            xs_gen[out_start:out_end, t, 0, -5] = 0
            xs_gen[out_start:out_end, t, 0, -4] = float(t)
            xs_gen[out_start:out_end, t, 0, -3] = 0
            xs_gen[out_start:out_end, t, 0, -2] = 0
            xs_gen[out_start:out_end, t, 0, -1] = id1.cpu().float()

            if t < n_valid - 1:
                ys_gen[out_start:out_end, t, 0, :] = torch.nn.functional.one_hot(
                    actions_id1[t + 1], n_actions,
                ).float()

    dataset_gen = SpiceDataset(xs_gen, ys_gen, n_reward_features=0)

    if save_csv is not None:
        _save_generated_csv(dataset_gen, save_csv, source_csv=source_csv)

    print("Done generating replay behavior.")
    return dataset_gen
