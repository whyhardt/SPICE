import torch
import pandas as pd
import numpy as np
from typing import Union
from tqdm import tqdm

from spice import SpiceEstimator, SpiceDataset, BaseModel, csv_to_dataset, split_data_along_blockdim, dataset_to_csv


# --- Constants ---

N_ACTIONS = 4  # TODO: set to 5 when working with scratching data
N_TRIALS_DEFAULT = 20
# TODO: replace when working with scratching data
# ACTION_NAMES = {0: 'action', 1: 'grooming', 2: 'gesture', 3: 'scratching', 4: 'waiting'}
ACTION_NAMES = {0: 'action', 1: 'grooming', 2: 'gesture', 3: 'waiting'}


# --- Data Loading ---

def get_dataset(
    path_data: str = None,
    test_blocks: tuple[int] = None,
) -> tuple[SpiceDataset, SpiceDataset, dict]:

    if path_data is None:
        path_data = 'weinhardt2026/studies/hwang2026/data/hwang2025_processed.csv'

    dataset = csv_to_dataset(
        file=path_data,
        df_participant_id='interaction_id',
        df_choice='SigAct_ID1',
        df_feedback=None,
        additional_inputs=['SigAct_ID2', 'ID1', 'ID2'],
    )

    n_actions = dataset.n_actions

    # Remap participant_id → ID1, experiment_id → ID2
    dataset.xs[..., -1] = dataset.xs[..., n_actions + 1].nan_to_num(0)
    dataset.xs[..., -2] = dataset.xs[..., n_actions + 2].nan_to_num(0)

    n_participants = dataset.xs[:, 0, -1].nan_to_num(0).max().int().item() + 1

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
    path_data = 'weinhardt2026/studies/hwang2026/data/hwang2025_processed.csv'
    df = pd.read_csv(path_data)
    pairs = df.groupby(['ID1', 'ID2']).agg({
        'Dominance rank_ID1': 'first',
        'Dominance rank_ID2': 'first',
    }).reset_index()
    pairs.columns = ['id1', 'id2', 'rank1', 'rank2']
    pairs['sender_lower_rank'] = pairs['rank1'] < pairs['rank2']
    return pairs


# --- Dyadic Behavior Generation ---

@torch.no_grad()
def generate_behavior(
    model: Union[SpiceEstimator, torch.nn.Module],
    dataset: SpiceDataset,
    n_trials: int = N_TRIALS_DEFAULT,
    save_csv: str = None,
) -> SpiceDataset:
    """Generate synthetic behavior by letting two model instances communicate.

    For each (ID1, ID2) pair in the dataset, two "chimps" alternate turns:
      - Ape1 (sender=ID1, receiver=ID2) generates an action
      - Ape2 (sender=ID2, receiver=ID1) responds
      - They alternate for n_trials total exchanges

    The same model is used for both chimps — the participant/experiment
    embeddings distinguish them.

    Args:
        model: Fitted SPICE model (SpiceEstimator or nn.Module).
        dataset: Original dataset (used to extract ID pairs and metadata).
        n_trials: Number of exchange trials to generate per pair.
        save_csv: Optional path to save generated behavior as CSV.

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
    elif hasattr(inner_model, 'parameters'):
        device = next(inner_model.parameters()).device
    else:
        device = torch.device('cpu')

    n_actions = dataset.n_actions
    n_features = dataset.xs.shape[-1]

    # Extract unique (ID1, ID2) pairs from the dataset
    id1_all = dataset.xs[:, 0, 0, -1].nan_to_num(0).long()
    id2_all = dataset.xs[:, 0, 0, -2].nan_to_num(0).long()
    pairs = torch.stack([id1_all, id2_all], dim=1).unique(dim=0)
    n_pairs = pairs.shape[0]

    id1s = pairs[:, 0]  # (n_pairs,)
    id2s = pairs[:, 1]  # (n_pairs,)

    # Storage for generated sequences
    xs_gen = torch.full((n_pairs, n_trials, 1, n_features), float('nan'), device=device)
    ys_gen = torch.full((n_pairs, n_trials, 1, n_actions), float('nan'), device=device)

    # Fill constant metadata columns for all trials
    xs_gen[:, :, 0, -1] = id1s.unsqueeze(1).expand(-1, n_trials).float()  # participant_id = ID1
    xs_gen[:, :, 0, -2] = id2s.unsqueeze(1).expand(-1, n_trials).float()  # experiment_id = ID2
    xs_gen[:, :, 0, -3] = 0  # block
    xs_gen[:, :, 0, -4] = torch.arange(n_trials, device=device).unsqueeze(0).expand(n_pairs, -1).float()  # trial index
    xs_gen[:, :, 0, -5] = 0  # time_trial
    # Additional inputs: ID1 and ID2 columns
    xs_gen[:, :, 0, n_actions + 1] = id1s.unsqueeze(1).expand(-1, n_trials).float()
    xs_gen[:, :, 0, n_actions + 2] = id2s.unsqueeze(1).expand(-1, n_trials).float()

    # Hidden states for ape1 (sender=ID1) and ape2 (sender=ID2)
    state_ape1 = None
    state_ape2 = None

    # Initial actions: waiting (scratch) for both
    action_ape1 = torch.full((n_pairs,), 3, dtype=torch.long, device=device)  # scratch/waiting
    action_ape2 = torch.randint(0, n_actions, (n_pairs,), device=device)  # random initial from ape2

    for t in tqdm(range(n_trials)):
        if t % 2 == 0:
            # Ape1's turn: ape1 is sender, ape2 is receiver
            # Build observation for ape1: its own last action + ape2's last action as SigAct_ID2
            obs = _build_observation(
                own_action=action_ape1,
                partner_action=action_ape2,
                sender_id=id1s,
                receiver_id=id2s,
                trial_idx=t,
                n_actions=n_actions,
                n_features=n_features,
                device=device,
            )

            logits, state_ape1 = inner_model(obs, state_ape1)
            action_ape1 = _sample_action(logits, n_actions)

            # Record in dataset (from ape1's perspective)
            xs_gen[:, t, 0, :n_actions] = torch.nn.functional.one_hot(action_ape1, n_actions).float()
            xs_gen[:, t, 0, n_actions] = action_ape2.float()  # SigAct_ID2
            ys_gen[:, t, 0, :] = torch.nn.functional.one_hot(action_ape1, n_actions).float()
        else:
            # Ape2's turn: ape2 is sender, ape1 is receiver
            # FLIP: sender=ID2, receiver=ID1
            obs = _build_observation(
                own_action=action_ape2,
                partner_action=action_ape1,
                sender_id=id2s,
                receiver_id=id1s,
                trial_idx=t,
                n_actions=n_actions,
                n_features=n_features,
                device=device,
            )

            logits, state_ape2 = inner_model(obs, state_ape2)
            action_ape2 = _sample_action(logits, n_actions)

            # Record in dataset (from ape1's perspective for consistency)
            xs_gen[:, t, 0, :n_actions] = torch.nn.functional.one_hot(action_ape1, n_actions).float()
            xs_gen[:, t, 0, n_actions] = action_ape2.float()  # SigAct_ID2
            ys_gen[:, t, 0, :] = torch.nn.functional.one_hot(action_ape2, n_actions).float()

    xs_gen = xs_gen.cpu()
    ys_gen = ys_gen.cpu()

    dataset_gen = SpiceDataset(xs_gen, ys_gen, n_reward_features=0)

    if save_csv is not None:
        _save_generated_csv(dataset_gen, save_csv)

    print("Done generating dyadic behavior.")
    return dataset_gen


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
        own_action: (n_pairs,) int — sender's last action.
        partner_action: (n_pairs,) int — receiver's last action.
        sender_id: (n_pairs,) int — sender ape ID (= participant_id).
        receiver_id: (n_pairs,) int — receiver ape ID (= experiment_id).
        trial_idx: int — current trial index.
        n_actions: int — number of action categories.
        n_features: int — total feature dimension.
        device: torch device.

    Returns:
        obs: (n_pairs, 1, 1, n_features) tensor ready for model forward pass.
    """
    n_pairs = own_action.shape[0]
    obs = torch.zeros(n_pairs, 1, 1, n_features, device=device)

    # Own action (one-hot)
    obs[:, 0, 0, :n_actions] = torch.nn.functional.one_hot(own_action, n_actions).float()
    # Partner's action as SigAct_ID2 (integer)
    obs[:, 0, 0, n_actions] = partner_action.float()
    # ID1 (sender) and ID2 (receiver) in additional inputs
    obs[:, 0, 0, n_actions + 1] = sender_id.float()
    obs[:, 0, 0, n_actions + 2] = receiver_id.float()
    # Metadata
    obs[:, 0, 0, -5] = 0  # time_trial
    obs[:, 0, 0, -4] = float(trial_idx)  # trial index
    obs[:, 0, 0, -3] = 0  # block
    obs[:, 0, 0, -2] = receiver_id.float()  # experiment_id = receiver
    obs[:, 0, 0, -1] = sender_id.float()  # participant_id = sender

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
        logits = logits.mean(dim=0)  # ensemble mean
    if logits.dim() == 4:  # (B, T, W, A)
        logits_t = logits[:, -1, 0, :n_actions]
    else:  # (B, T, A)
        logits_t = logits[:, -1, :n_actions]

    probs = torch.softmax(logits_t, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def _save_generated_csv(dataset: SpiceDataset, path: str) -> None:
    """Save generated dataset to CSV matching the original data format."""
    xs = dataset.xs.numpy()
    ys = dataset.ys.numpy()
    n_actions = dataset.n_actions
    n_sessions = xs.shape[0]

    rows = []
    for s in range(n_sessions):
        n_valid = int((~np.isnan(xs[s, :, 0, 0])).sum())
        for t in range(n_valid):
            sigact_id1 = int(xs[s, t, 0, :n_actions].argmax())
            rows.append({
                'ID1': int(xs[s, t, 0, -1]),
                'ID2': int(xs[s, t, 0, -2]),
                'SigAct_ID1': sigact_id1,
                'SigAct_ID2': int(xs[s, t, 0, n_actions]),
                'interaction_id': s,
                'block': int(xs[s, t, 0, -3]),
            })

    pd.DataFrame(rows).to_csv(path, index=False)
