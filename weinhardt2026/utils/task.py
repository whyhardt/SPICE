import torch
from typing import Union
from tqdm import tqdm

from spice import SpiceEstimator, SpiceDataset, BaseModel, dataset_to_csv


class Env:
    """Base class for task environments used in generative benchmarking.

    Subclasses implement the task-specific reward mechanics via `reset()` and `step()`.
    Both methods operate on batched sessions (one entry per session = participant x block).
    """

    def __init__(self, n_actions: int, n_participants: int, n_blocks: int):
        self.n_actions = n_actions
        self.n_participants = n_participants
        self.n_blocks = n_blocks

    @property
    def n_sessions(self) -> int:
        return self.n_participants * self.n_blocks

    def reset(self, block_ids: torch.Tensor, participant_ids: torch.Tensor = None) -> None:
        """Reset environment for all sessions.

        Args:
            block_ids: (n_sessions,) block index per session (values from dataset metadata).
            participant_ids: (n_sessions,) participant index per session.
        """
        pass

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one trial for all sessions.

        Args:
            action: (n_sessions,) integer action indices.

        Returns:
            reward: (n_sessions,) float reward per session.
            terminated: (n_sessions,) bool termination flags.
        """
        raise NotImplementedError


@torch.no_grad()
def generate_behavior(
    dataset: SpiceDataset,
    model: Union[SpiceEstimator, torch.nn.Module],
    environment: Env,
    save_dataset: str = None,
    kwargs_dataset: dict = None,
    ) -> SpiceDataset:
    """Generate synthetic behavioral data by running a model through a task environment.

    Replays the experimental structure from `dataset` (same sessions, trial counts, metadata)
    but with model-generated actions and environment-generated rewards.

    Args:
        dataset: Original dataset (used for structure: session count, trial count, metadata).
        model: Fitted model (SpiceEstimator or torch.nn.Module).
        environment: Task environment that provides rewards given actions.
        save_dataset: Optional path to save generated dataset as CSV.
        kwargs_dataset: Optional kwargs forwarded to dataset_to_csv.

    Returns:
        SpiceDataset with generated behavior.
    """
    print("Generating behavior...")

    n_sessions, n_trials, _, n_features = dataset.xs.shape
    n_actions = dataset.n_actions
    n_rewards = dataset.n_reward_features

    # Extract metadata from dataset
    block_ids = dataset.xs[:, 0, 0, -3].long()
    participant_ids = dataset.xs[:, 0, 0, -1].long()

    # Reset environment for all sessions
    environment.reset(block_ids=block_ids, participant_ids=participant_ids)

    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'parameters'):
        device = next(model.parameters()).device
    else:
        Warning("Could not determine model device. Setting device to cpu.")
        device = torch.device('cpu')

    # Clone dataset structure (preserves metadata columns)
    xs_gen = dataset.xs.clone().to(device)
    ys_gen = torch.zeros_like(dataset.ys, device=device)

    # Valid trial mask from original dataset (NaN = padded / inactive)
    valid_mask = ~torch.isnan(dataset.xs[:, :, 0, 0])  # (n_sessions, n_trials)

    # Initial action: default to action 0
    action_idx = torch.zeros(n_sessions, dtype=torch.long, device=device)
    state = None

    for t in tqdm(range(n_trials)):
        # Get reward from environment
        observation, _ = environment.step(action_idx.cpu())
        observation = observation.to(device)

        action_oh = torch.nn.functional.one_hot(action_idx, n_actions).float()
        xs_gen[:, t, 0, :n_actions] = action_oh
        xs_gen[:, t, 0, n_actions:-5] = observation
        
        # Zero out inactive sessions so they don't affect model state
        inactive = ~valid_mask[:, t]
        xs_gen[inactive, t] = 0

        # Forward pass (single trial)
        obs = xs_gen[:, t:t+1]  # (B, 1, 1, F)
        logits, state = model(obs, state)
        
        # Normalize logits to (B, A)
        if logits.dim() == 5:  # BaseModel: (E, B, T, W, A)
            logits = logits.mean(dim=0)
        if logits.dim() == 4:  # (B, T, W, A)
            logits_t = logits[:, -1, 0, :n_actions]
        else:  # (B, T, A)
            logits_t = logits[:, -1, :n_actions]

        # Sample next action
        probs = torch.softmax(logits_t, dim=-1)
        action_idx = torch.multinomial(probs, 1).squeeze(-1)

        # Store as target (next-action prediction)
        ys_gen[:, t, 0, :] = torch.nn.functional.one_hot(action_idx, n_actions).float()

    # Restore NaN padding for inactive trials
    for t in range(n_trials):
        inactive = ~valid_mask[:, t]
        xs_gen[inactive, t] = float('nan')
        ys_gen[inactive, t] = float('nan')

    xs_gen = xs_gen.cpu()
    ys_gen = ys_gen.cpu()

    dataset_gen = SpiceDataset(xs_gen, ys_gen, n_reward_features=n_rewards)

    if save_dataset is not None:
        dataset_to_csv(dataset_gen, save_dataset, **(kwargs_dataset or {}))

    print("Done generating behavior.")
    return dataset_gen
