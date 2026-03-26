import numpy as np
import torch

from spice.resources.spice_utils import SpiceDataset
from spice.utils.convert_dataset import dataset_to_csv

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from weinhardt2025.benchmarking.benchmarking_qlearning import QLearning


# --- Configuration ---
list_n_participants = [32, 64, 128, 256, 512]
n_trials_per_session = 100
n_blocks_per_session = 4
n_iterations_per_n_sessions = 8
n_actions = 2
sigma = [0.2]

base_name = 'weinhardt2025/data/synthetic/synthetic_*.csv'

zero_threshold = 0.2
parameter_variance = 0.2
parameters_mean = {
    'beta_reward': 3.0,
    'beta_choice': 1.0,
    'alpha_reward': 0.5,
    'alpha_penalty': 0.5,
    'forget_rate': 0.2,
    'alpha_choice': 0.5,
}


# --- Utility functions ---

def compute_beta_dist_params(mean, var):
    """Compute Beta distribution (a, b) from desired mean and variance."""
    n = mean * (1 - mean) / var ** 2
    a = mean * n
    b = (1 - mean) * n
    return a, b


def sample_parameters_for_n(n_participants: int) -> dict:
    """Sample Q-learning parameters for n_participants with valid ranges.

    Sampling follows the same scheme as the original generator:
    - beta_reward, beta_choice: Beta(0.5, var) scaled by 2*mean, zero-thresholded,
      with at least one non-zero per participant
    - forget_rate, alpha_choice: Beta(mean, var), zero-thresholded
    - alpha_reward, alpha_penalty: Beta(mean, var), symmetrized when close

    Returns:
        dict of (n_participants,) arrays
    """
    parameters = {}

    # Scale parameters: ensure at least one of beta_reward, beta_choice is non-zero
    parameters['beta_reward'] = np.zeros(n_participants)
    parameters['beta_choice'] = np.zeros(n_participants)

    while np.any((parameters['beta_reward'] == 0) & (parameters['beta_choice'] == 0)):
        mask = (parameters['beta_reward'] == 0) & (parameters['beta_choice'] == 0)
        n_resample = mask.sum()
        parameters['beta_reward'][mask] = np.random.beta(
            *compute_beta_dist_params(0.5, parameter_variance), n_resample)
        parameters['beta_choice'][mask] = np.random.beta(
            *compute_beta_dist_params(0.5, parameter_variance), n_resample)
        parameters['beta_reward'][mask] *= (
            2 * parameters_mean['beta_reward']
            * (parameters['beta_reward'][mask] > zero_threshold))
        parameters['beta_choice'][mask] *= (
            2 * parameters_mean['beta_choice']
            * (parameters['beta_choice'][mask] > zero_threshold))

    # Auxiliary parameters with zero-threshold
    parameters['forget_rate'] = np.random.beta(
        *compute_beta_dist_params(parameters_mean['forget_rate'], parameter_variance),
        n_participants)
    parameters['forget_rate'] *= (parameters['forget_rate'] > zero_threshold)

    parameters['alpha_choice'] = np.random.beta(
        *compute_beta_dist_params(parameters_mean['alpha_choice'], parameter_variance),
        n_participants)
    parameters['alpha_choice'] *= (parameters['alpha_choice'] > zero_threshold)

    # Learning rates: no zero-threshold, symmetrize when close
    parameters['alpha_reward'] = np.random.beta(
        *compute_beta_dist_params(parameters_mean['alpha_reward'], parameter_variance),
        n_participants)
    parameters['alpha_penalty'] = np.random.beta(
        *compute_beta_dist_params(parameters_mean['alpha_penalty'], parameter_variance),
        n_participants)
    idx_sym = np.abs(parameters['alpha_reward'] - parameters['alpha_penalty']) < zero_threshold
    mean_alpha = (parameters['alpha_reward'][idx_sym] + parameters['alpha_penalty'][idx_sym]) / 2
    parameters['alpha_reward'][idx_sym] = mean_alpha
    parameters['alpha_penalty'][idx_sym] = mean_alpha

    return parameters


class VectorizedBanditsDrift:
    """Vectorized drifting two-armed bandit for all sessions in parallel."""

    def __init__(self, n_sessions: int, drift_sigma: float, n_actions: int = 2):
        self.n_sessions = n_sessions
        self.drift_sigma = drift_sigma
        self.n_actions = n_actions
        self.reward_probs = None

    def new_sess(self):
        """Sample fresh reward probabilities for all sessions."""
        self.reward_probs = np.random.rand(self.n_sessions, self.n_actions)

    def step(self, choices: np.ndarray) -> np.ndarray:
        """Vectorized step: sample rewards, apply drift.

        Args:
            choices: (n_sessions,) int array of chosen actions

        Returns:
            rewards: (n_sessions, n_actions) with NaN for unchosen arms
        """
        # Sample binary rewards based on current probabilities
        all_rewards = (np.random.rand(self.n_sessions, self.n_actions)
                       < self.reward_probs).astype(float)

        # Partial feedback: only reveal chosen arm
        rewards = np.full((self.n_sessions, self.n_actions), np.nan)
        idx = np.arange(self.n_sessions)
        rewards[idx, choices] = all_rewards[idx, choices]

        # Apply Gaussian drift and clip
        drift = np.random.normal(0, self.drift_sigma, (self.n_sessions, self.n_actions))
        self.reward_probs = np.clip(self.reward_probs + drift, 0, 1)

        return rewards


def build_single_step_input(
    choice_onehot: np.ndarray,
    rewards: np.ndarray,
    block: int,
    experiment_id: int,
    participant_ids: np.ndarray,
) -> torch.Tensor:
    """Build a single-timestep input tensor for the QLearning model.

    Args:
        choice_onehot: (B, n_actions) one-hot choices
        rewards: (B, n_actions) rewards with NaN for unchosen
        block: block index
        experiment_id: experiment index
        participant_ids: (B,) participant IDs

    Returns:
        xs: (B, T=1, W=1, F) tensor ready for model.forward(batch_first=True)
    """
    B = choice_onehot.shape[0]
    metadata = np.stack([
        np.zeros(B),                                  # time_trial
        np.zeros(B),                                  # trials (overwritten by init_forward_pass)
        np.full(B, block, dtype=float),               # block
        np.full(B, experiment_id, dtype=float),        # experiment_id
        participant_ids.astype(float),                 # participant_id
    ], axis=-1)  # (B, 5)

    xs = np.concatenate([choice_onehot, rewards, metadata], axis=-1)  # (B, 9)
    xs = torch.tensor(xs, dtype=torch.float32).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 9)
    return xs


def build_output_xs(
    choices_onehot: np.ndarray,
    rewards: np.ndarray,
    params: dict,
    block: int,
    experiment_id: int,
    participant_ids: np.ndarray,
) -> torch.Tensor:
    """Build output xs tensor with parameter columns for saving.

    Args:
        choices_onehot: (B, T, n_actions) one-hot choices
        rewards: (B, T, n_actions) rewards
        params: dict of (total_participants,) parameter arrays
        block: block index
        experiment_id: experiment index
        participant_ids: (B,) participant IDs (into the params arrays)

    Returns:
        xs: (B, T, W=1, F) tensor with format:
            [actions, rewards, params, mean_params, time_trial, trials, block, experiment_id, participant_id]
    """
    B, T, A = choices_onehot.shape

    # Build parameter columns (constant across trials)
    param_cols = []
    mean_param_cols = []
    for key in params:
        param_vals = params[key][participant_ids]  # (B,)
        param_cols.append(np.broadcast_to(param_vals[:, None], (B, T)))
        mean_param_cols.append(np.broadcast_to(
            np.full((B, T), parameters_mean[key]), (B, T)))

    params_block = np.stack(param_cols, axis=-1)          # (B, T, n_params)
    means_block = np.stack(mean_param_cols, axis=-1)      # (B, T, n_params)

    # Metadata
    metadata = np.zeros((B, T, 5), dtype=np.float32)
    metadata[:, :, 0] = 0                                  # time_trial
    metadata[:, :, 1] = 0                                  # trials
    metadata[:, :, 2] = block                              # block
    metadata[:, :, 3] = experiment_id                      # experiment_id
    metadata[:, :, 4] = participant_ids[:, None]           # participant_id

    xs = np.concatenate([
        choices_onehot,
        rewards,
        params_block,
        means_block,
        metadata,
    ], axis=-1)  # (B, T, 2*A + 2*n_params + 5)

    return torch.tensor(xs, dtype=torch.float32).unsqueeze(2)  # (B, T, W=1, F)


# --- Main generation ---

def generate_all():
    """Generate synthetic datasets for all participant counts, iterations, and experiments.

    Pools all participants across dataset sizes and iterations into a single batch,
    runs the QLearning model vectorized, then splits and saves per (n_participants, iteration).
    """

    # Total participant pool
    total_participants = sum(list_n_participants) * n_iterations_per_n_sessions

    # Build index mapping: (start_idx, end_idx, n_p, iteration)
    participant_groups = []
    idx = 0
    for iteration in range(n_iterations_per_n_sessions):
        for n_p in list_n_participants:
            participant_groups.append((idx, idx + n_p, n_p, iteration))
            idx += n_p

    # Participant IDs for the full pool (0..total-1)
    pool_participant_ids = np.arange(total_participants)

    for experiment_id in range(len(sigma)):

        # Sample parameters for the entire pool
        params = sample_parameters_for_n(total_participants)

        # Create the QLearning model with all participants
        model = QLearning(
            n_actions=n_actions,
            n_participants=total_participants,
            **{k: torch.tensor(v.reshape(-1, 1), dtype=torch.float32)
               for k, v in params.items()},
        )
        model.eval()

        # Vectorized environment
        env = VectorizedBanditsDrift(total_participants, sigma[experiment_id], n_actions)

        # Storage: list of (xs, ys) per block, each (total_participants, T, ...)
        all_xs_blocks = []
        all_ys_blocks = []

        for block_idx in range(n_blocks_per_session):
            # Reset environment and model state
            env.new_sess()
            model.init_state(batch_size=total_participants)

            # Storage for this block
            T = n_trials_per_session
            choices_block = np.zeros((T + 1, total_participants), dtype=int)
            rewards_block = np.full((T + 1, total_participants, n_actions), np.nan)

            # Initial logits = zeros (from initial state value_reward=0, value_choice=0)
            logits = torch.zeros(total_participants, n_actions)

            for trial in range(T + 1):
                # Sample choices from softmax over logits
                probs = torch.softmax(logits, dim=-1).numpy()
                cumprobs = np.cumsum(probs, axis=-1)
                u = np.random.rand(total_participants, 1)
                choices = (u >= cumprobs).sum(axis=-1).astype(int)
                choices = np.clip(choices, 0, n_actions - 1)

                # Get rewards from environment
                rewards = env.step(choices)

                # Store
                choices_block[trial] = choices
                rewards_block[trial] = rewards

                # Build single-timestep input and run model forward
                choice_onehot = np.eye(n_actions, dtype=np.float32)[choices]  # (B, A)
                xs_step = build_single_step_input(
                    choice_onehot, rewards, block_idx, experiment_id, pool_participant_ids)

                with torch.no_grad():
                    model_logits, _ = model(
                        xs_step, model.get_state(detach=True), batch_first=True)

                # model_logits shape: (E=1, B, T=1, W=1, A)
                logits = model_logits[0, :, 0, 0, :]  # (B, A)

            # Build output xs and ys from stored data
            # xs[t] = (choice[t], reward[t], params, means, metadata) for t=0..T-1
            # ys[t] = choice[t+1] for t=0..T-1
            choices_onehot_all = np.eye(n_actions, dtype=np.float32)[choices_block]  # (T+1, B, A)

            xs_choices = choices_onehot_all[:-1].transpose(1, 0, 2)   # (B, T, A)
            xs_rewards = rewards_block[:-1].transpose(1, 0, 2)        # (B, T, A)
            ys_block = choices_onehot_all[1:].transpose(1, 0, 2)      # (B, T, A)

            xs_block = build_output_xs(
                xs_choices, xs_rewards, params,
                block_idx, experiment_id, pool_participant_ids)

            ys_block = torch.tensor(ys_block, dtype=torch.float32).unsqueeze(2)  # (B, T, W=1, A)

            all_xs_blocks.append(xs_block)
            all_ys_blocks.append(ys_block)

        # Concatenate blocks along session dim: (B * n_blocks, T, W, F)
        all_xs = torch.cat(all_xs_blocks, dim=0)
        all_ys = torch.cat(all_ys_blocks, dim=0)

        # Split by (n_participants, iteration) and save
        parameter_cols = list(params.keys())
        mean_parameter_cols = ['mean_' + key for key in params]

        for start, end, n_p, iteration in participant_groups:
            dataset_name = base_name.replace('*', f'{n_p}p_{iteration}_{experiment_id}')

            # Gather all blocks for this participant group
            block_slices = []
            ys_slices = []
            for block_idx in range(n_blocks_per_session):
                offset = block_idx * total_participants
                block_slices.append(all_xs[offset + start : offset + end])
                ys_slices.append(all_ys[offset + start : offset + end])

            xs_group = torch.cat(block_slices, dim=0)   # (n_p * n_blocks, T, W, F)
            ys_group = torch.cat(ys_slices, dim=0)       # (n_p * n_blocks, T, W, A)

            # Remap participant_ids to 0..n_p-1 within this group
            xs_group[..., -1] = xs_group[..., -1] - start

            dataset = SpiceDataset(xs=xs_group, ys=ys_group)
            dataset_to_csv(
                dataset=dataset, path=dataset_name,
                additional_inputs=parameter_cols + mean_parameter_cols)
            print(f'Data saved to {dataset_name}')


if __name__ == '__main__':
    generate_all()