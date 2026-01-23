import sys, os
import argparse
import torch
from tqdm import tqdm
import pickle
from typing import List
from joblib import Parallel, delayed
from adabelief_pytorch import AdaBelief

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spice.resources.spice_utils import SpiceDataset
from spice.utils.convert_dataset import convert_dataset, split_data_along_sessiondim, reshape_data_along_participantdim
from spice.utils.agent import Agent
from spice.resources.spice_training import batch_train


class Castro2025Model(torch.nn.Module):
    """
    Cognitive model from Castro et al. 2025 (discovered program) for multi-armed bandit task.

    Model features:
    - Loss aversion learning (gamma parameter)
    - Exploration rate with decay
    - Perseveration and switch bonuses
    - Attention biases
    - Cumulative choice tracking
    - Softmax with temperature and lapse rate

    Original JAX code:
    def agent(params, choice, reward, agent_state = None):
        num_params = 13
        params = jnp.clip(params, -5, 5)
        beta_r = jnp.clip(jax.nn.softplus(params[0]), 0.01, 20)
        lapse = jnp.clip(jax.nn.sigmoid(params[1]), 0.01, 0.99)
        prior = jnp.clip(jax.nn.softplus(params[2]), 0.01, 0.99)
        alpha_exploration_rate = jnp.clip(jax.nn.sigmoid(params[3]), 0.01, 0.99)
        decay_rate = jnp.clip(jax.nn.sigmoid(params[4]), 0.01, 0.99)
        attention_bias1 = params[5]
        attention_bias2 = params[6]
        perseveration_strength = jax.nn.softplus(params[7])
        switch_strength = params[8]
        lambda_param = jnp.clip(jax.nn.softplus(params[9]), 0.0, 1.0)
        gamma = jax.nn.softplus(params[10]) # Loss aversion parameter
        temperature = jnp.clip(jax.nn.softplus(params[11]) + 1e-6, 1e-6, 100) #Softmax temperature
        beta_p = jax.nn.softplus(params[12])
        if agent_state is None:
            q_values = jnp.ones((4,)) * prior
            old_choice = -1
            trial_since_last_switch = 0
            exploration_rate = alpha_exploration_rate
            cumchoice = jnp.zeros((4,))
        else:
            q_values = agent_state[:4]
            old_choice = agent_state[4]
            trial_since_last_switch = agent_state[5]
            exploration_rate = agent_state[6]
            cumchoice = agent_state[7:11]
        if choice is not None and reward is not None:
            delta = reward - gamma*(1-reward) - q_values[choice]
            q_values = q_values.at[choice].set(q_values[choice] + delta)
            trial_since_last_switch = jnp.where(choice == old_choice, trial_since_last_switch + 1, 0)
            exploration_rate = exploration_rate * (1 - 1e-3) # decay exploration rate slowly
            cumchoice = cumchoice.at[choice].set(cumchoice[choice] + 1)
            q_values = (1 - exploration_rate) * q_values + exploration_rate * jnp.mean(q_values)
            q_values = q_values * decay_rate
        choice_probs = (1 - lapse) * jax.nn.softmax(beta_r * q_values / temperature + beta_p * jnp.log(
            1 + cumchoice)) + lapse / 4
        choice_logits = jnp.log(choice_probs)
        if choice is not None:
            perseveration_bonus = (choice == old_choice) * perseveration_strength * jax.nn.one_hot(
                choice, num_classes=4)
            switch_bonus = (choice != old_choice) * switch_strength * jax.nn.one_hot(choice, num_classes=4)
            attention_bonus1 = attention_bias1 * jax.nn.one_hot(old_choice, num_classes=4)
            attention_bonus2 = attention_bias2 * jax.nn.one_hot((choice + 2) % 4, num_classes=4)
            choice_logits = (
                choice_logits + perseveration_bonus + switch_bonus + attention_bonus1 + attention_bonus2 +
                jax.nn.one_hot(choice, 4) * jnp.log(trial_since_last_switch + 1))
        agent_state = jnp.concatenate(
            [q_values, jnp.array([choice, trial_since_last_switch, exploration_rate]),
            cumchoice])
        return choice_logits, agent_state
    """

    init_values = {
        'x_value_reward': None,  # Will be set based on prior parameter
        'x_old_choice': -1,
        'x_trial_since_last_switch': 0,
        'x_exploration_rate': None,  # Will be set based on alpha_exploration_rate
        'x_cumchoice': 0.0,
    }

    def __init__(self, n_actions: int = 4):
        super().__init__()

        self.n_actions = n_actions
        self._n_actions = n_actions  # For compatibility

        # Initialize 13 parameters (raw, will be transformed)
        self.beta_r_raw = torch.nn.Parameter(torch.zeros(1))
        self.lapse_raw = torch.nn.Parameter(torch.zeros(1))
        self.prior_raw = torch.nn.Parameter(torch.zeros(1))
        self.alpha_exploration_rate_raw = torch.nn.Parameter(torch.zeros(1))
        self.decay_rate_raw = torch.nn.Parameter(torch.zeros(1))
        self.attention_bias1 = torch.nn.Parameter(torch.zeros(1))
        self.attention_bias2 = torch.nn.Parameter(torch.zeros(1))
        self.perseveration_strength_raw = torch.nn.Parameter(torch.zeros(1))
        self.switch_strength = torch.nn.Parameter(torch.zeros(1))
        self.lambda_param_raw = torch.nn.Parameter(torch.zeros(1))
        self.gamma_raw = torch.nn.Parameter(torch.zeros(1))
        self.temperature_raw = torch.nn.Parameter(torch.zeros(1))
        self.beta_p_raw = torch.nn.Parameter(torch.zeros(1))

        self.device = torch.device('cpu')

    def get_transformed_params(self):
        """Get transformed parameters with appropriate constraints."""
        beta_r = torch.clamp(torch.nn.functional.softplus(self.beta_r_raw), 0.01, 20)
        lapse = torch.clamp(torch.sigmoid(self.lapse_raw), 0.01, 0.99)
        prior = torch.clamp(torch.nn.functional.softplus(self.prior_raw), 0.01, 0.99)
        alpha_exploration_rate = torch.clamp(torch.sigmoid(self.alpha_exploration_rate_raw), 0.01, 0.99)
        decay_rate = torch.clamp(torch.sigmoid(self.decay_rate_raw), 0.01, 0.99)
        perseveration_strength = torch.nn.functional.softplus(self.perseveration_strength_raw)
        lambda_param = torch.clamp(torch.nn.functional.softplus(self.lambda_param_raw), 0.0, 1.0)
        gamma = torch.nn.functional.softplus(self.gamma_raw)
        temperature = torch.clamp(torch.nn.functional.softplus(self.temperature_raw) + 1e-6, 1e-6, 100)
        beta_p = torch.nn.functional.softplus(self.beta_p_raw)

        return {
            'beta_r': beta_r,
            'lapse': lapse,
            'prior': prior,
            'alpha_exploration_rate': alpha_exploration_rate,
            'decay_rate': decay_rate,
            'attention_bias1': self.attention_bias1,
            'attention_bias2': self.attention_bias2,
            'perseveration_strength': perseveration_strength,
            'switch_strength': self.switch_strength,
            'lambda_param': lambda_param,
            'gamma': gamma,
            'temperature': temperature,
            'beta_p': beta_p,
        }

    def init_forward_pass(self, inputs, prev_state, batch_first):
        """Initialize forward pass."""
        if batch_first:
            inputs = inputs.permute(1, 0, 2)

        # Extract actions and rewards
        actions = inputs[:, :, :self.n_actions].float()  # (T, B, n_actions)
        rewards = inputs[:, :, self.n_actions:2*self.n_actions].float()  # (T, B, n_actions)

        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])

        timesteps = torch.arange(actions.shape[0])
        logits = torch.zeros_like(actions)

        return (actions, rewards), logits, timesteps

    def set_initial_state(self, batch_size=1):
        """Initialize the hidden state for each session."""
        params = self.get_transformed_params()

        state = {
            'x_value_reward': torch.ones(batch_size, self.n_actions, dtype=torch.float32) * params['prior'],
            'x_old_choice': torch.full((batch_size,), -1, dtype=torch.long),
            'x_trial_since_last_switch': torch.zeros(batch_size, dtype=torch.long),
            'x_exploration_rate': torch.full((batch_size,), params['alpha_exploration_rate'].item(), dtype=torch.float32),
            'x_cumchoice': torch.zeros(batch_size, self.n_actions, dtype=torch.float32),
        }
        self.set_state(state)
        return self.get_state()

    def set_state(self, state_dict):
        """Set the latent variables."""
        self.state = state_dict

    def get_state(self, detach=False):
        """Return the memory state."""
        state = self.state
        if detach:
            state = {key: state[key].detach() for key in state}
        return state

    def update_step(self, q_values, old_choice, trial_since_last_switch, exploration_rate, cumchoice, action, reward):
        """
        Perform one update step.

        Args:
            q_values: Current Q-values (batch_size, n_actions)
            old_choice: Previous choice (batch_size,)
            trial_since_last_switch: Trials since last switch (batch_size,)
            exploration_rate: Current exploration rate (batch_size,)
            cumchoice: Cumulative choices (batch_size, n_actions)
            action: Current action one-hot encoded (batch_size, n_actions)
            reward: Current reward (batch_size, n_actions)

        Returns:
            Updated state variables
        """
        params = self.get_transformed_params()

        # Convert one-hot action to integer choice
        current_choice = torch.argmax(action, dim=-1)  # (batch_size,)

        # Get scalar reward for the chosen action
        reward_value = torch.sum(reward * action, dim=-1)  # (batch_size,)

        # Update Q-values with loss aversion
        # delta = reward - gamma*(1-reward) - q_values[choice]
        gamma = params['gamma']
        delta = reward_value - gamma * (1 - reward_value) - q_values[torch.arange(q_values.shape[0]), current_choice]

        # Update Q-value for chosen action
        q_values_new = q_values.clone()
        q_values_new[torch.arange(q_values.shape[0]), current_choice] += delta

        # Update trial_since_last_switch
        trial_since_last_switch_new = torch.where(
            current_choice == old_choice,
            trial_since_last_switch + 1,
            torch.zeros_like(trial_since_last_switch)
        )

        # Decay exploration rate
        exploration_rate_new = exploration_rate * (1 - 1e-3)

        # Update cumchoice
        cumchoice_new = cumchoice.clone()
        cumchoice_new[torch.arange(cumchoice.shape[0]), current_choice] += 1

        # Apply exploration (regress to mean)
        q_mean = q_values_new.mean(dim=-1, keepdim=True)
        q_values_new = (1 - exploration_rate_new.unsqueeze(-1)) * q_values_new + exploration_rate_new.unsqueeze(-1) * q_mean

        # Apply decay
        q_values_new = q_values_new * params['decay_rate']

        return q_values_new, current_choice, trial_since_last_switch_new, exploration_rate_new, cumchoice_new

    def compute_choice_logits(self, q_values, old_choice, trial_since_last_switch, cumchoice, current_choice=None):
        """
        Compute choice logits.

        Args:
            q_values: Q-values (batch_size, n_actions)
            old_choice: Previous choice (batch_size,)
            trial_since_last_switch: Trials since last switch (batch_size,)
            cumchoice: Cumulative choices (batch_size, n_actions)
            current_choice: Current choice if available (batch_size,) - used for bonuses

        Returns:
            Choice logits (batch_size, n_actions)
        """
        params = self.get_transformed_params()
        batch_size = q_values.shape[0]

        # Compute base choice probabilities
        # choice_probs = (1 - lapse) * softmax(beta_r * q_values / temperature + beta_p * log(1 + cumchoice)) + lapse / 4
        softmax_input = params['beta_r'] * q_values / params['temperature'] + params['beta_p'] * torch.log(1 + cumchoice)
        choice_probs = (1 - params['lapse']) * torch.softmax(softmax_input, dim=-1) + params['lapse'] / self.n_actions
        logits = torch.log(choice_probs + 1e-10)  # Add small epsilon to avoid log(0)

        # Add bonuses if we have current choice
        if current_choice is not None:
            # Perseveration bonus
            perseveration_mask = (current_choice == old_choice).float().unsqueeze(-1)
            perseveration_bonus = perseveration_mask * params['perseveration_strength'] * torch.nn.functional.one_hot(current_choice, num_classes=self.n_actions).float()

            # Switch bonus
            switch_mask = (current_choice != old_choice).float().unsqueeze(-1)
            switch_bonus = switch_mask * params['switch_strength'] * torch.nn.functional.one_hot(current_choice, num_classes=self.n_actions).float()

            # Attention bias 1 (based on old choice)
            attention_bonus1 = torch.zeros(batch_size, self.n_actions)
            valid_old = old_choice >= 0
            if valid_old.any():
                attention_bonus1[valid_old] = params['attention_bias1'] * torch.nn.functional.one_hot(old_choice[valid_old], num_classes=self.n_actions).float()

            # Attention bias 2 (based on (choice + 2) % 4)
            attention_idx = (current_choice + 2) % 4
            attention_bonus2 = params['attention_bias2'] * torch.nn.functional.one_hot(attention_idx, num_classes=self.n_actions).float()

            # Trial-based bonus
            trial_bonus = torch.nn.functional.one_hot(current_choice, num_classes=self.n_actions).float() * torch.log(trial_since_last_switch.float().unsqueeze(-1) + 1)

            logits = logits + perseveration_bonus + switch_bonus + attention_bonus1 + attention_bonus2 + trial_bonus

        return logits

    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass through the model."""
        input_variables, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards = input_variables

        # Get initial state
        q_values = self.state['x_value_reward']
        old_choice = self.state['x_old_choice']
        trial_since_last_switch = self.state['x_trial_since_last_switch']
        exploration_rate = self.state['x_exploration_rate']
        cumchoice = self.state['x_cumchoice']

        # Process each timestep
        for t, (action, reward) in enumerate(zip(actions, rewards)):
            # Get current choice
            current_choice = torch.argmax(action, dim=-1)

            # Compute logits for this timestep (prediction for current action)
            logits[t] = self.compute_choice_logits(q_values, old_choice, trial_since_last_switch, cumchoice, current_choice)

            # Update state based on observed action and reward
            q_values, current_choice, trial_since_last_switch, exploration_rate, cumchoice = self.update_step(
                q_values, old_choice, trial_since_last_switch, exploration_rate, cumchoice, action, reward
            )
            old_choice = current_choice

        # Update state
        self.state['x_value_reward'] = q_values
        self.state['x_old_choice'] = old_choice
        self.state['x_trial_since_last_switch'] = trial_since_last_switch
        self.state['x_exploration_rate'] = exploration_rate
        self.state['x_cumchoice'] = cumchoice

        if batch_first:
            logits = logits.swapaxes(0, 1)

        return logits, self.get_state()


def train_single_participant(index_participant, xs, ys, xs_test, ys_test, n_actions, epochs, lr, convergence_threshold, n_total_participants, max_restarts=10, convergence_check_interval=100):
    """
    Train a single participant's model with multiple restarts.

    Following Castro et al. 2025:
    - Use AdaBelief optimizer with lr=5e-2
    - Check convergence every 100 steps by comparing Omega_k to Omega_{k-100}
    - Convergence criterion: |(Omega_k - Omega_{k-100})/Omega_{k-100}| < 1e-2
    - Repeat from different initial parameters up to 10 times
    - Stop when 3 programs converge to current best

    Args:
        index_participant: Index of participant
        xs: Training data
        ys: Training labels
        xs_test: Test data
        ys_test: Test labels
        n_actions: Number of actions
        epochs: Maximum number of epochs (10000 in paper)
        lr: Learning rate (5e-2 in paper)
        convergence_threshold: Convergence threshold (1e-2 in paper)
        n_total_participants: Total number of participants
        max_restarts: Maximum number of restarts (10 in paper)
        convergence_check_interval: Check convergence every N steps (100 in paper)
    """

    best_model = None
    best_loss = float('inf')
    converged_models = []

    for restart in range(max_restarts):
        # Initialize new model with random parameters
        model_participant = Castro2025Model(n_actions=n_actions)
        optimizer = AdaBelief(model_participant.parameters(), lr=lr, print_change_log=False)

        loss_history = []
        converged = False

        # Create inner progress bar for epochs
        epoch_pbar = tqdm(range(epochs), desc=f"P{index_participant+1}/{n_total_participants} R{restart+1}/{max_restarts}", leave=False, position=index_participant % 10)

        for e in epoch_pbar:
            # Train model
            model_participant, optimizer, current_loss = batch_train(
                model=model_participant,
                xs=xs,
                ys=ys,
                optimizer=optimizer,
            )

            loss_history.append(current_loss)

            # Check convergence every 100 steps (Castro et al. 2025 strategy)
            convergence_value = float('inf')
            if (e + 1) % convergence_check_interval == 0 and len(loss_history) >= convergence_check_interval:
                omega_k = loss_history[-1]
                omega_k_minus_100 = loss_history[-convergence_check_interval]

                # Relative change: |(Omega_k - Omega_{k-100})/Omega_{k-100}|
                if abs(omega_k_minus_100) > 1e-10:
                    convergence_value = abs((omega_k - omega_k_minus_100) / omega_k_minus_100)

                    if convergence_value < convergence_threshold:
                        converged = True

            # Test model
            if xs_test is not None:
                with torch.no_grad():
                    model_participant.eval()
                    _, _, loss_test = batch_train(
                        model=model_participant,
                        xs=xs_test,
                        ys=ys_test,
                        optimizer=optimizer,
                    )
                    model_participant.train()

                    epoch_pbar.set_postfix({
                        'train': f'{current_loss:.5f}',
                        'test': f'{loss_test:.5f}',
                        'conv': f'{convergence_value:.2e}',
                        'best': f'{best_loss:.5f}'
                    })
            else:
                epoch_pbar.set_postfix({
                    'loss': f'{current_loss:.5f}',
                    'conv': f'{convergence_value:.2e}',
                    'best': f'{best_loss:.5f}'
                })

            # Early stopping if converged
            if converged:
                break

        epoch_pbar.close()

        # Evaluate final loss
        final_loss = loss_history[-1] if loss_history else float('inf')

        # Update best model if this one is better
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model_participant

            # If converged, add to converged models list
            if converged:
                converged_models.append(final_loss)

                # Check if we have 3 converged models close to the best
                # "converged to current best" means within 1% of best loss
                converged_to_best = sum(1 for loss in converged_models if abs(loss - best_loss) / max(abs(best_loss), 1e-10) < 0.01)

                if converged_to_best >= 3:
                    print(f"Participant {index_participant+1}: 3 models converged to best loss {best_loss:.5f} after {restart+1} restarts")
                    break

        # Print restart summary
        status = "converged" if converged else "max epochs"
        print(f"Participant {index_participant+1} restart {restart+1}: loss={final_loss:.5f} ({status}), best={best_loss:.5f}")

    return best_model


def training(n_actions: int, dataset_training: SpiceDataset, epochs: int, lr: float = 0.05, dataset_test: SpiceDataset = None, convergence_threshold: float = 1e-2, n_jobs: int = 1, max_restarts: int = 10, convergence_check_interval: int = 100):
    """
    Training loop for Castro2025 model using MLE (gradient descent).

    Following Castro et al. 2025 defaults:
    - lr: 5e-2
    - convergence_threshold: 1e-2
    - max_restarts: 10
    - convergence_check_interval: 100
    """

    n_participants = len(dataset_training)

    if n_jobs == 1:
        # Sequential training
        all_models = []
        for index_participant in range(n_participants):
            xs = dataset_training.xs[index_participant]
            ys = dataset_training.ys[index_participant]
            xs_test = dataset_test.xs[index_participant] if dataset_test is not None else None
            ys_test = dataset_test.ys[index_participant] if dataset_test is not None else None

            model = train_single_participant(
                index_participant, xs, ys, xs_test, ys_test,
                n_actions, epochs, lr, convergence_threshold, n_participants,
                max_restarts, convergence_check_interval
            )
            all_models.append(model)
    else:
        # Parallel training
        print(f"Training {n_participants} participants in parallel with {n_jobs} jobs...")

        all_models = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(train_single_participant)(
                index_participant,
                dataset_training.xs[index_participant],
                dataset_training.ys[index_participant],
                dataset_test.xs[index_participant] if dataset_test is not None else None,
                dataset_test.ys[index_participant] if dataset_test is not None else None,
                n_actions, epochs, lr, convergence_threshold, n_participants,
                max_restarts, convergence_check_interval
            )
            for index_participant in range(n_participants)
        )

    return all_models


class AgentCastro2025(Agent):
    """A class that allows running a pretrained Castro2025 model as an agent."""

    def __init__(
        self,
        model: Castro2025Model,
        deterministic: bool = True,
    ):
        """Initialize the agent network."""
        super().__init__(model_rnn=None, n_actions=model.n_actions, deterministic=deterministic)

        assert isinstance(model, Castro2025Model), "The passed model is not an instance of Castro2025Model."

        self.model = model
        self.model.eval()

    @property
    def q(self):
        """Return the action values."""
        return self.state['x_value_reward'].squeeze(0).detach().cpu().numpy()


def setup_agent_benchmark(path_model: str, deterministic: bool = True, **kwargs) -> List[AgentCastro2025]:
    """Setup Castro2025 agents from saved model."""

    # Load models
    with open(path_model, 'rb') as file:
        all_models = pickle.load(file)

    agents = []
    for model in all_models:
        agents.append(AgentCastro2025(model=model, deterministic=deterministic))

    # Count parameters (13 parameters)
    n_parameters = 13

    return agents, n_parameters


def main(path_save_model: str, path_data: str, n_actions: int, n_epochs: int, lr: float, split_ratio=None, convergence_threshold: float = 1e-2, n_jobs: int = 1, max_restarts: int = 10, convergence_check_interval: int = 100):
    """
    Main training function.

    Following Castro et al. 2025 defaults:
    - n_epochs: 10000
    - lr: 5e-2
    - convergence_threshold: 1e-2
    - max_restarts: 10
    - convergence_check_interval: 100
    """

    # Load and split data
    dataset_full = convert_dataset(
        file=path_data,
        df_participant_id='s_id',
        df_choice='action',
        )

    if split_ratio is not None:
        dataset_training, dataset_test = split_data_along_sessiondim(
            dataset_full,
            list_test_sessions=split_ratio
        )
    else:
        dataset_training = dataset_full
        dataset_test = None

    dataset_training = reshape_data_along_participantdim(dataset_training)
    if dataset_test is not None:
        dataset_test = reshape_data_along_participantdim(dataset_test)

    print('Training Castro2025 Model...')
    all_models = training(
        dataset_training=dataset_training,
        dataset_test=dataset_test,
        n_actions=n_actions,
        epochs=n_epochs,
        lr=lr,
        convergence_threshold=convergence_threshold,
        n_jobs=n_jobs,
        max_restarts=max_restarts,
        convergence_check_interval=convergence_check_interval,
    )

    # Save models
    with open(path_save_model, 'wb') as file:
        pickle.dump(all_models, file)

    print(f'Model saved to {path_save_model}')
    print('Training Castro2025 Model done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Castro2025 model with PyTorch (MLE via gradient descent) using Castro et al. 2025 early stopping strategy')

    parser.add_argument('--path_save_model', type=str, default='params/castro2025/castro2025.pkl', help='Path to save the trained model')
    parser.add_argument('--path_data', type=str, default='data/eckstein2024/eckstein2024.csv', help='Path to the dataset')
    parser.add_argument('--n_actions', type=int, default=4, help='Number of actions')
    parser.add_argument('--n_epochs', type=int, default=10000, help='Maximum number of training epochs (10000 in Castro et al. 2025)')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate (5e-2 in Castro et al. 2025)')
    parser.add_argument('--split_ratio', type=str, default=None, help='Sessions to use for testing (comma-separated)')
    parser.add_argument('--convergence_threshold', type=float, default=1e-2, help='Convergence threshold for early stopping (1e-2 in Castro et al. 2025)')
    parser.add_argument('--convergence_check_interval', type=int, default=100, help='Check convergence every N steps (100 in Castro et al. 2025)')
    parser.add_argument('--max_restarts', type=int, default=10, help='Maximum number of random restarts (10 in Castro et al. 2025)')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs for training participants (-1 for all CPUs)')

    args = parser.parse_args()

    # Parse split ratio
    if args.split_ratio is not None:
        split_ratio = [int(x) for x in args.split_ratio.split(',')]
    else:
        split_ratio = None

    main(
        path_save_model=args.path_save_model,
        path_data=args.path_data,
        n_actions=args.n_actions,
        n_epochs=args.n_epochs,
        lr=args.lr,
        split_ratio=split_ratio,
        convergence_threshold=args.convergence_threshold,
        n_jobs=args.n_jobs,
        max_restarts=args.max_restarts,
        convergence_check_interval=args.convergence_check_interval,
    )
