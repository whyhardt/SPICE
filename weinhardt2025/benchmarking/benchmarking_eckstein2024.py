import sys, os
import argparse
import torch
from tqdm import tqdm
import pickle
from typing import List
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spice.resources.spice_utils import SpiceDataset
from spice.utils.convert_dataset import convert_dataset, split_data_along_sessiondim, reshape_data_along_participantdim
from spice.resources.bandits import AgentNetwork
from spice.resources.spice_training import batch_train


class Eckstein2024Model(torch.nn.Module):
    """
    Cognitive model from Eckstein et al. 2024 for multi-armed bandit task.

    Model features:
    - Separate learning rates for learning and forgetting
    - One-back perseveration bonus
    - Inverse temperature (beta_r)

    Original JAX code:
    def agent(
        params: chex.Array,
        choice: int,
        reward: int,
        agent_state: Optional[chex.Array],
    ) -> Tuple[chex.Array, chex.Array]:
        # Cognitive model describing human behavior on a multi-armed bandit task.
        # Assumes the agent is presented with four options on each trial.
        # Args:
        # params: a list containing [beta_r, alpha_learn, alpha_forget, and p]
        # choice: Choice made by the agent on the previous trial. 0, 1, 2, or 3
        # reward: Reward received by the agent on the previous trial. A scalar between
        # 0 and 100.
        # agent_state: [Q1, Q2, Q3, Q4, previous_choice] If None, assumes this is the
        # first trial of a new session and defaults to [0, 0, 0, 0, -1].
        # Returns:
        # choice_logits: The probabilities that the agent will choose option 0, 1, 2,
        # or 3 on the next trial, expressed as logits.
        # agent_state: New agent state
        if agent_state is None:
            agent_state = jnp.array(([0, 0, 0, 0, -1]))
        qs = agent_state[:4]
        prev_choice = jnp.int32(agent_state[4])
        beta_r = params[0]
        alpha_learn = 1 / (1 + jnp.exp(-params[1]))
        alpha_forget = 1 / (1 + jnp.exp(-params[2]))
        p = params[3]
        # One-back perseveration param should be 0 on the first trial of the session
        # We indicate this using prev_choice of -1
        p = p * (prev_choice != -1)
        # Update Q for chosen action
        choice = jnp.int32(choice)
        qs = qs.at[choice].set(alpha_learn * (reward - qs[choice]) + qs[choice])
        # Update Q for unchosen actions using a mask
        mask = jnp.ones_like(qs, dtype=bool)
        mask = mask.at[choice].set(False)
        qs = jnp.where(mask, qs * alpha_forget, qs)
        # Values for choice: Qs plus bonus for p
        qs_for_choice = qs.at[prev_choice].set(p + qs[prev_choice])
        agent_state = jnp.append(qs, choice)
        # Compute choice logits
        choice_logits = beta_r * qs_for_choice
        return choice_logits, agent_state
    """

    init_values = {
        'x_value_reward': 0.0,  # Initialize Q-values to 0
        'x_prev_choice': -1,     # Initialize previous choice to -1
    }

    def __init__(self, n_actions: int = 4):
        super().__init__()

        self.n_actions = n_actions
        self._n_actions = n_actions  # For compatibility

        # Initialize parameters
        # beta_r: inverse temperature (no constraints)
        self.beta_r = torch.nn.Parameter(torch.ones(1))

        # alpha_learn and alpha_forget: learning rates (will be passed through sigmoid)
        # Initialize to 0 so sigmoid gives 0.5
        self.alpha_learn_logit = torch.nn.Parameter(torch.zeros(1))
        self.alpha_forget_logit = torch.nn.Parameter(torch.zeros(1))

        # p: perseveration bonus (no constraints)
        self.p = torch.nn.Parameter(torch.zeros(1))

        self.device = torch.device('cpu')

    def get_alpha_learn(self):
        """Get alpha_learn parameter (sigmoid transform)."""
        return torch.sigmoid(self.alpha_learn_logit)

    def get_alpha_forget(self):
        """Get alpha_forget parameter (sigmoid transform)."""
        return torch.sigmoid(self.alpha_forget_logit)

    def init_forward_pass(self, inputs, prev_state, batch_first):
        """Initialize forward pass."""
        if batch_first:
            inputs = inputs.permute(1, 0, 2)

        # Extract actions and rewards
        # Assuming inputs has one-hot encoded actions followed by rewards
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
        state = {
            'x_value_reward': torch.full(
                size=[batch_size, self.n_actions],
                fill_value=self.init_values['x_value_reward'],
                dtype=torch.float32,
            ),
            'x_prev_choice': torch.full(
                size=[batch_size],
                fill_value=self.init_values['x_prev_choice'],
                dtype=torch.long,
            )
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

    def update_step(self, q_values, prev_choice, action, reward):
        """
        Perform one update step.

        Args:
            q_values: Current Q-values (batch_size, n_actions)
            prev_choice: Previous choice (batch_size,)
            action: Current action one-hot encoded (batch_size, n_actions)
            reward: Current reward (batch_size, n_actions)

        Returns:
            Updated q_values, current_choice
        """
        batch_size = q_values.shape[0]

        # Get parameters
        alpha_learn = self.get_alpha_learn()
        alpha_forget = self.get_alpha_forget()

        # Convert one-hot action to integer choice
        current_choice = torch.argmax(action, dim=-1)  # (batch_size,)

        # Get scalar reward for the chosen action
        reward_value = torch.sum(reward * action, dim=-1, keepdim=True)  # (batch_size, 1)

        # Update Q-values for chosen actions
        # Q[chosen] = Q[chosen] + alpha_learn * (reward - Q[chosen])
        #           = (1 - alpha_learn) * Q[chosen] + alpha_learn * reward
        chosen_mask = action  # (batch_size, n_actions)
        prediction_error = reward_value - q_values  # (batch_size, n_actions)
        q_update_chosen = alpha_learn * prediction_error * chosen_mask

        # Update Q-values for unchosen actions (decay/forget)
        # Q[unchosen] = Q[unchosen] * alpha_forget
        unchosen_mask = 1 - chosen_mask
        q_update_unchosen = (alpha_forget - 1) * q_values * unchosen_mask

        # Combined update
        q_values_new = q_values + q_update_chosen + q_update_unchosen

        return q_values_new, current_choice

    def compute_choice_logits(self, q_values, prev_choice):
        """
        Compute choice logits with perseveration bonus.

        Args:
            q_values: Q-values (batch_size, n_actions)
            prev_choice: Previous choice (batch_size,)

        Returns:
            Choice logits (batch_size, n_actions)
        """
        batch_size = q_values.shape[0]

        # Add perseveration bonus to previously chosen action
        # p should be 0 on first trial (when prev_choice == -1)
        perseveration = torch.zeros_like(q_values)

        # Only add perseveration if prev_choice is valid (>= 0)
        valid_prev = prev_choice >= 0

        for i in range(batch_size):
            if valid_prev[i]:
                perseveration[i, prev_choice[i]] = self.p

        # Compute logits
        q_with_perseveration = q_values + perseveration
        logits = self.beta_r * q_with_perseveration

        return logits

    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass through the model."""
        input_variables, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards = input_variables

        # Get initial state
        q_values = self.state['x_value_reward']  # (batch_size, n_actions)
        prev_choice = self.state['x_prev_choice']  # (batch_size,)

        # Process each timestep
        for t, (action, reward) in enumerate(zip(actions, rewards)):
            # Compute logits for this timestep (prediction for current action)
            logits[t] = self.compute_choice_logits(q_values, prev_choice)

            # Update Q-values based on observed action and reward
            q_values, current_choice = self.update_step(q_values, prev_choice, action, reward)
            prev_choice = current_choice

        # Update state
        self.state['x_value_reward'] = q_values
        self.state['x_prev_choice'] = prev_choice

        if batch_first:
            logits = logits.swapaxes(0, 1)

        return logits, self.get_state()


def train_single_participant(index_participant, xs, ys, xs_test, ys_test, n_actions, epochs, lr, convergence_threshold, n_total_participants):
    """Train a single participant's model."""

    model_participant = Eckstein2024Model(n_actions=n_actions)
    optimizer = torch.optim.Adam(model_participant.parameters(), lr=lr)

    participant_losses = []
    current_loss = 0
    prev_loss = float('inf')
    converged = False

    # Create inner progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc=f"Participant {index_participant+1}/{n_total_participants}", leave=False, position=index_participant % 10)

    for e in epoch_pbar:

        # Train model
        model_participant, optimizer, current_loss = batch_train(
            model=model_participant,
            xs=xs,
            ys=ys,
            optimizer=optimizer,
        )

        participant_losses.append(current_loss)

        # Check convergence
        convergence = abs(prev_loss - current_loss)
        if convergence < convergence_threshold:
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

                # Update progress bar with test loss too
                epoch_pbar.set_postfix({
                    'train_loss': f'{current_loss:.5f}',
                    'test_loss': f'{loss_test:.5f}',
                    'conv': f'{convergence:.2e}',
                    'epoch': f'{e+1}/{epochs}'
                })
        else:
            epoch_pbar.set_postfix({
                'loss': f'{current_loss:.5f}',
                'conv': f'{convergence:.2e}',
                'epoch': f'{e+1}/{epochs}'
            })

        prev_loss = current_loss

        # Early stopping if converged
        if converged:
            epoch_pbar.close()
            break

    if not converged:
        epoch_pbar.close()

    # Print final summary for this participant
    final_loss = participant_losses[-1] if participant_losses else 0
    convergence_msg = " (converged)" if converged else ""
    print(f"Participant {index_participant+1}: Final loss = {final_loss:.5f} (improved from {participant_losses[0]:.5f}){convergence_msg}")

    return model_participant


def training(n_actions: int, dataset_training: SpiceDataset, epochs: int, lr: float = 0.01, dataset_test: SpiceDataset = None, convergence_threshold: float = 1e-5, n_jobs: int = 1):
    """Training loop for Eckstein2024 model using MLE (gradient descent)."""

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
                n_actions, epochs, lr, convergence_threshold, n_participants
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
                n_actions, epochs, lr, convergence_threshold, n_participants
            )
            for index_participant in range(n_participants)
        )

    return all_models


class AgentEckstein2024(AgentNetwork):
    """A class that allows running a pretrained Eckstein2024 model as an agent."""

    def __init__(
        self,
        model: Eckstein2024Model,
        deterministic: bool = True,
    ):
        """Initialize the agent network."""
        super().__init__(model_rnn=None, n_actions=model.n_actions, deterministic=deterministic)

        assert isinstance(model, Eckstein2024Model), "The passed model is not an instance of Eckstein2024Model."

        self._model = model
        self._model.eval()

    @property
    def q(self):
        """Return the action values including perseveration bonus."""
        with torch.no_grad():
            q_values = self._state['x_value_reward'].squeeze(0)  # (n_actions,)
            prev_choice = self._state['x_prev_choice'].item()

            # Add perseveration bonus
            q_with_persev = q_values.clone()
            if prev_choice >= 0:
                q_with_persev[prev_choice] += self._model.p.item()

            return q_with_persev.detach().cpu().numpy()

    @property
    def q_reward(self):
        """Return the reward-based Q-values (without perseveration)."""
        return self._state['x_value_reward'].squeeze(0).detach().cpu().numpy()


def setup_agent_benchmark(path_model: str, deterministic: bool = True, **kwargs) -> List[AgentEckstein2024]:
    """Setup Eckstein2024 agents from saved model."""

    # Load models
    with open(path_model, 'rb') as file:
        all_models = pickle.load(file)

    agents = []
    for model in all_models:
        agents.append(AgentEckstein2024(model=model, deterministic=deterministic))

    # Count parameters (4 parameters: beta_r, alpha_learn, alpha_forget, p)
    n_parameters = 4

    return agents, n_parameters


def main(path_save_model: str, path_data: str, n_actions: int, n_epochs: int, lr: float, split_ratio=None, convergence_threshold: float = 1e-5, n_jobs: int = 1):
    """Main training function."""

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

    print('Training Eckstein2024 Model...')
    all_models = training(
        dataset_training=dataset_training,
        dataset_test=dataset_test,
        n_actions=n_actions,
        epochs=n_epochs,
        lr=lr,
        convergence_threshold=convergence_threshold,
        n_jobs=n_jobs,
    )

    # Save models
    with open(path_save_model, 'wb') as file:
        pickle.dump(all_models, file)

    print(f'Model saved to {path_save_model}')
    print('Training Eckstein2024 Model done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Eckstein2024 model with PyTorch (MLE via gradient descent)')

    parser.add_argument('--path_save_model', type=str, default='params/eckstein2024/eckstein2024.pkl', help='Path to save the trained model')
    parser.add_argument('--path_data', type=str, default='data/eckstein2024/eckstein2024.csv', help='Path to the dataset')
    parser.add_argument('--n_actions', type=int, default=4, help='Number of actions')
    parser.add_argument('--n_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--split_ratio', type=str, default=None, help='Sessions to use for testing (comma-separated)')
    parser.add_argument('--convergence_threshold', type=float, default=1e-5, help='Convergence threshold for early stopping')
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
    )
