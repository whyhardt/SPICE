import sys, os
import argparse
import torch
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import pickle
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spice.resources.spice_utils import SpiceDataset
from spice.utils.convert_dataset import convert_dataset, split_data_along_sessiondim, reshape_data_along_participantdim
from spice.resources.bandits import Agent, AgentNetwork
from spice.resources.spice_training import batch_train


class Dezfouli2019GQL(torch.nn.Module):
    
    init_values = {
        'x_value_reward': 0.5,  # Initialize Q-values to 0.5
        'x_value_choice': 0.0,  # Initialize choice histories to 0
        'x_learning_rate_reward': 0.0,  # dummy place-holder
    }
    
    def __init__(self, model: str, n_actions: int = 2, dimensions: int = 2):
        super().__init__()
        
        self.d = dimensions
        self.n_actions = n_actions
        self._n_actions = n_actions  # For compatibility
        
        # Initialize parameters based on model configuration
        if 'Phi' in model:
            self.phi_logit = torch.nn.Parameter(torch.zeros((self.d, 1)))
            self._phi_learnable = True
        else:
            self.register_buffer('phi_fixed', torch.ones((self.d, 1)))
            self._phi_learnable = False
            
        if 'Chi' in model:
            self.chi_logit = torch.nn.Parameter(torch.zeros((self.d, 1)))
            self._chi_learnable = True
        else:
            self.register_buffer('chi_fixed', torch.ones((self.d, 1)))
            self._chi_learnable = False
            
        if 'Beta' in model:
            self.beta = torch.nn.Parameter(torch.ones((self.d, 1)))
        else:
            self.register_buffer('beta', torch.ones((self.d, 1)))
            
        if 'Kappa' in model:
            self.kappa = torch.nn.Parameter(torch.zeros((self.d, 1)))
        else:
            self.register_buffer('kappa', torch.zeros((self.d, 1)))
        
        if 'C' in model:
            self.C = torch.nn.Parameter(torch.zeros(self.d, self.d))
        else:
            self.register_buffer('C', torch.zeros(self.d, self.d))
            
        self.device = torch.device('cpu')
    
    def get_phi(self):
        """Get phi parameter values."""
        if self._phi_learnable:
            return torch.sigmoid(self.phi_logit)
        else:
            return self.phi_fixed
    
    def get_chi(self):
        """Get chi parameter values."""
        if self._chi_learnable:
            return torch.sigmoid(self.chi_logit)
        else:
            return self.chi_fixed
    
    def init_forward_pass(self, inputs, prev_state, batch_first):
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
        
        actions = inputs[:, :, :self.n_actions].float()
        rewards = inputs[:, :, self.n_actions:2*self.n_actions].float()
        additional_inputs = inputs[:, :, 2*self.n_actions:-3].float()
        blocks = inputs[:, :, -3:-2].int().repeat(1, 1, 2)
        
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        timesteps = torch.arange(actions.shape[0])
        logits = torch.zeros_like(actions)
        
        return (actions, rewards, blocks, additional_inputs), logits, timesteps
    
    def set_initial_state(self, batch_size=1):
        """Initialize the hidden state for each session."""
        state = {}
        for key in self.init_values:
            # Initialize Q-values: (batch_size, n_actions, d)
            state[key] = torch.full(
                size=[batch_size, self.n_actions, self.d], 
                fill_value=self.init_values[key], 
                dtype=torch.float32, 
            )
            
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
    
    def gql_update_step(self, q_values, h_values, choice, reward):
        """
        Perform GQL update step.
        
        Args:
            q_values: Current Q-values (batch_size, n_actions, d)
            h_values: Current choice histories (batch_size, n_actions, d)
            choice: Choice made (batch_size, n_actions) - one-hot encoded
            reward: Reward received (batch_size, n_actions)
        
        Returns:
            Updated q_values, h_values
        """
        batch_size = q_values.shape[0]
        
        # Get parameters
        phi = self.get_phi().T  # (1, d)
        chi = self.get_chi().T  # (1, d)
        
        # Expand choice and reward for broadcasting
        choice_expanded = choice.unsqueeze(-1)  # (batch_size, n_actions, 1)
        reward_expanded = reward.unsqueeze(-1).expand_as(q_values)  # (batch_size, n_actions, d)
        
        # Q-value updates: Q_t = (1 - phi) * Q_{t-1} + phi * reward (for chosen action)
        phi_expanded = phi.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 1, d)
        q_update = phi_expanded * reward_expanded * choice_expanded
        q_values_new = (1 - phi_expanded) * q_values + q_update
        
        # History updates: H_t = (1 - chi) * H_{t-1} + chi (for chosen action)
        chi_expanded = chi.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 1, d)
        h_chosen_update = chi_expanded * choice_expanded
        h_values_new = (1 - chi_expanded) * h_values + h_chosen_update
        
        return q_values_new, h_values_new
    
    def compute_action_values(self, q_values, h_values):
        """
        Compute combined action values.
        
        Args:
            q_values: Q-values (batch_size, n_actions, d)
            h_values: Choice histories (batch_size, n_actions, d)
            
        Returns:
            Combined action values (batch_size, n_actions)
        """
        batch_size = q_values.shape[0]
        
        # Weighted Q-values and histories
        beta_expanded = self.beta.T.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 1, d)
        kappa_expanded = self.kappa.T.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 1, d)
        
        q_weighted = torch.sum(beta_expanded * q_values, dim=-1)  # (batch_size, n_actions)
        h_weighted = torch.sum(kappa_expanded * h_values, dim=-1)  # (batch_size, n_actions)
        
        # Interaction terms: H^T * C * Q for each action
        interaction = torch.zeros(batch_size, self.n_actions)
        for a in range(self.n_actions):
            # h_a^T @ C @ q_a for each batch element
            h_a = h_values[:, a, :]  # (batch_size, d)
            q_a = q_values[:, a, :]  # (batch_size, d)
            
            # Use einsum for cleaner batch matrix multiplication
            # h_a: (batch_size, d), C: (d, d), q_a: (batch_size, d)
            # Result: (batch_size,)
            interaction[:, a] = torch.einsum('bi,ij,bj->b', h_a, self.C, q_a)
        
        return q_weighted + h_weighted + interaction
    
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass through the model."""
        input_variables, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)            
        actions, rewards, _, _ = input_variables
        
        # Get initial state
        q_values = self.state['x_value_reward']  # (batch_size, n_actions, d)
        h_values = self.state['x_value_choice']  # (batch_size, n_actions, d)
        
        # Process each timestep
        for t, (action, reward) in enumerate(zip(actions, rewards)):
            # Update Q-values and histories
            q_values, h_values = self.gql_update_step(q_values, h_values, action, reward)
            
            # Compute action values for next timestep prediction
            if t < len(actions) - 1:  # Don't predict after last timestep
                combined_values = self.compute_action_values(q_values, h_values)
                logits[t] = combined_values
        
        # Update state
        self.state['x_value_reward'] = q_values
        self.state['x_value_choice'] = h_values
        
        if batch_first:
            logits = logits.swapaxes(0, 1)
        
        return logits, self.get_state()  # Return all but last timestep


def training(model_config: str, n_actions: int, dimensions: int,  dataset_training: SpiceDataset, epochs: int, lr: float = 0.01, dataset_test: SpiceDataset = None):
    """Training loop for GQL model."""
    all_models = []
    
    for index_participant in range(len(dataset_training)):
        
        print(f"Training participant {index_participant+1}...")
        
        model_participant = Dezfouli2019GQL(model=model_config, n_actions=n_actions, dimensions=dimensions)
        
        optimizer = torch.optim.Adam(model_participant.parameters(), lr=lr)
        
        xs = dataset_training.xs[index_participant]
        ys = dataset_training.ys[index_participant]
        
        participant_losses = []
        current_loss = 0
        
        # Create inner progress bar for epochs
        epoch_pbar = tqdm(range(epochs), desc=f"Participant {index_participant}", leave=False)
        
        for e in epoch_pbar:
            
            # Train model
            model_participant, optimizer, current_loss = batch_train(
                model=model_participant,
                xs=xs,
                ys=ys,
                optimizer=optimizer,
            )
            
            participant_losses.append(current_loss)
            
            # Update progress bar with current loss
            # epoch_pbar.set_postfix({
            #     'loss': f'{current_loss:.5f}',
            #     'epoch': f'{e+1}/{epochs}'
            # })
            
            # Test model
            if dataset_test is not None:
                xs_test = dataset_test.xs[index_participant]
                ys_test = dataset_test.ys[index_participant]
                    
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
                        'epoch': f'{e+1}/{epochs}'
                    })
        
        epoch_pbar.close()
        
        all_models.append(model_participant)
        
        # Print final summary for this participant
        if participant_losses:
            final_loss = participant_losses[-1]
            print(f"Participant {index_participant}: Final loss = {final_loss:.5f} (improved from {participant_losses[0]:.5f})")
    
    return all_models


class AgentGQL(AgentNetwork):
    """A class that allows running a pretrained GQL model as an agent."""

    def __init__(
        self,
        model: Dezfouli2019GQL,
        deterministic: bool = True,
    ):
        """Initialize the agent network."""
        super().__init__(model_rnn=None, n_actions=model.n_actions, deterministic=deterministic)
        
        assert isinstance(model, Dezfouli2019GQL), "The passed model is not an instance of Dezfouli2019GQL."
        
        self.model = model
        self.model.eval()

    # def new_sess(self, *args, **kwargs):
    #     """Reset the network for the beginning of a new session."""
    #     self.model.set_initial_state(batch_size=1)
        
    #     # Extract state as numpy arrays for compatibility with Agent interface
    #     state = self.model.get_state()
    #     self.state = {
    #         'x_value_reward': state['x_value_reward'][0].detach().cpu().numpy(),  # (n_actions, d)
    #         'x_value_choice': state['x_value_choice'][0].detach().cpu().numpy(),  # (n_actions, d)
    #     }

    def get_betas(self):
        """Return beta and kappa values."""
        with torch.no_grad():
            beta = self.model.beta.squeeze(0)#.detach().cpu().numpy()  # (d,)
            kappa = self.model.kappa.squeeze(0)#.detach().cpu().numpy()  # (d,)
        return beta, kappa
    
    @property
    def q(self):
        """Return the action values."""
        beta, kappa = self.get_betas()
        
        # Compute weighted values
        q_weighted = torch.sum(beta * self.state['x_value_reward'].squeeze(0), dim=-1)  # (n_actions,)
        h_weighted = torch.sum(kappa * self.state['x_value_choice'].squeeze(0), dim=-1)  # (n_actions,)
        
        # Compute interaction terms
        C = self.model.C#.detach().cpu().numpy()
        interaction = torch.zeros(self.n_actions)
        for a in range(self.n_actions):
            interaction[a] = self.state['x_value_choice'].squeeze(0)[a] @ C @ self.state['x_value_reward'].squeeze(0)[a]
        
        return (q_weighted + h_weighted + interaction).detach().cpu().numpy()
    
    @property
    def q_reward(self):
        beta, _ = self.get_betas()
        q_weighted = torch.sum(beta * self.state['x_value_reward'].squeeze(0), dim=-1)  # (n_actions,)
        return (q_weighted).detach().cpu().numpy()
    
    @property
    def q_choice(self):
        _, kappa = self.get_betas()
        q_weighted = torch.sum(kappa * self.state['x_value_choice'].squeeze(0), dim=-1)  # (n_actions,)
        return (q_weighted).detach().cpu().numpy()
    
    @property
    def learning_rate_reward(self):
        return self.state['x_learning_rate_reward'][0, 0]


def setup_agent_gql(path_model: str, model_config: str = "PhiChiBetaKappaC", deterministic: bool = True, **kwargs) -> AgentGQL:
    """Setup GQL agent from saved model."""
    
    # Load state dict
    with open(path_model, 'rb') as file:
        all_models = pickle.load(file)

    agent = []
    for model in all_models:
        agent.append(AgentGQL(model=model, deterministic=deterministic))
    
    n_parameters = 0
    for index_letter, letter in enumerate(model_config):
        if not letter.islower():
            n_parameters += 1 * model.d * model.d if letter == 'C' and index_letter == len(model_config)-1 else 1 * model.d
    
    return agent, n_parameters


def main(path_save_model: str, path_data: str, model_config: str, n_actions: int, dimensions: int, n_epochs: int, lr: float, split_ratio=[3, 6, 9]):
    """Main training function."""
    
    path_save_model = path_save_model.replace('.', '_'+model_config+'.')
    
    # Load and split data
    dataset_training, dataset_test = split_data_along_sessiondim(
        convert_dataset(path_data), 
        list_test_sessions=split_ratio
    )
    dataset_training = reshape_data_along_participantdim(dataset_training)
    dataset_test = reshape_data_along_participantdim(dataset_test)
    
    print('Training GQL Model...')
    all_models = training(
        dataset_training=dataset_training, 
        dataset_test=None,#dataset_test, 
        model_config=model_config,
        n_actions=n_actions,
        dimensions=dimensions, 
        epochs=n_epochs,
        lr=lr,
    )
    
    # Save model
    with open(path_save_model, 'wb') as file:
        pickle.dump(all_models, file)  
    # torch.save(model.state_dict(), path_save_model)
    print(f'Model saved to {path_save_model}')
    print('Training GQL Model done!')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train GQL model with PyTorch')
    
    parser.add_argument('--path_save_model', type=str, default='params/dezfouli2019/gql_dezfouli2019.pkl', help='Path to save the trained model')
    parser.add_argument('--path_data', type=str, default='data/dezfouli2019/dezfouli2019.csv', help='Path to the dataset')
    parser.add_argument('--model', type=str, default='PhiChiBetaKappaC', help='Model configuration (e.g., PhiChiBeta)')
    parser.add_argument('--n_actions', type=int, default=2, help='Number of actions')
    parser.add_argument('--dimensions', type=int, default=2, help='Number of dimensions (d parameter)')
    parser.add_argument('--n_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--split_ratio', type=str, default=None, help='Sessions to use for testing (comma-separated)')
    
    args = parser.parse_args()
    
    # Parse split ratio
    if args.split_ratio is not None:
        split_ratio = [int(x) for x in args.split_ratio.split(',')]
    else:
        split_ratio = None
        
    main(
        path_save_model=args.path_save_model,
        path_data=args.path_data,
        model_config=args.model,
        n_actions=args.n_actions,
        dimensions=args.dimensions,
        n_epochs=args.n_epochs,
        lr=args.lr,
        split_ratio=split_ratio,
    )