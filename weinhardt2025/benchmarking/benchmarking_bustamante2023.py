import sys, os
import argparse
import torch
from tqdm import tqdm
import pickle
from typing import List, Optional
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spice.resources.spice_utils import SpiceDataset
from spice.utils.convert_dataset import convert_dataset, split_data_along_sessiondim, reshape_data_along_participantdim
from spice.resources.bandits import AgentNetwork
from spice.resources.spice_training import batch_train


class MarginalValueTheoremModel(torch.nn.Module):
    """
    Marginal Value Theorem (MVT) model for foraging task (Bustamante et al. 2023).

    The MVT predicts that a forager should leave a patch when the instantaneous
    gain rate in the current patch drops to the average gain rate in the environment.

    Key components:
    - Within-patch gain rate tracking (with depletion)
    - Environmental average gain rate estimation
    - Travel time and harvest time from data
    - Individual differences in parameters

    Model parameters (per individual):
    - alpha_env: Learning rate for environmental gain rate
    - beta: Inverse temperature for decision softmax
    - c: Intercept/bias term for decision rule
    - depletion (optional): Rate of within-patch depletion
    - baseline_gain (optional): Initial expected gain in a patch
    """

    init_values = {
        'cumulative_reward': 0.0,  # Cumulative reward in current patch
        'n_harvests': 0,  # Number of harvests in current patch
        'time_in_patch': 0.0,  # Time spent in current patch
        'env_gain_rate': 0.0,  # Estimated environmental gain rate
        'current_tree_state': 0.0,  # Current expected reward from tree (si)
    }

    def __init__(
        self,
        n_actions: int = 2,
        n_participants: int = 1,
        depletion: Optional[float] = None,
        baseline_gain: Optional[float] = None,
        batch_first: Optional[bool] = False,
    ):
        super().__init__()
        
        self.n_actions = n_actions
        self.n_participants = n_participants
        self.batch_first = batch_first

        # Initialize parameters (raw, will be transformed)
        # Learning rate for environmental gain rate (0-1)
        self.alpha_env_raw = torch.nn.Parameter(torch.zeros(n_participants))

        # Inverse temperature for softmax decision (positive)
        self.beta_raw = torch.nn.Parameter(torch.zeros(n_participants))

        # Intercept/bias term for decision rule (unconstrained)
        self.c_raw = torch.nn.Parameter(torch.zeros(n_participants))

        # Within-patch depletion rate (positive)
        if depletion is None:
            self.depletion_raw = torch.nn.Parameter(torch.zeros(n_participants))
            self.learn_depletion = True
        else:
            self.register_buffer('depletion_fixed', torch.full((n_participants), depletion))
            self.learn_depletion = False
            
        # Baseline gain rate expectation (positive)
        if baseline_gain is None:
            self.baseline_gain_raw = torch.nn.Parameter(torch.zeros(n_participants))
            self.learn_baseline_gain = True
        else:
            self.register_buffer('baseline_gain_fixed', torch.full((n_participants), baseline_gain))
            self.learn_baseline_gain = False

        self.device = torch.device('cpu')
        
    def forward(self, inputs, prev_state=None):
        """Forward pass through the model.

        Implements the MVT learning model from Constantino et al. (2015), Table 2:
        - P(harvest) = 1/{1 + exp[-c - β(kappa * si - ro * h)]}
        - delta_i = r_i/tau_i - ro_i
        - ro_{i+1} = ro_i + [1-(1-alpha)^tau_i] · delta_i
        """
        input_variables, participant_ids, logits, timesteps = self.init_forward_pass(inputs, prev_state, self.batch_first)
        actions, rewards, harvest_duration, travel_duration = input_variables

        # Process each timestep
        for t in timesteps:
            # Get current timestep data (shape: [batch_size, ...])
            action_t = actions[t]  # (B, n_actions)
            reward_t = rewards[t]  # (B, n_actions)
            harvest_duration_t = harvest_duration[t].squeeze(-1)  # (B,)
            travel_duration_t = travel_duration[t].squeeze(-1)  # (B,)

            # Compute expected next reward (si) using MVT formulation
            # si is the expected reward if we harvest now
            expected_next_reward = self.state['current_tree_state'] * self.depletion[participant_ids]

            # Compute stay value using MVT: c + β(si - ρh)
            # Stay is better when expected reward exceeds opportunity cost
            value_stay = expected_next_reward - self.state['env_gain_rate'] * harvest_duration_t

            # Compute stay logit for this timestep (for action 0 = harvest); exit value is always 0
            # Include intercept c (bias term) and inverse temperature β
            logits[t, :, 0] = self.c[participant_ids] + self.beta[participant_ids] * value_stay

            # Determine which action was taken
            # action_t[:, 0] = 1 means harvest, action_t[:, 1] = 1 means leave
            did_harvest = action_t[:, 0]  # (B,)
            did_leave = 1.0 - did_harvest

            # Get actual reward received
            actual_reward = reward_t.nansum(dim=-1)

            # Determine time spent on this action
            time_spent = harvest_duration_t * did_harvest + travel_duration_t * did_leave

            # Update environmental gain rate using MVT learning rule (Table 2)
            # delta_i = r_i/tau_i - ro_i
            reward_rate = actual_reward / (time_spent + 1e-8)  # Avoid division by zero
            prediction_error = reward_rate - self.state['env_gain_rate']

            # Effective learning rate: [1 - (1-alpha)^taui]
            # This accounts for the time duration of the action
            effective_lr = 1.0 - torch.pow(1.0 - self.alpha_env[participant_ids], time_spent)
            self.state['env_gain_rate'] = self.state['env_gain_rate'] + effective_lr * prediction_error

            # Update patch state based on action taken
            # If harvested: update cumulative stats for current patch
            # If left: reset patch state for new patch
            self.state['cumulative_reward'] = torch.where(
                did_harvest > 0.5,
                self.state['cumulative_reward'] + actual_reward,
                0
            )
            self.state['n_harvests'] = torch.where(
                did_harvest > 0.5,
                self.state['n_harvests'] + 1,
                0
            )
            self.state['time_in_patch'] = torch.where(
                did_harvest > 0.5,
                self.state['time_in_patch'] + time_spent,
                0
            )
            self.state['current_tree_state'] = torch.where(
                did_harvest > 0.5,
                actual_reward,  # Update state to observed reward
                self.baseline_gain[participant_ids].detach(),
            )

        if self.batch_first:
            logits = logits.swapaxes(0, 1)

        return logits, self.get_state()

    @property
    def alpha_env(self):
        return torch.clamp(torch.sigmoid(self.alpha_env_raw), 0.01, 0.99)

    @property
    def beta(self):
        return torch.clamp(torch.nn.functional.softplus(self.beta_raw), 0.1, 10.0)

    @property
    def c(self):
        """Intercept/bias term for decision rule (unconstrained, but clamped for stability)"""
        return torch.clamp(self.c_raw, -10.0, 10.0)

    @property
    def depletion(self):
        if self.learn_depletion:
            depletion = torch.clamp(torch.nn.functional.softplus(self.depletion_raw), 0.001, 1.0)
        else:
            depletion = self.depletion_fixed
        return depletion

    @property
    def baseline_gain(self):
        if self.learn_baseline_gain:
            baseline_gain = torch.clamp(torch.nn.functional.softplus(self.baseline_gain_raw), 0.1, 20.0)
        else:
            baseline_gain = self.baseline_gain_fixed
        return baseline_gain

    def init_forward_pass(self, inputs, prev_state, batch_first):
        """Initialize forward pass."""
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
        
        inputs = inputs.nan_to_num(0.)
        
        # Extract actions and rewards
        actions = inputs[:, :, :self.n_actions].float()  # (T, B, n_actions)
        rewards = inputs[:, :, self.n_actions:2*self.n_actions].float()  # (T, B, n_actions)

        # Extract harvest_duration and travel_duration from additional inputs
        # additional_inputs: [harvest_duration, travel_duration]
        harvest_duration = inputs[:, :, 2*self.n_actions:2*self.n_actions+1].float()  # (T, B, 1)
        travel_duration = inputs[:, :, 2*self.n_actions+1:2*self.n_actions+2].float()  # (T, B, 1)

        participant_ids = inputs[0, :, -1].long()
        
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
            self.state['env_gain_rate'] = self.baseline_gain[participant_ids]
            self.state['current_tree_state'] = self.baseline_gain[participant_ids]

        timesteps = torch.arange(actions.shape[0])
        logits = torch.zeros_like(actions)

        return (actions, rewards, harvest_duration, travel_duration), participant_ids, logits, timesteps

    def set_initial_state(self, batch_size=1):
        """Initialize the hidden state for each session."""

        state = {
            'cumulative_reward': torch.zeros(batch_size, dtype=torch.float32),
            'n_harvests': torch.zeros(batch_size, dtype=torch.float32),  # Changed to float for torch.where compatibility
            'time_in_patch': torch.zeros(batch_size, dtype=torch.float32),
            'env_gain_rate': torch.zeros(batch_size, dtype=torch.float32),
            'current_tree_state': torch.zeros(batch_size, dtype=torch.float32),
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
    
    
if __name__=='__main__':
    
    from spice.utils.convert_dataset import convert_dataset
    
    file = 'weinhardt2025/data/bustamante2023/bustamante2023_processed.csv'
    
    dataset = convert_dataset(
        file=file,
        df_participant_id='subject_id',
        df_choice='decision',
        df_reward='reward',
        df_block='overall_round',
        additional_inputs=['harvest_duration', 'travel_duration'],   
    )
    
    n_participants = len(dataset.xs[..., -1].unique())
    
    mvt = MarginalValueTheoremModel(n_participants=n_participants)
    
    # benchmark training
    epochs = 1000
    metric = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=mvt.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        
        random_index = torch.randint(len(dataset.xs), (len(dataset.xs), 1))[:, 0]
        
        mask = ~torch.isnan(dataset.xs[random_index, :, 0]).reshape(-1)
        
        logits, state = mvt(inputs=dataset.xs[random_index], batch_first=True)
        
        # compute loss
        loss = metric(
            logits.reshape(-1, mvt.n_actions)[mask],
            dataset.ys.argmax(dim=-1, keepdim=True).long().reshape(-1)[mask], 
            )
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} --- Loss: {loss.item():.5f}")
        
    print("Fitted parameters:")
    print("\nAlpha")
    print(mvt.alpha_env)
    print("\nBeta")
    print(mvt.beta)
    print("\nC")
    print(mvt.c)
    print("\nBaseline Gain")
    print(mvt.baseline_gain)
    print("\nDepletion")
    print(mvt.depletion)
    