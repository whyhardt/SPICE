#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from spice.resources.rnn import BaseRNN
from spice.estimator import SpiceConfig

WEINHARDT_2025_LIBRARY_CONFIG = SpiceConfig(
    rnn_modules=['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen'],
    control_parameters=['c_action', 'c_reward_chosen', 'c_value_reward', 'c_value_choice'],
    # The new module which handles the not-chosen value, does not need any additional inputs except for the value
    library_setup = {
        # 'x_value_reward_chosen': ['c_reward'] -> Remove this one from the library as we are not going to identify the dynamics of a hard-coded equation
        'x_learning_rate_reward': ['c_reward_chosen', 'c_value_reward', 'c_value_choice'],
        'x_value_reward_not_chosen': ['c_reward_chosen', 'c_value_choice'],
        'x_value_choice_chosen': ['c_value_reward'],
        'x_value_choice_not_chosen': ['c_value_reward'],
    },

    # Further, the new module should be applied only to the not-chosen values
    filter_setup = {
        'x_learning_rate_reward': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
    }
)


class Weinhardt2025LibraryRNN(BaseRNN):

    init_values = {
            'x_value_reward': 0.5,
            'x_value_choice': 0.,
            'x_learning_rate_reward': 0.,
        }

    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        embedding_size = 32,
        dropout = 0.5,
        leaky_relu = 0.01,
        degree = 2,
        device = torch.device('cpu'),
        **kwargs,
    ):

        super().__init__(n_actions=n_actions, device=device, n_participants=n_participants, embedding_size=embedding_size)

        self.degree = degree

        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(num_embeddings=n_participants, embedding_size=embedding_size)

        # scaling factor (inverse noise temperature) for each participant for the values which are handled by an hard-coded equation
        self.betas['x_value_reward'] = self.setup_constant(embedding_size=self.embedding_size, leaky_relu=leaky_relu)
        self.betas['x_value_choice'] = self.setup_constant(embedding_size=self.embedding_size, leaky_relu=leaky_relu)

        # set up the submodules using polynomial libraries instead of GRUs
        self.submodules_rnn['x_learning_rate_reward'] = self.setup_library(input_size=3+self.embedding_size, embedding_size=self.embedding_size, degree=degree)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_library(input_size=2+self.embedding_size, embedding_size=self.embedding_size, degree=degree)
        self.submodules_rnn['x_value_choice_chosen'] = self.setup_library(input_size=1+self.embedding_size, embedding_size=self.embedding_size, degree=degree)
        self.submodules_rnn['x_value_choice_not_chosen'] = self.setup_library(input_size=1+self.embedding_size, embedding_size=self.embedding_size, degree=degree)

        # set up hard-coded equations
        self.submodules_eq['x_value_reward_chosen'] = self.x_value_reward_chosen

    def x_value_reward_chosen(self, value, inputs):
        return value + inputs[..., 1] * (inputs[..., 0] - value)

    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """

        # First, we have to initialize all the inputs and outputs (i.e. logits)
        input_variables, ids, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, _ = input_variables
        participant_id, _ = ids

        # derive more observations
        rewards_chosen = (actions * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)
        # rewards_not_chosen = ((1-actions) * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)

        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_id[:, 0].int())

        for timestep, action, reward_chosen in zip(timesteps, actions, rewards_chosen): #, rewards_not_chosen

            # record the current memory state and control inputs to the modules for SINDy training
            if not self.training and len(self.submodules_sindy)==0:
                self.record_signal('c_action', action)
                self.record_signal('c_reward_chosen', reward_chosen)
                # self.record_signal('c_reward_not_chosen', reward_not_chosen)
                self.record_signal('c_value_reward', self.state['x_value_reward'])
                self.record_signal('c_value_choice', self.state['x_value_choice'])
                self.record_signal('x_learning_rate_reward', self.state['x_learning_rate_reward'])
                self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
                self.record_signal('x_value_choice_chosen', self.state['x_value_choice'])
                self.record_signal('x_value_choice_not_chosen', self.state['x_value_choice'])

            # updates for x_value_reward
            learning_rate_reward = self.call_module(
                key_module='x_learning_rate_reward',
                key_state='x_learning_rate_reward',
                action=action,
                inputs=(
                    reward_chosen,
                    # reward_not_chosen,
                    self.state['x_value_reward'],
                    self.state['x_value_choice'],
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(
                    reward_chosen,
                    learning_rate_reward,
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                )

            next_value_reward_not_chosen = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=(
                    reward_chosen,
                    # reward_not_chosen,
                    self.state['x_value_choice'],
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )

            # updates for x_value_choice
            next_value_choice_chosen = self.call_module(
                key_module='x_value_choice_chosen',
                key_state='x_value_choice',
                action=action,
                inputs=(self.state['x_value_reward']),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )

            next_value_choice_not_chosen = self.call_module(
                key_module='x_value_choice_not_chosen',
                key_state='x_value_choice',
                action=1-action,
                inputs=(self.state['x_value_reward']),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )

            # updating the memory state
            self.state['x_learning_rate_reward'] = learning_rate_reward
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            self.state['x_value_choice'] = next_value_choice_chosen + next_value_choice_not_chosen

            # Now keep track of the logit in the output array
            logits[timestep] = self.state['x_value_reward'] * self.betas['x_value_reward'](participant_embedding) + self.state['x_value_choice'] * self.betas['x_value_choice'](participant_embedding)

        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)

        return logits, self.get_state()


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    from torch.optim import Adam

    print("Creating library-based Weinhardt2025RNN and fitting to data...")

    # Load data
    print("Loading data...")
    df = pd.read_csv('/home/daniel/repositories/SPICE/weinhardt2025/data/eckstein2022.csv')

    # Process data
    sessions = df['session'].unique()
    n_participants = len(sessions)
    n_actions = 2

    print(f"Data loaded: {len(df)} trials, {n_participants} participants")

    # Create tensors for each session
    data_tensors = []
    for i, session in enumerate(sessions[:10]):  # Use first 10 sessions for testing
        session_data = df[df['session'] == session].copy()

        # Convert to format expected by RNN: (timesteps, batch, features)
        actions = torch.FloatTensor(session_data['choice'].values)
        rewards = torch.FloatTensor(session_data['reward'].values)

        # Create action one-hot encoding
        action_onehot = torch.zeros(len(actions), n_actions)
        action_onehot[torch.arange(len(actions)), actions.long()] = 1

        # Create reward vector
        reward_vec = torch.zeros(len(actions), n_actions)
        reward_vec[torch.arange(len(actions)), actions.long()] = rewards

        # Expected format: (timesteps, batch=1, features)
        # Features: [actions(2), rewards(2), additional_inputs(0), blocks(1), exp_id(1), participant_id(1)]

        timesteps = len(actions)

        # Create input tensor with correct shape
        input_tensor = torch.zeros(timesteps, 1, 7)  # 2+2+0+1+1+1 = 7 features

        # Fill in actions (one-hot)
        input_tensor[:, 0, :2] = action_onehot

        # Fill in rewards
        input_tensor[:, 0, 2:4] = reward_vec

        # Additional inputs (none) - skip indices 4:4

        # Blocks (dummy)
        input_tensor[:, 0, 4] = 0

        # Experiment ID (dummy)
        input_tensor[:, 0, 5] = 0

        # Participant ID
        input_tensor[:, 0, 6] = i

        data_tensors.append(input_tensor)

    # Create model
    print("Creating model...")
    device = torch.device('cpu')  # Force CPU usage for testing
    model = Weinhardt2025LibraryRNN(
        n_actions=n_actions,
        n_participants=10,  # Using first 10 sessions
        embedding_size=16,   # Smaller for testing
        degree=2,
        device=device
    ).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Using device: {device}")

    # Training loop - 1 epoch
    print("Starting training (1 epoch)...")
    model.train()
    total_loss = 0
    n_batches = 0

    for i, session_tensor in enumerate(data_tensors):
        session_tensor = session_tensor.to(device)

        # Add batch dimension: (4, n_actions, 1, timesteps)
        batch_input = session_tensor.unsqueeze(2)

        # Forward pass
        logits, _ = model(batch_input, batch_first=False)

        # Get target actions (ground truth choices)
        targets = torch.argmax(session_tensor[0], dim=0).unsqueeze(0).to(device)  # (1, timesteps)

        # Compute loss
        loss = F.cross_entropy(logits.permute(1, 2, 0), targets.long())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (i + 1) % 5 == 0:
            print(f"  Session {i+1}/10: Loss = {loss.item():.4f}")

    avg_loss = total_loss / n_batches
    print(f"Training completed! Average loss: {avg_loss:.4f}")

    # Test forward pass
    print("Testing model...")
    model.eval()
    with torch.no_grad():
        test_input = data_tensors[0].unsqueeze(2).to(device)
        test_logits, test_state = model(test_input, batch_first=False)
        print(f"Test output shape: {test_logits.shape}")
        print(f"Test output range: [{test_logits.min():.3f}, {test_logits.max():.3f}]")

    print("✓ Library-based Weinhardt2025RNN successfully created and trained!")