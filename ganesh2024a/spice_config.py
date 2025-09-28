import torch

from spice.resources.rnn import BaseRNN
from spice.estimator import SpiceConfig


# Simple contrast config
CONTRAST_CONFIG = SpiceConfig(
    rnn_modules=['x_value_reward_chosen', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen'],
    
    control_parameters=['c_action', 'c_reward', 'c_contr_diff'],
    
    library_setup={
        'x_value_reward_chosen': ['c_contr_diff', 'c_reward'],
        'x_value_reward_not_chosen': ['c_contr_diff', 'c_reward'],
        'x_value_choice_chosen': ['c_contr_diff'],
        'x_value_choice_not_chosen': ['c_contr_diff'],
    },
    
    filter_setup={
        'x_value_reward_chosen': ['c_action', 1, True],  # standard chosen action filter
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
    }
)


class RNN_ContrDiff(BaseRNN):
    
    init_values = {
            'x_value_reward': 0.5,
            'x_value_choice': 0.,
        }
    
    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        embedding_size = 32,
        dropout = 0.5,
        leaky_relu = 0.01,
        device = torch.device('cpu'),
        **kwargs,
    ):
        
        super().__init__(n_actions=n_actions, device=device, n_participants=n_participants, embedding_size=embedding_size)
        
        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(num_embeddings=n_participants, embedding_size=embedding_size, leaky_relu=leaky_relu, dropout=dropout)
        
        # scaling factor (inverse noise temperature) for each participant for the values which are handled by an hard-coded equation
        self.betas['x_value_reward'] = self.setup_constant(embedding_size=embedding_size, leaky_relu=leaky_relu)
        self.betas['x_value_choice'] = self.setup_constant(embedding_size=embedding_size, leaky_relu=leaky_relu)
        
        # set up the submodules
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=2+self.embedding_size)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=2+self.embedding_size)
        self.submodules_rnn['x_value_choice_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        self.submodules_rnn['x_value_choice_not_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        input_variables, embedding_variables, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, contr_diffs = input_variables
        contr_diffs = contr_diffs.repeat(1, 1, 2)
        participant_id, _ = embedding_variables
        
        # derive more observations
        rewards_chosen = (actions * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)
        
        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_id[:, 0].int())
        
        for timestep, action, reward_chosen, contr_diff in zip(timesteps, actions, rewards_chosen, contr_diffs):
            
            # record signals 
            self.record_signal('c_action', action)
            self.record_signal('c_reward', reward_chosen)
            self.record_signal('c_contr_diff', torch.abs(contr_diff))
            self.record_signal('c_value_reward', self.state['x_value_reward'])
            self.record_signal('c_value_choice', self.state['x_value_choice'])
            self.record_signal('x_value_reward_chosen', self.state['x_value_reward'])
            self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
            self.record_signal('x_value_choice_chosen', self.state['x_value_choice'])
            self.record_signal('x_value_choice_not_chosen', self.state['x_value_choice'])
            
            # reward expectancy updates
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(
                    torch.abs(contr_diff),
                    reward_chosen,
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            next_value_reward_not_chosen = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=(
                    torch.abs(contr_diff),
                    reward_chosen,
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            # choice preference updates
            next_value_choice_chosen = self.call_module(
                key_module='x_value_choice_chosen',
                key_state='x_value_choice',
                action=action,
                inputs=(
                    torch.abs(contr_diff),
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            next_value_choice_not_chosen = self.call_module(
                key_module='x_value_choice_not_chosen',
                key_state='x_value_choice',
                action=1-action,
                inputs=(
                    torch.abs(contr_diff),
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            # updating the memory state
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            self.state['x_value_choice'] = next_value_choice_chosen + next_value_choice_not_chosen
            
            # final [batch, 2] tensor (left, right)
            logits[timestep] = self.state['x_value_reward']*self.betas['x_value_reward'](participant_embedding) + self.state['x_value_choice']*self.betas['x_value_choice'](participant_embedding)
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()
    

class RNN_ContrDiff_LoHi(BaseRNN):
    
    init_values = {
            'x_value_reward': 0.5,
            'x_value_choice': 0.,
        }
    
    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        embedding_size = 32,
        dropout = 0.5,
        leaky_relu = 0.01,
        device = torch.device('cpu'),
        **kwargs,
    ):
        
        super().__init__(n_actions=n_actions, device=device, n_participants=n_participants, embedding_size=embedding_size)
        
        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(num_embeddings=n_participants, embedding_size=embedding_size, leaky_relu=leaky_relu, dropout=dropout)
        
        # scaling factor (inverse noise temperature) for each participant for the values which are handled by an hard-coded equation
        self.betas['x_value_reward'] = self.setup_constant(embedding_size=embedding_size, leaky_relu=leaky_relu)
        self.betas['x_value_choice'] = self.setup_constant(embedding_size=embedding_size, leaky_relu=leaky_relu)
        
        # set up the submodules
        self.submodules_rnn['x_value_reward_low'] = self.setup_module(input_size=2+self.embedding_size)
        self.submodules_rnn['x_value_reward_high'] = self.setup_module(input_size=2+self.embedding_size)
        self.submodules_rnn['x_value_choice_low'] = self.setup_module(input_size=1+self.embedding_size)
        self.submodules_rnn['x_value_choice_high'] = self.setup_module(input_size=1+self.embedding_size)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        input_variables, embedding_variables, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, contr_diffs = input_variables
        contr_diffs = contr_diffs.repeat(1, 1, 2)
        participant_id, _ = embedding_variables
        
        # derive more observations
        rewards_chosen = (actions * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)
        
        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_id[:, 0].int())
        
        for timestep, action, reward_chosen, contr_diff in zip(timesteps, actions, rewards_chosen, contr_diffs): #, rewards_not_chosen
            
            # determine whether low or high contrast was chosen
            contr_low_chosen, contr_high_chosen = torch.zeros_like(action), torch.zeros_like(action)
            contr_low_chosen[..., 0] = torch.logical_or(torch.logical_and(contr_diff[:, :1] < 0, (action[:, 0] == 1).reshape(-1, 1)), torch.logical_and(contr_diff[:, :1] > 0, (action[:, 1] == 1).reshape(-1, 1))).float().reshape(-1)  # low contrast chosen
            contr_high_chosen[..., 1] = torch.logical_or(torch.logical_and(contr_diff[:, :1] > 0, (action[:, 0] == 1).reshape(-1, 1)), torch.logical_and(contr_diff[:, :1] < 0, (action[:, 1] == 1).reshape(-1, 1))).float().reshape(-1) # high contrast chosen
            
            # reward expectancy updates
            next_value_reward_contrast_low = self.call_module(
                key_module='x_value_reward_low',
                key_state='x_value_reward',
                action=contr_low_chosen,
                inputs=(
                    torch.abs(contr_diff),
                    reward_chosen,
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            next_value_reward_contrast_high = self.call_module(
                key_module='x_value_reward_high',
                key_state='x_value_reward',
                action=contr_high_chosen,
                inputs=(
                    torch.abs(contr_diff),
                    reward_chosen,
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            # choice preference updates
            next_value_choice_contrast_low = self.call_module(
                key_module='x_value_choice_low',
                key_state='x_value_choice',
                action=contr_low_chosen,
                inputs=(
                    torch.abs(contr_diff),
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            next_value_choice_contrast_high = self.call_module(
                key_module='x_value_choice_high',
                key_state='x_value_choice',
                action=contr_high_chosen,
                inputs=(
                    torch.abs(contr_diff),
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            # updating the memory state
            self.state['x_value_reward'] = next_value_reward_contrast_low + next_value_reward_contrast_high
            self.state['x_value_choice'] = next_value_choice_contrast_low + next_value_choice_contrast_high
            
            # logits for low and high contrast (shape [batch])
            logits_low  = (next_value_reward_contrast_low * self.betas['x_value_reward'](participant_embedding) + next_value_choice_contrast_low * self.betas['x_value_choice'](participant_embedding))[:, 0]
            logits_high = (next_value_reward_contrast_high * self.betas['x_value_reward'](participant_embedding) + next_value_choice_contrast_high * self.betas['x_value_choice'](participant_embedding))[:, 1]

            # now map to left/right according to contr_diff
            # If contr_diff < 0: left=low, right=high
            # If contr_diff > 0: left=high, right=low
            left_logits  = torch.where(contr_diff[:, 0] < 0, logits_low, logits_high)
            right_logits = torch.where(contr_diff[:, 0] < 0, logits_high, logits_low)

            # final [batch, 2] tensor (left, right)
            logits[timestep, :] = torch.stack([left_logits, right_logits], dim=-1)
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()


class RNN_ContrDiff_LoHi_2(BaseRNN):
    
    init_values = {
            'x_value_reward': 0.5,
        }
    
    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        embedding_size = 32,
        dropout = 0.5,
        leaky_relu = 0.01,
        device = torch.device('cpu'),
        **kwargs,
    ):
        
        super().__init__(n_actions=n_actions, device=device, n_participants=n_participants, embedding_size=embedding_size)
        
        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(num_embeddings=n_participants, embedding_size=embedding_size, leaky_relu=leaky_relu, dropout=dropout)
        
        # scaling factor (inverse noise temperature) for each participant for the values which are handled by an hard-coded equation
        self.betas['x_value_reward'] = self.setup_constant(embedding_size=embedding_size, leaky_relu=leaky_relu)
        
        # set up the submodules
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=2+self.embedding_size)
        # self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=2+self.embedding_size)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        input_variables, embedding_variables, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, contr_diffs = input_variables
        contr_diffs = contr_diffs.repeat(1, 1, 2)
        participant_id, _ = embedding_variables
        
        # derive more observations
        rewards_chosen = (actions * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)
        
        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_id[:, 0].int())
        
        for timestep, action, reward_chosen, contr_diff in zip(timesteps, actions, rewards_chosen, contr_diffs): #, rewards_not_chosen
            
            # determine whether low or high contrast was chosen
            contr_chosen = torch.zeros_like(action)
            contr_chosen[..., 0] = torch.logical_or(torch.logical_and(contr_diff[:, 0] < 0, action[:, 0] == 1), torch.logical_and(contr_diff[:, 0] > 0, action[:, 1] == 1)).float()
            contr_chosen[..., 1] = torch.logical_or(torch.logical_and(contr_diff[:, 0] > 0, action[:, 0] == 1), torch.logical_and(contr_diff[:, 0] < 0, action[:, 1] == 1)).float()
            
            # reward expectancy updates
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=contr_chosen,
                inputs=(
                    torch.abs(contr_diff),
                    reward_chosen,
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            next_value_reward_not_chosen = (1-contr_chosen) * self.state['x_value_reward']
            
            # next_value_reward_not_chosen = self.call_module(
            #     key_module='x_value_reward_not_chosen',
            #     key_state='x_value_reward',
            #     action=1-contr_chosen,
            #     inputs=(
            #         torch.abs(contr_diff),
            #         reward_chosen,
            #         ),
            #     participant_embedding=participant_embedding,
            #     participant_index=participant_id,
            #     activation_rnn=torch.nn.functional.sigmoid,
            # )

            # updating the memory state
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            
            # logits for low and high contrast (shape [batch])
            logits_contrast = self.state['x_value_reward'] * self.betas['x_value_reward'](participant_embedding)
            
            # assume logits_contrast has shape [batch, 2] = (low, high)
            logits_low  = logits_contrast[:, 0]
            logits_high = logits_contrast[:, 1]

            # If contr_diff < 0: left=low, right=high
            # If contr_diff > 0: left=high, right=low
            left_logits  = torch.where(contr_diff[:, 0] < 0, logits_low, logits_high)
            right_logits = torch.where(contr_diff[:, 0] < 0, logits_high, logits_low)

            # final [batch, 2] tensor (left, right)
            logits[timestep, :] = torch.stack([left_logits, right_logits], dim=-1)
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()
