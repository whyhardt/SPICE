import torch

from spice.resources.rnn import BaseRNN
from spice.estimator import SpiceConfig


CONTR_DIFF_CONFIG = SpiceConfig(
    rnn_modules=['x_value_reward_chosen', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen', 'x_value_state_left', 'x_value_state_right'],  # 'x_learning_rate_reward
    control_parameters=['c_action', 'c_reward_chosen', 'c_value_reward', 'c_value_choice', 'c_contr_diff', 'c_value_state', 'c_choice_left', 'c_choice_right'],
    
    # The new module which handles the not-chosen value, does not need any additional inputs except for the value
    library_setup = {
        # 'x_learning_rate_reward': ['c_reward_chosen', 'c_value_reward', 'c_value_choice'],
        'x_value_reward_chosen': ['c_reward_chosen', 'c_value_choice'],
        'x_value_reward_not_chosen': ['c_reward_chosen', 'c_value_choice'],
        'x_value_choice_chosen': ['c_value_reward'],
        'x_value_choice_not_chosen': ['c_value_reward'],
        'x_value_state_left': ['c_contr_diff', 'c_value_reward', 'c_value_choice'],
        'x_value_state_right': ['c_contr_diff', 'c_value_reward', 'c_value_choice'],
    },

    # Further, the new module should be applied only to the not-chosen values
    filter_setup = {
        # 'x_learning_rate_reward': ['c_action', 1, True],
        'x_value_reward_chosen': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
        'x_value_state_left': ['c_choice_left', 1, True],
        'x_value_state_right': ['c_choice_right', 1, True],
    }    
)


class RNN_ContrDiff(BaseRNN):
    
    init_values = {
            'x_value_reward': 0.5,
            'x_value_choice': 0.,
            'x_learning_rate_reward': 0.,
            'x_value_state': 0,
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
        self.betas['x_value_state'] = self.setup_constant(embedding_size=embedding_size, leaky_relu=leaky_relu)
        
        # set up the submodules
        # self.submodules_rnn['x_learning_rate_reward'] = self.setup_module(input_size=4+self.embedding_size)
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=2+self.embedding_size)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=2+self.embedding_size)
        self.submodules_rnn['x_value_choice_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        self.submodules_rnn['x_value_choice_not_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        
        self.submodules_rnn['x_value_state_left'] = self.setup_module(input_size=3+self.embedding_size)
        self.submodules_rnn['x_value_state_right'] = self.setup_module(input_size=3+self.embedding_size)
        
        # NOTE: Only necessary if x_value_reward_chosen not a RNN-module
        # set up hard-coded equations
        # self.submodules_eq['x_value_reward_chosen'] = self.x_value_reward_chosen
    
    # NOTE: Only necessary if x_value_reward_chosen not a RNN-module
    # def x_value_reward_chosen(self, value, inputs):
    #     return value + inputs[..., 1] * (inputs[..., 0] - value)
    
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
        # rewards_not_chosen = ((1-actions) * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)
        
        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_id[:, 0].int())
        
        # set choice arrays to distinguish between left choices and right choices for the contrastive difference
        choice_left, choice_right = torch.zeros_like(actions[0]), torch.zeros_like(actions[0])
        choice_left[..., 0] = 1
        choice_right[..., 1] = 1
        
        for timestep, action, reward_chosen, contr_diff in zip(timesteps, actions, rewards_chosen, contr_diffs): #, rewards_not_chosen
            
            # record the current memory state and control inputs to the modules for SINDy training
            if not self.training and len(self.submodules_sindy)==0:
                self.record_signal('c_action', action)
                self.record_signal('c_reward_chosen', reward_chosen)
                # self.record_signal('c_reward_not_chosen', reward_not_chosen)
                self.record_signal('c_value_reward', self.state['x_value_reward'])
                self.record_signal('c_value_choice', self.state['x_value_choice'])
                self.record_signal('c_contr_diff', contr_diff)
                self.record_signal('c_value_state', self.state['x_value_state'])
                self.record_signal('c_choice_left', choice_left)
                self.record_signal('c_choice_right', choice_right)
                # NOTE: Only necessary if x_value_reward_chosen not a RNN-module
                # self.record_signal('x_learning_rate_reward', self.state['x_learning_rate_reward'])
                self.record_signal('x_value_reward_chosen', self.state['x_value_reward'])
                self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
                self.record_signal('x_value_choice_chosen', self.state['x_value_choice'])
                self.record_signal('x_value_choice_not_chosen', self.state['x_value_choice'])
                self.record_signal('x_value_state_left', self.state['x_value_state'])
                self.record_signal('x_value_state_right', self.state['x_value_state'])
                
            # NOTE: Only necessary if x_value_reward_chosen not a RNN-module            
            # # updates for x_value_reward
            # learning_rate_reward = self.call_module(
            #     key_module='x_learning_rate_reward',
            #     key_state='x_learning_rate_reward',
            #     action=action,
            #     inputs=(
            #         reward_chosen, 
            #         # reward_not_chosen, 
            #         self.state['x_value_reward'], 
            #         self.state['x_value_choice'],
            #         value_contr_diff,
            #         ),
            #     participant_embedding=participant_embedding,
            #     participant_index=participant_id,
            #     activation_rnn=torch.nn.functional.sigmoid,
            # )
            # next_value_reward_chosen = self.call_module(
            #     key_module='x_value_reward_chosen',
            #     key_state='x_value_reward',
            #     action=action,
            #     inputs=(
            #         reward_chosen, 
            #         learning_rate_reward,
            #         ),
            #     participant_embedding=participant_embedding,
            #     participant_index=participant_id,
            #     )
            
            # updates for x_value_reward
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(
                    reward_chosen, 
                    # reward_not_chosen, 
                    self.state['x_value_choice'],
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
                inputs=(
                    self.state['x_value_reward'],
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
                    self.state['x_value_reward'],
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )
            
            # get value for contrastive difference
            next_value_state_choice_left = self.call_module(
                key_module='x_value_state_left',
                key_state='x_value_state',
                inputs=(
                    contr_diff,
                    self.state['x_value_reward'],
                    self.state['x_value_choice'],
                    ),
                action=choice_left,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            next_value_state_choice_right = self.call_module(
                key_module='x_value_state_right',
                key_state='x_value_state',
                inputs=(
                    contr_diff,
                    self.state['x_value_reward'],
                    self.state['x_value_choice'],
                    ),
                action=choice_right,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            # updating the memory state
            # self.state['x_learning_rate_reward'] = learning_rate_reward
            self.state['x_value_state'] = next_value_state_choice_left + next_value_state_choice_right
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            self.state['x_value_choice'] = next_value_choice_chosen + next_value_choice_not_chosen
            
            # Now keep track of the logit in the output array
            logits[timestep] = self.state['x_value_reward'] * self.betas['x_value_reward'](participant_embedding) + self.state['x_value_choice'] * self.betas['x_value_choice'](participant_embedding) + self.state['x_value_state'] * self.betas['x_value_state'](participant_embedding)
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
                
        return logits, self.get_state()