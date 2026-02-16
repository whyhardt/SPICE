from ..resources.estimator import SpiceConfig
from ..resources.rnn import BaseRNN

import torch


# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) INTERACTIONS BETWEEN REWARD-BASED AND CHOICE-BASED MODULES
# 2) BUFFER-BASED REWARD AND CHOICE MEMORY FOR 3 TIMESTEPS (could be easily extended)
# -------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    library_setup={
        # Value learning can depend on recent reward sequence (working memory)
        'value_reward_chosen': [
            'reward[t]',           
            'reward[t-1]', 
            'reward[t-2]',
            'reward[t-3]',
            # 'value_choice',
        ],
        'value_reward_not_chosen': [
            'reward[t-1]', 
            'reward[t-2]',
            'reward[t-3]',
            # 'value_choice',
            ],
        'value_reward_not_displayed': [
            'reward[t-1]', 
            'reward[t-2]',
            'reward[t-3]',
            # 'value_choice',
            ],
        'value_choice_chosen': [
            'choice[t-1]', 
            'choice[t-2]',
            'choice[t-3]',
            # 'value_reward',
            ],
        'value_choice_not_chosen': [
            'choice[t-1]', 
            'choice[t-2]',
            'choice[t-3]',
            # 'value_reward',
            ],
        'value_choice_not_displayed': [
            'choice[t-1]',
            'choice[t-2]',
            'choice[t-3]',
            # 'value_reward',
            ],
    },
    
    memory_state = {
        'value_reward': 0.5,      # reward value (enables slow learning)
        'value_choice': 0.0,      # choice value (enables slow learning)
        'buffer_reward_1': 0.5,   # t-1 reward
        'buffer_reward_2': 0.5,   # t-2 reward
        'buffer_reward_3': 0.5,   # t-3 reward
        'buffer_choice_1': 0.5,   # t-1 choice
        'buffer_choice_2': 0.5,   # t-2 choice
        'buffer_choice_3': 0.5,   # t-3 choice
    }
)


class SpiceModel(BaseRNN):
    """
    Working memory as explicit buffer of recent rewards.
    
    Key difference from value learning:
    - Stores individual past rewards (not aggregated statistics)
    - Fixed capacity (buffer size)
    - Perfect memory for items in buffer
    - Items fall out of buffer (discrete forgetting)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            
        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size)
        
        # Value learning module (slow updates)
        # Can use recent reward history to modulate learning
        self.setup_module(key_module='value_reward_chosen', input_size=4 + self.embedding_size)  # -> 21 terms
        self.setup_module(key_module='value_reward_not_chosen', input_size=3 + self.embedding_size)  # -> 15 terms
        self.setup_module(key_module='value_reward_not_displayed', input_size=3 + self.embedding_size)  # -> 15 terms
        self.setup_module(key_module='value_choice_chosen', input_size=3 + self.embedding_size) # -> 15 terms
        self.setup_module(key_module='value_choice_not_chosen', input_size=3 + self.embedding_size) # -> 15 terms -> 21+15+15+15 = 66 terms in total
        self.setup_module(key_module='value_choice_not_displayed', input_size=3 + self.embedding_size) # -> 15 terms -> 21+15+15+15 = 66 terms in total

    def forward(self, inputs, prev_state=None, batch_first=False):
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)

        # Get shown items (raw indices) - these are time-shifted, so they refer to the NEXT trial
        # additional_inputs shape: [T_out, W, B, n_add]; for RL W=1
        shown_at_0_current = spice_signals.additional_inputs[:, 0, :, 0].long()  # [T_out, B]
        shown_at_1_current = spice_signals.additional_inputs[:, 0, :, 1].long()  # [T_out, B]
        shown_at_0_next = spice_signals.additional_inputs[:, 0, :, 2].long()     # [T_out, B]
        shown_at_1_next = spice_signals.additional_inputs[:, 0, :, 3].long()     # [T_out, B]

        # perform time-invariant computations
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # perform time-variant computations
        for timestep in spice_signals.trials:

            actions_t = spice_signals.actions[timestep, 0]   # [B, n_actions]
            rewards_t = spice_signals.rewards[timestep, 0]   # [B, n_actions]

            # Transform input data from action space to item space

            # Determine which action was chosen
            action_idx = actions_t.argmax(dim=-1)

            # Map to item indices using current trial's shown items
            item_chosen_idx = torch.where(action_idx == 0, shown_at_0_current[timestep], shown_at_1_current[timestep])
            item_not_chosen_idx = torch.where(action_idx == 1, shown_at_0_current[timestep], shown_at_1_current[timestep])

            # Create one-hot masks
            item_chosen_onehot = torch.nn.functional.one_hot(item_chosen_idx, num_classes=self.n_items).float()
            item_not_chosen_onehot = torch.nn.functional.one_hot(item_not_chosen_idx, num_classes=self.n_items).float()
            item_not_displayed_onehot = 1 - (item_chosen_onehot + item_not_chosen_onehot)

            # Map rewards from action space to item space
            reward_action = rewards_t  # shape: (batch, n_actions)

            # Create reward tensor in item space (batch, n_items)
            reward_item = torch.zeros(reward_action.shape[0], self.n_items, device=reward_action.device)

            # Scatter rewards to the corresponding items:
            # Item at shown_at_0_current gets reward for action 0
            # Item at shown_at_1_current gets reward for action 1
            reward_item.scatter_(1, shown_at_0_current[timestep].unsqueeze(-1), reward_action[:, 0].unsqueeze(-1))
            reward_item.scatter_(1, shown_at_1_current[timestep].unsqueeze(-1), reward_action[:, 1].unsqueeze(-1))
            
            # REWARD VALUE UPDATES
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=item_chosen_onehot,
                inputs=(
                    reward_item,
                    self.state['buffer_reward_1'],  # Recent reward history
                    self.state['buffer_reward_2'],
                    self.state['buffer_reward_3'],
                    # self.state['value_choice'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=item_not_chosen_onehot,
                inputs=(
                    self.state['buffer_reward_1'],  # Recent reward history
                    self.state['buffer_reward_2'],
                    self.state['buffer_reward_3'],
                    # self.state['value_choice'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            self.call_module(
                key_module='value_reward_not_displayed',
                key_state='value_reward',
                action_mask=item_not_displayed_onehot,
                inputs=(
                    self.state['buffer_reward_1'],  # Recent reward history
                    self.state['buffer_reward_2'],
                    self.state['buffer_reward_3'],
                    # self.state['value_choice'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            # CHOICE VALUE UPDATES
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=item_chosen_onehot,
                inputs=(
                    self.state['buffer_choice_1'],  # Recent choice history
                    self.state['buffer_choice_2'],
                    self.state['buffer_choice_3'],
                    # self.state['value_reward'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=item_not_chosen_onehot,
                inputs=(
                    self.state['buffer_choice_1'],  # Recent choice history
                    self.state['buffer_choice_2'],
                    self.state['buffer_choice_3'],
                    # self.state['value_reward'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            self.call_module(
                key_module='value_choice_not_displayed',
                key_state='value_choice',
                action_mask=item_not_displayed_onehot,
                inputs=(
                    self.state['buffer_choice_1'],  # Recent choice history
                    self.state['buffer_choice_2'],
                    self.state['buffer_choice_3'],
                    # self.state['value_reward'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            # BUFFER UPDATES: 
            # REWARD BUFFER UPDATES: Shift reward buffer for chosen action, keep for not chosen action (NOTE: deterministic; not learned by SPICE -> Could be made learnable: e.g. decay-rate for not chosen action)
            # CHOICE BUFFER UPDATES: Shift all buffer entries according to action
            self.state['buffer_reward_3'] = self.state['buffer_reward_2'] * item_chosen_onehot + self.state['buffer_reward_3'] * (item_not_chosen_onehot+item_not_displayed_onehot)
            self.state['buffer_reward_2'] = self.state['buffer_reward_1'] * item_chosen_onehot + self.state['buffer_reward_2'] * (item_not_chosen_onehot+item_not_displayed_onehot)
            self.state['buffer_reward_1'] = torch.where(item_chosen_onehot==1, item_chosen_onehot, 0) + torch.where(item_chosen_onehot==0, self.state['buffer_reward_1'], 0)  # updating buffer_reward[t-1] with reward for chosen action and keeping values for not-chosen actions
            self.state['buffer_choice_3'] = self.state['buffer_choice_2']
            self.state['buffer_choice_2'] = self.state['buffer_choice_1']
            self.state['buffer_choice_1'] = item_chosen_onehot
            
            # Transform values from item space to action space for NEXT trial (for prediction)
            # Use the time-shifted items (next trial's items)
            value_reward_at_0 = torch.gather(self.state['value_reward'], 1, shown_at_0_next[timestep].unsqueeze(-1))
            value_reward_at_1 = torch.gather(self.state['value_reward'], 1, shown_at_1_next[timestep].unsqueeze(-1))
            value_choice_at_0 = torch.gather(self.state['value_choice'], 1, shown_at_0_next[timestep].unsqueeze(-1))
            value_choice_at_1 = torch.gather(self.state['value_choice'], 1, shown_at_1_next[timestep].unsqueeze(-1))
            
            value_at_0 = value_reward_at_0 + value_choice_at_0
            value_at_1 = value_reward_at_1 + value_choice_at_1
            
            # log action values
            spice_signals.logits[timestep] = torch.concat([value_at_0, value_at_1], dim=-1)

        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.get_state()
