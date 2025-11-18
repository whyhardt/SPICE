import torch

from spice.estimator import SpiceConfig
from spice.resources.rnn import BaseRNN


spice_config = SpiceConfig(
    library_setup={
        'value_reward_chosen': ['reward'],
        'value_reward_not_chosen': [],
        'value_reward_not_displayed': [],
    },
    
    memory_state={
        'value_reward': 0.5,
        },
)

class SPICERNN(BaseRNN):

    def __init__(
        self, 
        n_actions, 
        spice_config, 
        n_participants, 
        n_items,
        use_sindy=False, 
        **kwargs):
        super().__init__(
            n_actions=n_actions, 
            spice_config=spice_config,
            n_participants=n_participants, 
            n_items=n_items, 
            embedding_size=32,
            sindy_ensemble_size=1,
            use_sindy=use_sindy,
            )

        self.participant_embedding = self.setup_embedding(num_embeddings=n_participants, embedding_size=self.embedding_size)

        self.submodules_rnn['value_reward_chosen'] = self.setup_module(1+self.embedding_size)
        self.submodules_rnn['value_reward_not_chosen'] = self.setup_module(self.embedding_size)
        self.submodules_rnn['value_reward_not_displayed'] = self.setup_module(self.embedding_size)

    def forward(self, inputs, prev_state, batch_first=False):

        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)

        # Get shown items (raw indices) - these are time-shifted, so they refer to the NEXT trial
        shown_at_0_current = spice_signals.additional_inputs[..., 0].long()
        shown_at_1_current = spice_signals.additional_inputs[..., 1].long()
        shown_at_0_next = spice_signals.additional_inputs[..., 2].long()
        shown_at_1_next = spice_signals.additional_inputs[..., 3].long()

        participant_embeddings = self.participant_embedding(spice_signals.participant_ids)

        for timestep in spice_signals.timesteps:

            # Transform input data from action space to item space

            # Determine which action was chosen
            action_idx = spice_signals.actions[timestep].argmax(dim=-1)

            # Map to item indices using current trial's shown items
            item_chosen_idx = torch.where(action_idx == 0, shown_at_0_current[timestep], shown_at_1_current[timestep])
            item_not_chosen_idx = torch.where(action_idx == 1, shown_at_0_current[timestep], shown_at_1_current[timestep])

            # Create one-hot masks
            item_chosen_onehot = torch.nn.functional.one_hot(item_chosen_idx, num_classes=self.n_items).float()
            item_not_chosen_onehot = torch.nn.functional.one_hot(item_not_chosen_idx, num_classes=self.n_items).float()
            item_not_displayed_onehot = 1 - (item_chosen_onehot + item_not_chosen_onehot)

            # Map rewards from action space to item space
            reward_action = spice_signals.rewards[timestep, :]  # shape: (batch, n_actions)

            # Create reward tensor in item space (batch, n_items)
            reward_item = torch.zeros(reward_action.shape[0], self.n_items, device=reward_action.device)

            # Scatter rewards to the corresponding items:
            # Item at shown_at_0_current gets reward for action 0
            # Item at shown_at_1_current gets reward for action 1
            reward_item.scatter_(1, shown_at_0_current[timestep].unsqueeze(-1), reward_action[:, 0].unsqueeze(-1))
            reward_item.scatter_(1, shown_at_1_current[timestep].unsqueeze(-1), reward_action[:, 1].unsqueeze(-1))
            
            # Update chosen
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=item_chosen_onehot,
                inputs=reward_item,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )

            # Update not chosen
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=item_not_chosen_onehot,
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )

            # Update not displayed
            self.call_module(
                key_module='value_reward_not_displayed',
                key_state='value_reward',
                action_mask=item_not_displayed_onehot,
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embeddings,
            )

            # Transform values from item space to action space for NEXT trial (for prediction)
            # Use the time-shifted items (next trial's items)
            value_at_0 = torch.gather(self.state['value_reward'], 1, shown_at_0_next[timestep].unsqueeze(-1))
            value_at_1 = torch.gather(self.state['value_reward'], 1, shown_at_1_next[timestep].unsqueeze(-1))

            # log action values
            spice_signals.logits[timestep] = torch.concat([value_at_0, value_at_1], dim=-1)

        spice_signals = self.post_forward_pass(spice_signals, batch_first)

        return spice_signals.logits, self.get_state()