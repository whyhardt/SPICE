import torch

from spice import BaseModel
from spice import SpiceConfig


# RL architecture shared with the bandit studies (dezfouli2019, eckstein2026, ganesh2024a).
# Only the harvest item carries a value; the exit item stays at 0 as the reference point (as in the MVT).
# Harvesting takes the role of "chosen" and exiting the role of "not chosen" for the harvest value.
CONFIG = SpiceConfig(
    library_setup={
        'value_wm_reward_chosen': ['reward[t]'],
        'value_wm_reward_not_chosen': [],
        # 'value_reward_chosen': ['reward[t]'],
        # 'value_reward_not_chosen': [],
        'value_choice_chosen': [],
        'value_choice_not_chosen': [],
        # 'value_reward_environment': ['reward[t]'],
    },
    memory_state={
        'value_wm_reward': None,
        # 'value_reward': None,
        'value_choice': None,
        # 'value_reward_environment': None,
    },
    states_in_logit=[
        'value_wm_reward',
        # 'value_reward',
        'value_choice',
        # 'value_reward_environment',
    ],
    additional_inputs=('harvest_duration', 'travel_duration'),
)


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants, dropout=self.dropout,
        )

        self.setup_module(key_module='value_wm_reward_chosen', include_state=False)

    def forward(self, inputs, state=None):
        spice_signals = self.init_forward_pass(inputs, state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # only the harvest item (index 0) is updated; the exit item is kept at 0 as reference
        mask_harvest_value = torch.zeros_like(spice_signals.actions[0])
        mask_harvest_value[..., 0] = 1

        action_harvest = spice_signals.actions[..., 0].unsqueeze(-1).expand_as(spice_signals.actions)
        action_exit = spice_signals.actions[..., 1].unsqueeze(-1).expand_as(spice_signals.actions)
        rewards = spice_signals.feedback[..., 0].unsqueeze(-1).expand_as(spice_signals.actions)

        for trial in spice_signals.trials:

            harvested = action_harvest[trial] * mask_harvest_value
            exited = action_exit[trial] * mask_harvest_value

            # --- WORKING MEMORY UPDATES ---
            self.call_module(
                key_module='value_wm_reward_chosen',
                key_state='value_wm_reward',
                action_mask=harvested,
                inputs=(rewards[trial],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_wm_reward_not_chosen',
                key_state='value_wm_reward',
                action_mask=exited,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- REWARD VALUE UPDATES (patch value) ---
            # self.call_module(
            #     key_module='value_reward_chosen',
            #     key_state='value_reward',
            #     action_mask=harvested,
            #     inputs=(rewards[trial],),
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embedding,
            # )

            # self.call_module(
            #     key_module='value_reward_not_chosen',
            #     key_state='value_reward',
            #     action_mask=exited,
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embedding,
            # )

            # --- ENVIRONMENT REWARD UPDATES (not reset on exit; MVT opportunity cost) ---
            # self.call_module(
            #     key_module='value_reward_environment',
            #     key_state='value_reward_environment',
            #     action_mask=harvested,
            #     inputs=(rewards[trial],),
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embedding,
            # )

            # --- CHOICE VALUE UPDATES (continuation) ---
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=harvested,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=exited,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- LOGITS ---
            spice_signals.logits[trial] = (
                self.state['value_wm_reward']
                # + self.state['value_reward']
                + self.state['value_choice']
                # + self.state['value_reward_environment']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
