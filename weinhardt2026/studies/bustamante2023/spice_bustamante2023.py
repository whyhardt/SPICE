import pandas as pd
import torch

from spice import BaseModel
from spice import SpiceConfig


# RL architecture shared with the bandit studies (dezfouli2019, eckstein2026, ganesh2024a).
# Only the harvest item carries a value; the exit item stays at 0 as the reference point (as in the MVT).
# Harvesting takes the role of "chosen" and exiting the role of "not chosen" for the harvest value.
# The leave threshold depends on the block's effort condition (travel task x effort level), as the fixed
# per-condition MVT threshold in Bustamante et al. (2023, Eqs. 2-5): its SINDy terms b0 + b1*is_msit + b2*is_high
# + b3*is_msit*is_high are the four condition thresholds, with b2 the extra cost of high-effort travel.
CONFIG = SpiceConfig(
    library_setup={
        'value_wm_reward_chosen': ['reward[t]'],
        'value_wm_reward_not_chosen': [],
        'value_reward_chosen': ['reward[t]'],
        'value_reward_not_chosen': [],
        'value_choice': ['choice[t]'],
        'leave_threshold': ['is_msit', 'is_high'],
        # 'value_choice_chosen': [],
        # 'value_choice_not_chosen': [],
        # 'value_reward_environment': ['reward[t]'],
    },
    memory_state={
        'value_wm_reward': 0,
        'value_reward': 0,
        'value_choice': 0,
        'leave_threshold': 0,       # per-trial leave threshold (reset each trial)
        # 'value_reward_environment': None,
    },
    states_in_logit=[
        'value_wm_reward',
        'value_reward',
        'value_choice',
        'leave_threshold',
        # 'value_reward_environment',
    ],
    additional_inputs=('harvest_duration', 'travel_duration', 'is_msit', 'is_high'),
)


BINARY_SIGNALS = ['choice[t]', 'is_msit', 'is_high']


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """The block's effort condition as binary inputs (participants knew it from the background colour):
    is_msit = cognitive travel task (MSIT) vs. physical (button pressing), is_high = high vs. low effort level."""
    df['is_msit'] = (df['travelTask'] == 'MSIT').astype(float)
    df['is_high'] = (df['effort_level'] == 'high').astype(float)
    return df


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants, dropout=self.dropout,
        )

        # self.setup_module(key_module='value_wm_reward_chosen', include_state=False)

        # leave_threshold: stateless (reset each trial), a function of the block's effort condition only
        self.setup_module(key_module='leave_threshold', include_state=False)
        
        self.preprocess_coefficients()
        
    def preprocess_coefficients(self):
        """Zero out SINDy terms that are structurally redundant for binary indicators.

        Binary indicators satisfy x^2 = x, so the squared term duplicates the linear one.
        Two indicators of the same mutually exclusive group satisfy x_i * x_j = 0.
        """
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            control_signals = self.spice_config.library_setup[module]
            binary_signals = [s for s in control_signals if s in BINARY_SIGNALS]
            for index_term, term in enumerate(candidate_terms[module]):
                factors = term.split('*')
                redundant = any(signal + '^2' in factors for signal in binary_signals)
                if redundant:
                    self.sindy_coefficients_presence[module][..., index_term] = 0
                    self.sindy_coefficients_prior_mask[module][..., index_term] = 0

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
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=harvested,
                inputs=(rewards[trial],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=exited,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

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
                key_module='value_choice',
                key_state='value_choice',
                action_mask=mask_harvest_value,
                inputs=harvested,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            # self.call_module(
            #     key_module='value_choice_chosen',
            #     key_state='value_choice',
            #     action_mask=harvested,
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embedding,
            # )

            # self.call_module(
            #     key_module='value_choice_not_chosen',
            #     key_state='value_choice',
            #     action_mask=exited,
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embedding,
            # )

            # --- LEAVE THRESHOLD (effort condition of the block; harvest item only, reset each trial) ---
            self.state['leave_threshold'] = torch.zeros_like(self.state['leave_threshold'])
            self.call_module(
                key_module='leave_threshold',
                key_state='leave_threshold',
                action_mask=mask_harvest_value,
                inputs=(
                    spice_signals.additional_inputs['is_msit'][trial],
                    spice_signals.additional_inputs['is_high'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- LOGITS ---
            spice_signals.logits[trial] = (
                self.state['value_wm_reward']
                + self.state['value_reward']
                + self.state['value_choice']
                - self.state['leave_threshold']
                # + self.state['value_reward_environment']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
