import torch

from spice import BaseModel
from spice import SpiceConfig


CONFIG = SpiceConfig(
    library_setup={
        'value_wm_reward_chosen': [
            'reward[t]',
            # 'reward[t-1]',
        ],
        'value_wm_reward_not_chosen': [],
        'value_reward_chosen': [
            'reward[t]',
            # 'value_reward_mean',
        ],
        'value_reward_not_chosen': [
            # 'value_reward_mean',
        ],
        'value_choice_chosen': [
            # 'action[t-1]',
        ],
        'value_choice_not_chosen': [
            # 'action[t-1]',
        ],
    },
    memory_state={
        'value_wm_reward': 0,
        'value_reward': 0,
        'value_choice': 0,
        
        # Buffers (excluded from logits)
        # 'action[t-1]': 0,
        # 'reward[t-1]': 0,
    },
    states_in_logit=[
        'value_wm_reward',
        'value_reward', 
        'value_choice', 
        ],
)


# Binary indicator control signals (x^2 = x) and mutually exclusive signal groups
# (x_i * x_j = 0): with 4 options no item is both adjacent and opposite to the choice,
# and the sign-split value change satisfies relu(dvalue) * relu(-dvalue) = 0. The squares
# of the sign-split signals are kept — unlike indicators, they are continuous.
BINARY_SIGNALS = {
    'reward[t]', 
    'reward[t-1]', 
    # 'action[t]', 
    # 'action[t-1]',
    }


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants, dropout=self.dropout,
        )
        
        self.setup_module(key_module='value_wm_reward_chosen', include_state=False)
                
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

        for trial in spice_signals.trials:
            
            # --- WORKING MEMORY UPDATES ---
            self.call_module(
                key_module='value_wm_reward_chosen',
                key_state='value_wm_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    spice_signals.feedback[trial],
                    # self.state['reward[t-1]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            self.call_module(
                key_module='value_wm_reward_not_chosen',
                key_state='value_wm_reward',
                action_mask=1-spice_signals.actions[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            # --- REWARD VALUE UPDATES ---
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    spice_signals.feedback[trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1 - spice_signals.actions[trial],
                # inputs=(
                # ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- CHOICE VALUE UPDATES ---
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[trial],
                # inputs=self.state['action[t-1]'],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1 - spice_signals.actions[trial],
                # inputs=self.state['action[t-1]'],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- BUFFER UPDATES ---
            # self.state['action[t-1]'] = spice_signals.actions[trial]

            # --- LOGITS ---
            spice_signals.logits[trial] = (
                self.state['value_reward']
                + self.state['value_wm_reward']
                + self.state['value_choice']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
