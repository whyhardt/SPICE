"""Visible-helicopter variant of the bruckner2025 model: spice_bruckner2025.py plus the placement under a helicopter
that is visible on the upcoming trial.

    delta rule:          d_t+1 = b_t + alpha_t * (target_t - b_t) + anchor_t          (as in spice_bruckner2025.py)
    hidden next trial:   b_t+1 = d_t+1
    visible next trial:  b_t+1 = d_t+1 + gamma_t * (mu_t+1 - d_t+1),   gamma_t = sigmoid(visible_value_t)

The helicopter is seen before the bucket is placed: on visible trials participants move a median 0.52 (no push) /
0.84 (push) of the way from the start position z to the helicopter on that same placement, against 0.22 / 0.72 on
hidden trials. spice_bruckner2025.py uses mu_t only for the following placement, so it cannot predict this jump.
Here the upcoming trial's visibility and helicopter position (v_next, mu_next) are inputs, as the upcoming start
position z_next already is; gamma = 1 places the bucket directly under the helicopter, gamma = 0 is
spice_bruckner2025.py (the variant nests it). The move toward the helicopter starts from the delta-rule placement,
not from the start position: after a changepoint the bag has often revealed the new mean already, and a participant
who has updated on it is close to the helicopter. Starting from the start position instead discards that update and
leaves gamma to average participants who jump to a far helicopter with those who ignore it (errors > 100 px).

The learning rate is untouched: it keeps its recurrence on every trial (visibility enters as its input v_t), and
after a visible trial the delta rule moves toward the revealed mean as before. gamma has its own stateless module so
that the jump cannot leak into the learning rate's recurrence. It is computed on every trial (input 0 when the next
trial is hidden) and used only before visible ones, so the SINDy fits in stage 1 and the stage-2 refit see the same
module.
"""

import pandas as pd
import torch

from spice import SpiceConfig, BaseModel

from studies.bruckner2025.spice_bruckner2025 import (  # shared loss, input scales and preprocessing
    ANCHOR_SCALE, BINARY_SIGNALS, ESTIMATOR_KWARGS, LOSS_FN, LOSS_FN_KWARGS, NORMALIZE_REWARDS, PE_SCALE,
    prepare_dataframe as prepare_dataframe_base,
)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """spice_bruckner2025.prepare_dataframe plus the upcoming trial's visibility and (scaled, masked) helicopter."""
    df = prepare_dataframe_base(df)
    blocks = df.groupby(['participant', 'experiment', 'block'])
    df['v_next'] = blocks['v_t'].shift(-1)
    df['mu_next'] = blocks['mu_t'].shift(-1)       # NaN unless the upcoming trial is visible
    return df


CONFIG = SpiceConfig(
    discovered_modules={
        # Learning rate: updated every trial from surprise (|PE|, scaled), catch and helicopter visibility
        'learning_rate_update': ('|prediction_error|', 'catch', 'v_t',),
        # Anchoring bias: learns compensation for bucket displacement
        'anchor_update': ('anchor_shift',),
        # Placement under the visible helicopter: share of the way from the delta-rule placement to the helicopter
        'visible_update': ('|helicopter_distance|',),
    },

    memory_state={
        'learning_rate_value': 0,  # alpha = sigmoid(value), initially 0.5
        'anchor_value': 0,         # per-trial anchoring correction (reset each trial)
        'visible_value': 0,        # per-trial placement gain under the visible helicopter (reset each trial)
    },


    additional_inputs=(
        'z_next',
        'catch',
        'v_t',
        'sigma',
        'r_t',
        'mu_t',
        'c_t',
        'v_next',
        'mu_next',
    ),
)


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size)

        # anchor_update: 1 control signal (anchor_shift); no state in library (always reset to 0)
        self.setup_module(key_module='anchor_update', include_state=False)
        # visible_update: 1 control signal (helicopter - delta-rule distance); no state in library (always reset to 0)
        self.setup_module(key_module='visible_update', include_state=False)

        self.preprocess_coefficients()

    def preprocess_coefficients(self):
        """Zero out SINDy terms that are structurally redundant for binary indicators.

        Binary indicators satisfy x^2 = x, so the squared term duplicates the linear one.
        """
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            binary_signals = [s for s in self.spice_config.discovered_modules[module] if s in BINARY_SIGNALS]
            for index_term, term in enumerate(candidate_terms[module]):
                factors = term.split('*')
                if any(signal + '^2' in factors for signal in binary_signals):
                    self.sindy_coefficients_presence[module][..., index_term] = 0
                    self.sindy_coefficients_prior_mask[module][..., index_term] = 0

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Extract signals — shapes after init: (T, W, E, B, ...)
        outcome = spice_signals.feedback                                 # x_t / 300
        bucket = spice_signals.actions                                  # b_t / 300

        # Embeddings
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids) if hasattr(self, 'experiment_embedding') else None
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        module_kwargs = dict(
            participant_index=spice_signals.participant_ids,
            participant_embedding=participant_embedding,
            experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
            experiment_embedding=experiment_embedding,
        )

        for trial in spice_signals.trials:

            # --- Prediction error relative to the current bucket position ---
            prediction_error = outcome[trial] - bucket[trial]

            # --- Anchor shift: bucket displacement from push (scaled to [-1, 1]) ---
            start = spice_signals.additional_inputs['z_next'][trial]
            anchor_shift = ((start - bucket[trial]) / ANCHOR_SCALE).clamp(min=-1., max=1.)

            # --- Update target: helicopter position when visible, outcome otherwise ---
            visible = spice_signals.additional_inputs['v_t'][trial]
            target = torch.where(visible > 0.5, spice_signals.additional_inputs['mu_t'][trial], outcome[trial])

            # --- Learning rate: updated every trial from surprise |PE| (scaled to [0, 1]) ---
            self.call_module(
                key_module='learning_rate_update',
                key_state='learning_rate_value',
                inputs=(
                    (prediction_error.abs() / PE_SCALE).clamp(max=1.),
                    spice_signals.additional_inputs['catch'][trial],
                    visible,
                ),
                **module_kwargs,
            )
            alpha = torch.sigmoid(self.state['learning_rate_value'])

            # --- Anchor correction: reset each trial, learns d * anchor_shift ---
            self.state['anchor_value'] = torch.zeros_like(self.state['anchor_value'])
            self.call_module(
                key_module='anchor_update',
                key_state='anchor_value',
                action_mask=None,
                inputs=(anchor_shift,),
                **module_kwargs,
            )

            # --- Delta rule + anchor correction: the placement without a visible helicopter ---
            delta_rule = bucket[trial] + alpha * (target - bucket[trial]) + self.state['anchor_value']

            # --- Move toward the helicopter if it is visible on the upcoming trial ---
            visible_next = spice_signals.additional_inputs['v_next'][trial] > 0.5
            
            # the helicopter where visible, the delta-rule placement elsewhere (no NaN: it would poison the gradients via where)
            helicopter = torch.where(visible_next, spice_signals.additional_inputs['mu_next'][trial].nan_to_num(), delta_rule)
            helicopter_distance = helicopter - delta_rule
            self.state['visible_value'] = torch.zeros_like(self.state['visible_value'])
            self.call_module(
                key_module='visible_update',
                key_state='visible_value',
                action_mask=None,
                inputs=((helicopter_distance.abs() / PE_SCALE).clamp(max=1.),),
                **module_kwargs,
            )
            gamma = torch.sigmoid(self.state['visible_value'])

            # --- Next bucket: a share gamma of the way from the delta-rule placement to the visible helicopter ---
            spice_signals.logits[trial] = delta_rule + gamma * helicopter_distance

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
