"""Delta-rule variant of the bruckner2025 model: a learned learning rate and an anchoring term.

    b_{t+1} = b_t + alpha_t * (target_t - b_t) + anchor_t
    target_t = mu_t if the helicopter is visible, else x_t

The learning rate alpha_t = sigmoid(learning_rate_value) carries all learning dynamics
(surprise, catch, visibility), so its equation is the only place where the gain of the
update lives -- unlike a separate belief module, whose gain would trade off against alpha.
"""

import pandas as pd
import torch

from spice import SpiceConfig, BaseModel


# Screen positions (bucket, outcome, helicopter) are given in pixels and
# normalized to [0, 1] by this scale before training.
POSITION_SCALE = 300.0

# Binary indicator control signals (x^2 = x)
BINARY_SIGNALS = {'catch', 'v_t'}


def mse_loss(prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    """MSE loss for continuous position prediction.

    Same interface as cross_entropy_loss: (prediction, target) -> scalar.
    Both tensors have shape (..., 1) after NaN masking in the training loop.
    Extra kwargs (e.g. label_smoothing) are accepted and ignored for compatibility.
    """
    return torch.nn.functional.mse_loss(prediction.reshape(-1), target.reshape(-1))


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Derive 'z_next', normalize screen positions and mask unobserved helicopter positions."""
    # next trial's initial bucket position (within each block)
    df['z_next'] = df.groupby(['participant', 'experiment', 'block'])['z_t'].shift(-1)

    # normalize positions to [0, 1]
    for col in ['b_t', 'x_t', 'mu_t', 'z_next', 'sigma']:
        df[col] = df[col] / POSITION_SCALE

    # participants can only observe the helicopter position when v_t = 1
    df.loc[df['v_t'] != 1, 'mu_t'] = float('nan')

    return df


# |PE| is mostly < 0.2 in screen units (p99 ~ 0.56); dividing by this scale (and clipping
# at 1) spreads it over [0, 1] so squared/interaction terms are not pruned for being tiny.
PE_SCALE = 0.5

# The push |z_{t+1} - b_t| is ~0.1 on average in screen units (p99 ~ 0.30); at that scale the
# anchor module's gain is too small to survive the gate penalty. Scaled by this and clipped to [-1, 1].
ANCHOR_SCALE = 0.3


CONFIG = SpiceConfig(
    library_setup={
        # Learning rate: updated every trial from surprise (|PE|, scaled), catch and helicopter visibility
        'learning_rate_update': ('|prediction_error|', 'catch', 'v_t',),
        # Anchoring bias: learns compensation for bucket displacement
        'anchor_update': ('anchor_shift',),
    },

    memory_state={
        'learning_rate_value': 0,  # alpha = sigmoid(value), initially 0.5
        'anchor_value': 0,         # per-trial anchoring correction (reset each trial)
    },

    states_in_logit=[
        'learning_rate_value',
        'anchor_value',
    ],

    additional_inputs=(
        'z_next',
        'catch',
        'v_t',
        'sigma',
        'r_t',
        'mu_t',
        'c_t',
    ),
)


# Hooks read by weinhardt2026/run.py
NORMALIZE_REWARDS = False       # outcomes are positions, already scaled by prepare_dataframe
LOSS_FN = mse_loss
LOSS_FN_KWARGS = {}
ESTIMATOR_KWARGS = {'n_items': 1}


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size)

        # anchor_update: 1 control signal (anchor_shift); no state in library (always reset to 0)
        self.setup_module(key_module='anchor_update', include_state=False)

        self.preprocess_coefficients()

    def preprocess_coefficients(self):
        """Zero out SINDy terms that are structurally redundant for binary indicators.

        Binary indicators satisfy x^2 = x, so the squared term duplicates the linear one.
        """
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            binary_signals = [s for s in self.spice_config.library_setup[module] if s in BINARY_SIGNALS]
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

        for trial in spice_signals.trials:

            # --- Prediction error relative to the current bucket position ---
            prediction_error = outcome[trial] - bucket[trial]

            # --- Anchor shift: bucket displacement from push (scaled to [-1, 1]) ---
            anchor_shift = (spice_signals.additional_inputs['z_next'][trial] - bucket[trial]) / ANCHOR_SCALE
            anchor_shift = anchor_shift.clamp(min=-1., max=1.)

            # --- Update target: helicopter position when visible, outcome otherwise ---
            mask_vt = spice_signals.additional_inputs['v_t'][trial]
            target = torch.where(mask_vt > 0.5, spice_signals.additional_inputs['mu_t'][trial], outcome[trial])

            # --- Learning rate: updated every trial from surprise |PE| (scaled to [0, 1]) ---
            self.call_module(
                key_module='learning_rate_update',
                key_state='learning_rate_value',
                inputs=(
                    (prediction_error.abs() / PE_SCALE).clamp(max=1.),
                    spice_signals.additional_inputs['catch'][trial],
                    mask_vt,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )
            alpha = torch.sigmoid(self.state['learning_rate_value'])

            # --- Anchor correction: reset each trial, learns d * anchor_shift ---
            self.state['anchor_value'] = torch.zeros_like(self.state['anchor_value'])
            self.call_module(
                key_module='anchor_update',
                key_state='anchor_value',
                action_mask=None,
                inputs=(anchor_shift,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Delta rule + anchor correction ---
            spice_signals.logits[trial] = (
                bucket[trial]
                + alpha * (target - bucket[trial])
                + self.state['anchor_value']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
