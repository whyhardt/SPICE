"""Angular delta-rule model for weber2024 (Bruckner-style): a learned learning rate on a latent
belief, and a learned movement gain that carries the shield toward it.

    PE_t       = wrap(laser_t - belief_t)                                  (degrees)
    belief_t+1 = belief_t + alpha_t * PE_t,     alpha_t = sigmoid(learning_rate_value_t)
    shield_t+1 = shield_t + g_t * wrap(belief_t+1 - shield_t),  g_t = sigmoid(movement_value_t)

The learning rate is the only gain on the belief update (recurrent: surprise, catch and the block's
volatility/stochasticity drive it); the movement gain is stateless and depends on how far the belief
is from the shield. Beliefs are angles, so the update is rotation-invariant and stays on the circle.
The output is a point estimate of the next shield position (sin, cos), scored by clamped_angular_mse.
"""

import math

import pandas as pd
import torch
import torch.nn.functional as F

from spice import SpiceConfig, BaseModel, SpiceDataset


MOVEMENT_SPEED = 1.0        # degrees per frame

# |PE| has median ~16°, p99 ~79°: dividing by this scale (and clipping at 1) spreads it over [0, 1]
PE_SCALE = 90.0

# |belief - shield| scaled the same way
DISTANCE_SCALE = 90.0

# Binary indicator control signals (x^2 = x)
BINARY_SIGNALS = {'caught', 'volatility', 'stochasticity'}


CONFIG = SpiceConfig(
    library_setup={
        # Learning rate: updated every beam from surprise (|PE|, scaled), catch and the block's conditions
        'learning_rate_update': ('|prediction_error|', 'caught', 'volatility', 'stochasticity'),
        # Movement gain: how far the shield moves toward the belief, given their distance
        'movement_update': ('|belief_shield_distance|',),
    },

    memory_state={
        'belief_value': 0,          # belief about the laser position (degrees); set to the first beam
        'learning_rate_value': 0,   # alpha = sigmoid(value), initially 0.5
        'movement_value': 0,        # per-beam movement gain (reset each beam), g = sigmoid(value)
    },

    states_in_logit=[
        'learning_rate_value',
        'movement_value',
    ],

    additional_inputs=(
        'laser_caught',             # scored on the beam's onset frame (data_raw/data_processing.py)
        'volatility',
        'stochasticity',
        'trial_duration_frames',
        'trueMean',                 # plotting only (weinhardt2026/studies/weber2024/weber2024.py); never a model input
    ),
)


def wrap_degrees(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle difference to [-180, 180)."""
    return (angle + 180.) % 360. - 180.


def sincos_to_degrees(sincos: torch.Tensor) -> torch.Tensor:
    """(..., 2) [sin, cos] -> (..., 1) angle in degrees."""
    return torch.rad2deg(torch.atan2(sincos[..., :1], sincos[..., 1:2]))


def degrees_to_sincos(angle: torch.Tensor) -> torch.Tensor:
    """(..., 1) angle in degrees -> (..., 2) [sin, cos]."""
    radians = torch.deg2rad(angle)
    return torch.cat((torch.sin(radians), torch.cos(radians)), dim=-1)


def clamped_angular_mse(prediction: torch.Tensor, target: torch.Tensor, speed: float = MOVEMENT_SPEED, **kwargs) -> torch.Tensor:
    """MSE loss with physical movement speed clamping.

    The model predicts a raw belief position in (sin, cos) space.
    This loss clamps the predicted movement to be physically feasible
    given the inter-beam interval (dt) and movement speed.

    Args:
        prediction: (N, 2) model output [belief_sin, belief_cos]
        target: (N, 5) packed target [sin(shield_{t+1}), cos(shield_{t+1}),
                sin(shield_t), cos(shield_t), dt]
        speed: movement speed in degrees per frame
    """
    belief = prediction[..., :2]                    # model's belief
    actual = target[..., :2]                        # actual shield at t+1
    shield_t = target[..., 2:4]                     # shield at t (sin, cos)
    dt = target[..., 4:5]                           # inter-beam interval (frames)

    delta = belief - shield_t
    delta_mag = delta.norm(dim=-1, keepdim=True)

    # Max angular movement in sin/cos space ≈ speed * dt * (pi/180) for small angles
    # For large angles, the sin/cos delta saturates, but this is a reasonable approximation
    max_move = speed * dt * (math.pi / 180.0)
    fraction = torch.clamp_max(max_move / (delta_mag + 1e-8), 1.0)

    clamped_pred = shield_t + fraction * delta
    return F.mse_loss(clamped_pred, actual)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the angular shield/laser positions (degrees) into sin/cos components.

    Action = shield position (sin, cos) — the model's own position.
    Reward = laser position (sin, cos) — the outcome / prediction error source.
    """
    shield_rad = df['shieldRotation'] * (math.pi / 180.0)
    laser_rad = df['laserRotation'] * (math.pi / 180.0)

    df['shield_sin'] = shield_rad.apply(math.sin)
    df['shield_cos'] = shield_rad.apply(math.cos)
    df['laser_sin'] = laser_rad.apply(math.sin)
    df['laser_cos'] = laser_rad.apply(math.cos)

    return df


def prepare_dataset(dataset: SpiceDataset) -> SpiceDataset:
    """Pack the loss metadata into ys, which `clamped_angular_mse` needs.

    csv_to_dataset produces ys = [next_shield_sin, next_shield_cos]; the clamping
    additionally needs the current shield position and the inter-beam interval:
    ys = [next_shield_sin, next_shield_cos, shield_sin_t, shield_cos_t, dt]
    """
    n_actions = 2
    n_rewards = 2
    dt_col = n_actions + n_rewards + 3  # column index 7: trial_duration_frames

    shield_t = dataset.xs[:, :, :, :n_actions].clone()
    dt = dataset.xs[:, :, :, dt_col:dt_col + 1].clone()
    ys = torch.cat([dataset.ys, shield_t, dt], dim=-1)

    return SpiceDataset(dataset.xs, ys, n_reward_features=n_rewards, continuous_action=True)


# Hooks read by weinhardt2026/run.py
NORMALIZE_REWARDS = False       # laser positions are sin/cos, already in [-1, 1]
LOSS_FN = clamped_angular_mse
LOSS_FN_KWARGS = {}
# ys carries 3 extra metadata columns, so the action count cannot be inferred from it;
# the 2x2 conditions enter as module inputs, so all experiments share one coefficient set
ESTIMATOR_KWARGS = {'n_actions': 2, 'n_items': 1, 'n_experiments': 1}


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
        )

        # movement_update: stateless (reset each beam), a function of the belief-shield distance only
        self.setup_module(key_module='movement_update', include_state=False)

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

        # Positions as angles (degrees) — shapes after init: (T, W, E, B, 1)
        shield = sincos_to_degrees(spice_signals.actions)
        laser = sincos_to_degrees(spice_signals.feedback)

        # Embeddings
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # Initialize belief to the first beam
        if prev_state is None:
            self.state['belief_value'] = laser[0].clone()

        for trial in spice_signals.trials:

            # --- Prediction error: shortest arc from belief to beam ---
            prediction_error = wrap_degrees(laser[trial] - self.state['belief_value'])

            # --- Learning rate: updated every beam from surprise |PE| (scaled to [0, 1]) ---
            self.call_module(
                key_module='learning_rate_update',
                key_state='learning_rate_value',
                inputs=(
                    (prediction_error.abs() / PE_SCALE).clamp(max=1.),
                    spice_signals.additional_inputs['laser_caught'][trial],
                    spice_signals.additional_inputs['volatility'][trial],
                    spice_signals.additional_inputs['stochasticity'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            alpha = torch.sigmoid(self.state['learning_rate_value'])

            # --- Belief update: angular delta rule ---
            self.state['belief_value'] = (self.state['belief_value'] + alpha * prediction_error) % 360.

            # --- Movement gain: reset each beam, learns g(|belief - shield|) ---
            distance = wrap_degrees(self.state['belief_value'] - shield[trial])
            self.state['movement_value'] = torch.zeros_like(self.state['movement_value'])
            self.call_module(
                key_module='movement_update',
                key_state='movement_value',
                inputs=((distance.abs() / DISTANCE_SCALE).clamp(max=1.),),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            movement_gain = torch.sigmoid(self.state['movement_value'])

            # --- Next shield position: move part of the way toward the belief (speed clamp in the loss) ---
            spice_signals.logits[trial] = degrees_to_sincos(shield[trial] + movement_gain * distance)

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
