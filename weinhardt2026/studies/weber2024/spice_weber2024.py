import math

import pandas as pd
import torch
import torch.nn.functional as F

from spice import SpiceConfig, BaseModel, SpiceDataset


MOVEMENT_SPEED = 1.0        # degrees per frame


CONFIG = SpiceConfig(
    library_setup={
        # Belief update: split by catch outcome (externalized gating)
        'belief_update_caught': ('pe',),   # update when shield catches laser
        'belief_update_missed': ('pe',),   # update when shield misses laser
        # Dynamic learning rate: modulates gated output
        'certainty_update_caught': (),    # LR adapts when catching (tracking well)
        'certainty_update_missed':  (),    # LR decays when missing (tracking poorly)
    },

    memory_state={
        'belief_value': 0,    # internal belief about laser position (sin/cos as items 0, 1)
        'certainty_raw': 0,        # dynamic learning rate state; sigmoid(3) = 1.0
    },

    states_in_logit=[
        'belief_value', 
        'certainty_raw',
        ],

    additional_inputs=(
        'laser_caught',
        'volatility',
        'stochasticity',
        'trial_duration_frames',
        'trueMean',
    ),
)


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
# ys carries 3 extra metadata columns, so the action count cannot be inferred from it
ESTIMATOR_KWARGS = {'n_actions': 2, 'n_items': 2}


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.experiment_embedding = self.setup_embedding(
        #     num_embeddings=self.n_experiments,
        #     embedding_size=2,
        # )

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
        )
        
        self.alpha_raw = self.setup_constant()
    
    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Extract signals — shapes after init: (T, W, E, B, 2)
        # actions:  shield position (sin, cos) at trial t
        # rewards:  laser position (sin, cos) at trial t
        shield = spice_signals.actions
        laser = spice_signals.feedback

        # Embeddings
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids) if hasattr(self, 'experiment_embedding') else None
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # Initialize belief to first laser observation
        if prev_state is None:
            self.state['belief_value'] = self.state['belief_value'] + laser[0]

        for trial in spice_signals.trials:

            # --- Prediction error: laser minus belief (per sin/cos component) ---
            prediction_error = laser[trial] - self.state['belief_value']

            # --- Catch mask: externalized binary gating ---
            caught = spice_signals.additional_inputs['laser_caught'][trial]
            caught_mask = caught.expand_as(self.state['belief_value'])

            # --- Dynamic learning rate ---
            self.call_module(
                key_module='certainty_update_caught',
                key_state='certainty_raw',
                action_mask=caught_mask,
                # inputs=(prediction_error.detach(),),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='certainty_update_missed',
                key_state='certainty_raw',
                action_mask=1 - caught_mask,
                # inputs=(prediction_error.detach(),),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Gated output: shield_t + alpha * (belief - shield_t) ---
            # alpha ∈ [0, 1] via sigmoid; interpolates between current position and belief
            certainty_value = torch.sigmoid(self.state['certainty_raw'])
            
            # --- Belief update: split by catch outcome ---
            self.call_module(
                key_module='belief_update_caught',
                key_state='belief_value',
                action_mask=caught_mask,
                inputs=(
                    prediction_error,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='belief_update_missed',
                key_state='belief_value',
                action_mask=1 - caught_mask,
                inputs=(
                    prediction_error,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )
            
            # E_idx = torch.arange(self.ensemble_size, device=self.device).unsqueeze(1)
            # alpha = torch.sigmoid(
            #     self.alpha_raw[E_idx, spice_signals.participant_ids, spice_signals.experiment_ids]
            # ).expand(-1, -1, self.n_items)
            spice_signals.logits[trial] = (1-certainty_value) * shield[trial] + certainty_value * self.state['belief_value']
            # spice_signals.logits[trial] = self.state['belief_value']
            
        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
