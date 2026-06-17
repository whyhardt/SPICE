import torch
import torch.nn as nn

from spice import SpiceEstimator, SpiceConfig, BaseModel, csv_to_dataset, SpiceDataset, split_data_along_blockdim

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from weinhardt2026.utils.benchmarking_gru import GRUModel, training


train_spice = True
train_gru = True


path_data = 'weinhardt2026/studies/archive/weber2024/data/weber2024_baseline.csv'
test_blocks = 2, 5, 9, 14

dataset = csv_to_dataset(
    file = path_data,
    df_participant_id='participant',
    df_experiment_id='experiment',
    df_choice='action',
    df_feedback='reward_change',
    df_block='block',
    additional_inputs=[
        # 'laserRotation',
        # 'trial_duration_frames',
        'shield_distance_initial',
        # 'total_movement_degrees',
        # 'frames_spent_moving',
        # 'button_press_onsets',
        # 'reaction_time_frames',
        # 'laser_caught',
        # 'hit_occurred',
        ],
    # timeshift_additional_inputs=(0, 0, 1),
)

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

# TODO: data preprocessing to produce shape (sessions, T_trials, W_within_trial_frames, features)
# with n_actions=2 (clockwise, counter-clockwise) and Gaussian-smoothed onset targets
# For now, assume dataset is provided in the correct shape.

path_data = 'weinhardt2026/studies/archive/weber2024/data/weber2024.csv'
test_blocks = 8, 10, 12

dataset = csv_to_dataset(
    file = path_data,
    df_participant_id='participant',
    df_experiment_id='experiment',
    df_choice='choice',
    df_feedback='reward',
    df_block='block',
    additional_inputs=['laserRotation', 'shieldRotation', 'totalReward'],
    timeshift_additional_inputs=(-1, -1, -1),
)

# restructure data to have only two actions (stay, move) instead of three (stay, move_clockwise, move_counter_clockwise)
move = dataset.xs[..., 1:3].sum(dim=-1, keepdim=True)
rewards_move = dataset.xs[..., 4:6].nan_to_num(0).sum(dim=-1, keepdim=True)
move_ys = dataset.ys[..., 1:3].sum(dim=-1, keepdim=True)
# create restructured dataset
xs = torch.concat((dataset.xs[..., :1], move, dataset.xs[..., 3:4], rewards_move, dataset.xs[..., 6:]), dim=-1)
ys = torch.concat((dataset.ys[..., :1], move_ys), dim=-1)
dataset = SpiceDataset(xs, ys)

if test_blocks is not None:
    dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)
else:
    dataset_train, dataset_test = dataset, None

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {dataset_train.n_participants}")
print(f"Number of experiments (baseline vs. infusion): {dataset_train.n_experiments}")
print(f"Number of actions in dataset: {dataset_train.n_actions}")
print(f"Number of additional inputs: {dataset_train.xs.shape[-1]-2*dataset_train.n_actions-3}")


# -------------------------------------------------------------------------------------------
# LOSS FUNCTION
# -------------------------------------------------------------------------------------------

def gaussian_onset_loss(prediction: torch.Tensor, target: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """MSE between predicted decision density and Gaussian-smoothed onset target.

    Applies a 1D Gaussian kernel (with std `sigma`) to the target along the
    frame dimension before computing MSE.  This turns a sharp one-hot onset
    indicator into a soft bump, rewarding the model for predicting decisions
    *near* the correct frame rather than requiring exact alignment.

    Args:
        prediction: (N, n_actions) — model decision density after NaN masking.
        target:     (N, n_actions) — raw onset indicators (1 at onset frame in
                    the chosen direction column, 0 elsewhere).
        sigma:      Gaussian kernel std in frames.  0 disables smoothing.
                    Adjust via ``functools.partial(gaussian_onset_loss, sigma=5)``.

    Note:
        The training loop flattens (E, B, T, W) into N before calling this
        function, so the convolution operates on the flat sequence.  Frames
        within a trial are contiguous; small sigma (≤5) keeps boundary bleed
        between adjacent trials negligible.
    """
    if sigma > 0 and target.shape[0] > 1:
        kernel_radius = int(3 * sigma)
        kernel_size = 2 * kernel_radius + 1  # always odd
        offsets = torch.arange(kernel_size, device=target.device, dtype=target.dtype) - kernel_radius
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        kernel = kernel / kernel.sum()

        n_actions = target.shape[-1]
        # (N, A) → (A, 1, N) for grouped conv1d, then back
        target_t = target.t().unsqueeze(1)                        # (A, 1, N)
        kernel_w = kernel.view(1, 1, -1).expand(n_actions, -1, -1)  # (A, 1, K)
        target_t = torch.nn.functional.conv1d(
            target_t, kernel_w, padding=kernel_radius, groups=n_actions,
        )
        target = target_t.squeeze(1).t()                          # (N, A)

    return torch.nn.functional.mse_loss(prediction, target)


# -------------------------------------------------------------------------------------------
# SPICE MODEL
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/archive/weber2024/params/spice_weber2024.pkl'

# The SPICE model combines between-trial learning with within-trial DDM dynamics.
# Actions: clockwise (0) vs counter-clockwise (1) — two DDM boundaries.
# Stay = DDM doesn't cross either boundary within the trial window.
# Each trial spans until: a) the next laser beam (max time) or b) DDM crosses a boundary.
#
# Between-trial modules:
#   mean_laser      — learns to estimate the laser mean from observed laser positions
#   start_position  — DDM initial evidence level; carries over between trials when
#                     participant doesn't decide in time (makes next decision faster)
#   drift           — computes DDM drift rate from laser-shield discrepancy

spice_config = SpiceConfig(
    library_setup={
        'mean_laser': ('laser_rotation',),       # between-trial: observed laser position
        'start_position': ('ddm_evidence',),     # between-trial: previous trial's final DDM evidence
        'drift': ('diff_rotation',),             # between-trial: laser-shield difference → drift rate
    },
    memory_state={
        'mean_laser': 0.,
        'start_position': None,                  # learnable per-participant initial DDM bias
    },
    states_in_logit=[],                          # logits produced by DDM, not sum-of-states
    additional_inputs=('laserRotation', 'shieldRotation', 'totalReward'),
)


class SpiceModel(BaseModel):

    def __init__(self, threshold: float = 1.0, temperature: float = 0.1, **kwargs):
        super().__init__(**kwargs)

        self.threshold = threshold       # DDM boundary height (symmetric: +threshold and -threshold)
        self.temperature = temperature   # softness of boundary crossing for differentiable decision density

        self.participant_embedding = self.setup_embedding(self.n_participants, dropout=self.dropout)
        self.experiment_embedding = self.setup_embedding(self.n_experiments, embedding_size=1, dropout=self.dropout)

        # Between-trial modules
        self.setup_module(key_module='mean_laser', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='start_position', input_size=1, dropout=self.dropout)  # input: previous final evidence
        self.setup_module(key_module='drift', input_size=1, dropout=self.dropout)            # input: laser-shield diff

    def forward(self, inputs, state=None):

        spice_signals = self.init_forward_pass(inputs, state)

        # Extract dimensions from canonical shape (T, W, E, B, F) after init_forward_pass
        first_state = self.state[next(iter(self.state))]
        T = len(spice_signals.trials)
        W = first_state.shape[0]  # within-trial timesteps from state shape
        E = self.ensemble_size
        B = first_state.shape[2]

        # Override logits shape: (T, W, E, B, n_actions=2) instead of standard (T, 1, E, B, A)
        spice_signals.logits = torch.zeros((T, W, E, B, self.n_actions), device=self.device)

        # Get additional inputs: shape (T, W, E, B, 1) → expand to items
        laser_rotation = spice_signals.additional_inputs['laserRotation'].expand(-1, -1, -1, -1, self.n_items)
        shield_rotation = spice_signals.additional_inputs['shieldRotation'].expand(-1, -1, -1, -1, self.n_items)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids)

        # Track DDM final evidence across trials (initialized to zero for first trial)
        ddm_final_evidence = torch.zeros(E, B, self.n_items, device=self.device)

        for trial in spice_signals.trials:

            # --- Between-trial module 1: Update laser mean estimate ---
            # Only use the first within-trial frame (laser position at onset is constant within trial)
            self.call_module(
                key_module='mean_laser',
                key_state='mean_laser',
                inputs=(
                    laser_rotation[trial, 0].unsqueeze(0),  # [1, E, B, I]
                ),
                participant_embedding=participant_embedding,
                participant_index=spice_signals.participant_ids,
                experiment_embedding=experiment_embedding,
                experiment_index=spice_signals.experiment_ids,
            )

            # --- Between-trial module 2: Update DDM starting evidence ---
            # Receives previous trial's final DDM evidence; learns carry-over dynamics
            self.call_module(
                key_module='start_position',
                key_state='start_position',
                inputs=(
                    ddm_final_evidence.unsqueeze(0),  # [1, E, B, I]
                ),
                participant_embedding=participant_embedding,
                participant_index=spice_signals.participant_ids,
                experiment_embedding=experiment_embedding,
                experiment_index=spice_signals.experiment_ids,
            )

            # --- Between-trial module 3: Compute drift rate ---
            # Drift based on discrepancy between laser mean estimate and shield position
            diff = self.state['mean_laser'][-1] - shield_rotation[trial, 0]  # [E, B, I]

            drift_rate = self.call_module(
                key_module='drift',
                inputs=(
                    diff.unsqueeze(0),  # [1, E, B, I]
                ),
                participant_embedding=participant_embedding,
                participant_index=spice_signals.participant_ids,
                experiment_embedding=experiment_embedding,
                experiment_index=spice_signals.experiment_ids,
            )  # [1, E, B, I]

            # --- Within-trial DDM evidence accumulation ---
            # Use first item's drift (single accumulator, direction encoded by sign)
            drift_scalar = drift_rate[-1, :, :, 0:1]    # [E, B, 1]
            start_evidence = self.state['start_position'][-1, :, :, 0:1]  # [E, B, 1]

            # Expand drift to all within-trial steps
            drift_expanded = drift_scalar.unsqueeze(0).expand(W, -1, -1, -1)  # [W, E, B, 1]
            epsilon = torch.randn_like(drift_expanded)

            # Evidence increments: drift + diffusion noise (dt=1, diffusion=1)
            evidence_increments = drift_expanded + epsilon
            evidence = torch.cumsum(evidence_increments, dim=0)  # [W, E, B, 1]

            # Add starting position (carries over from previous trial)
            evidence = evidence + start_evidence.unsqueeze(0)    # [W, E, B, 1]

            # --- Differentiable decision density ---
            decision_density = self._compute_decision_density(evidence)  # [W, E, B, 2]

            # Store in logits
            spice_signals.logits[trial] = decision_density

            # Save final evidence for next trial's start_position module
            ddm_final_evidence = evidence[-1].detach().expand(-1, -1, self.n_items)  # [E, B, I]

        spice_signals = self.post_forward_pass(spice_signals)

        return spice_signals.logits, self.state

    def _compute_decision_density(self, evidence: torch.Tensor) -> torch.Tensor:
        """Convert DDM evidence trajectory to per-frame decision density.

        Args:
            evidence: [W, E, B, 1] — accumulated evidence at each within-trial frame.
                      Positive → clockwise, negative → counter-clockwise.

        Returns:
            decision_density: [W, E, B, 2] — P(decide clockwise at frame t), P(decide counter-clockwise at frame t)
        """
        _W, E, B, _ = evidence.shape

        # Probability of crossing boundary at each frame (soft threshold)
        p_cross = torch.sigmoid((evidence.abs() - self.threshold) / self.temperature)  # [W, E, B, 1]

        # Survival probability: P(haven't decided before frame t)
        # Use log-space for numerical stability
        log_not_cross = torch.log(1 - p_cross + 1e-8)  # [W, E, B, 1]
        log_cumulative = torch.cumsum(log_not_cross, dim=0)

        # Shift: survival[t] = P(survived through frame t-1) = exp(sum of log(1-p) for s < t)
        # survival[0] = 1 (haven't had a chance to decide yet)
        survival_shifted = torch.cat([
            torch.zeros(1, E, B, 1, device=evidence.device),  # log(1) = 0
            log_cumulative[:-1]
        ], dim=0)
        survival_shifted = torch.exp(survival_shifted)  # [W, E, B, 1]

        # First-passage density: P(decide at exactly frame t)
        density = p_cross * survival_shifted  # [W, E, B, 1]

        # Direction: positive evidence → clockwise (action 0), negative → counter-clockwise (action 1)
        p_clockwise = torch.sigmoid(evidence / self.temperature)  # [W, E, B, 1]

        decision_cw = density * p_clockwise
        decision_ccw = density * (1 - p_clockwise)

        return torch.cat([decision_cw, decision_ccw], dim=-1)  # [W, E, B, 2]


# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=spice_config,
    spice_class=SpiceModel,
    kwargs_spice_class={
        'threshold': 1.0,
        'temperature': 0.1,
    },
    loss_fn=gaussian_onset_loss,

    n_actions=2,                          # clockwise, counter-clockwise
    n_participants=dataset_train.n_participants,
    n_experiments=dataset_train.n_experiments,

    epochs=500,
    warmup_steps=100,
    learning_rate=1e-3,
    device=device,

    ensemble_size=4,
    batch_size=None,                      # auto-detect

    use_sindy=False,                      # start with RNN-only; enable later for equation discovery
    sindy_weight=0.,
    sindy_alpha=0.,
    sindy_library_polynomial_degree=2,

    verbose=True,
)

if train_spice:
    estimator.fit(
        data=dataset_train.xs,
        targets=dataset_train.ys,
        data_test=dataset_test.xs if dataset_test is not None else None,
        target_test=dataset_test.ys if dataset_test is not None else None,
    )
    estimator.save_spice(path_spice)


# -------------------------------------------------------------------------------------------
# GRU FOR BENCHMARKING
# -------------------------------------------------------------------------------------------

path_gru = path_spice.replace('spice', 'gru')

gru = GRUModel(n_actions=dataset_train.n_actions, additional_inputs=3).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

if train_gru:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    epochs = 1000

    gru = training(
        model=gru,
        optimizer=optimizer,
        dataset_train=dataset_train,
        dataset_test=dataset_train,
        epochs=epochs,
        batch_size=1024,
        )

    torch.save(gru.state_dict(), path_gru)
    print("Trained GRU parameters saved to " + path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))
    

# -------------------------------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------------------------------

from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation

# estimator.eval()
gru.eval().to(torch.device('cpu'))

print(analysis_model_evaluation(
    dataset=dataset_train,
    # spice_model=estimator,
    gru_model=gru,
))

print(analysis_model_evaluation(
    dataset=dataset_test,
    # spice_model=estimator,
    gru_model=gru,
))

