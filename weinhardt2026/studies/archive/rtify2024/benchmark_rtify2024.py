import torch
from spice import SpiceDataset


DDM_PARAMETERS = {
    'drift_rates': [0.5, 0.75, 1.0, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75],
    'collapsing_bound_rate': 0.,
    'leak': 0.,
    'flip_time_range': None,
    'drift_update_kwargs': [
        dict(c_linear=0, c_quadratic=0),
        dict(c_linear=0, c_quadratic=0),
        dict(c_linear=0, c_quadratic=0),
        dict(c_linear=1.0, c_quadratic=0),
        dict(c_linear=0, c_quadratic=0.75),
        dict(c_linear=1.0, c_quadratic=-0.5),
        dict(c_linear=-1.0, c_quadratic=0),
        dict(c_linear=0, c_quadratic=-0.5),
        dict(c_linear=1.0, c_quadratic=-0.5),
        ],
    }

def update_drift(drift, c0, c1=0, c2=0):
    return c0 + c1 * drift + c2 * (drift ** 2)


def simulate_ddm(
    n_trials: int = 1000,
    t_max: float = 5.0,
    max_steps: int = 100,
    drift_rate: float = 1.0,
    diffusion_rate: float = 1.0,
    leak: float = 0.0,
    threshold: float = 3.0,
    collapsing_bound_rate: float = 0.0,
    threshold_min: float = 0.1,
    flip_time_range: tuple = None,
    drift_update_kwargs: dict = {},
    participant_id: int = 0,
    device=None,
) -> SpiceDataset:
    """Ground-truth simulator -- identical accumulator dynamics to
    rtify2024_c/benchmark_rtify2024_c.py. Only the OUTPUT format changes, to match the
    per-step [no_decision, up, down] cross-entropy training scheme (see
    spice_rtify2024_f.py's SpiceDDM docstring):

        dE/dt = -leak*E + drift(t) + diffusion*xi(t)
        threshold(t) = max(threshold - collapsing_bound_rate*t, threshold_min)

    xs: (n_trials, max_steps, 1, 10) -- [action_0, action_1, action_2 (unused placeholders;
        NaN after a trial's decision step, matching SPICE's `_run_batch_training` masking
        convention: `mask = ~isnan(xs[..., :n_actions])`), stimulus, time_elapsed, time_trial
        (metadata, unused), trial, block, experiment, participant].
    ys: (n_trials, max_steps, 1, 3) -- per-step one-hot [no_decision, up, down]: 1 in column 0
        for every step before the decision, 1 in column 1 or 2 at the exact decision step,
        all-zero (masked out via xs, ignored by the loss) after.

    Unlike rtify2024_c, this ys does NOT bake in `non_decision_time` -- it encodes the raw
    (undelayed) accumulator decision step, matching the model's own undelayed per-step
    prediction. `non_decision_time` is purely a reporting-time addition applied downstream, in
    analysis_rtify2024_f.py, when converting a decoded step back to a real-world RT for
    plotting -- both the "true" and the model-simulated RT get it added the same way there.
    This is a synthetic identity-check study with a known ground-truth non_decision_time, so
    there's no need to estimate it from data the way `estimate_non_decision_time` would for
    real behavioral RTs.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dt = t_max / max_steps

    stimulus_t = torch.full((n_trials, max_steps), 1, device=device)
    drift_t = torch.full((n_trials, max_steps), drift_rate, device=device)
    threshold_t = torch.full((n_trials, max_steps), threshold, device=device)
    evidence_t = torch.full((n_trials, max_steps), 0., device=device)
    
    if flip_time_range is not None:
        flip_frac = torch.empty(n_trials, device=device).uniform_(*flip_time_range)
        flip_step = (flip_frac * max_steps).long().clamp(1, max_steps - 1)
        time_idx = torch.arange(max_steps, device=device).unsqueeze(0)  # [1, max_steps]
        stimulus_t = torch.where(time_idx < flip_step.unsqueeze(1), -1, 1)  # [n_trials, max_steps]
        drift_t = drift_t * stimulus_t  # [n_trials, max_steps]

    noise = torch.randn(n_trials, max_steps, device=device)
    decay = max(1. - leak * dt, 0.)

    drift_step = drift_t[:, 0].clone()
    evidence_step = torch.zeros(n_trials, device=device)
    for w in range(max_steps):
        # Euler-integrated relaxation (dx = dt * (f(x) - x)) toward update_drift's fixed point,
        # rather than iterating the map directly -- the latter converges in a handful of steps
        # regardless of dt/max_steps, collapsing any c_linear/c_quadratic feedback into an
        # apparently-instant jump instead of a trajectory spread over the trial.
        drift_target = update_drift(
            drift_step,
            drift_update_kwargs.get('c0', drift_rate),
            drift_update_kwargs.get('c1', 0),
            drift_update_kwargs.get('c2', 0),
            )
        drift_step = drift_step + dt * (drift_target - drift_step)
        drift_t[:, w] = drift_step
        threshold_t[:, w] = max(threshold - collapsing_bound_rate * w * dt, threshold_min)
        evidence_t[:, w] = decay * evidence_step + drift_t[:, w] * dt + diffusion_rate * (dt ** 0.5) * noise[:, w]
        evidence_step = evidence_t[:, w]

    upper_mask = evidence_t >= threshold_t
    lower_mask = evidence_t <= -threshold_t
    has_upper = upper_mask.any(dim=1)
    has_lower = lower_mask.any(dim=1)
    first_upper = torch.where(has_upper, upper_mask.int().argmax(dim=1), torch.full_like(has_upper, max_steps, dtype=torch.long))
    first_lower = torch.where(has_lower, lower_mask.int().argmax(dim=1), torch.full_like(has_lower, max_steps, dtype=torch.long))

    decision_step = torch.minimum(first_upper, first_lower)
    # Trials that never cross a boundary within t_max are timeouts -- drop them, same as
    # rtify2024_c, rather than fabricating a censored decision step.
    responded = decision_step < max_steps
    n_valid = int(responded.sum().item())
    if n_valid < n_trials:
        print(f"simulate_ddm: dropped {n_trials - n_valid}/{n_trials} non-response (timeout) trials")

    choice = (first_upper <= first_lower).long()[responded]
    decision_step = decision_step[responded]
    stimulus_t = stimulus_t[responded]
    drift_t = drift_t[responded]
    threshold_t = threshold_t[responded]
    evidence_t = evidence_t[responded]
    is_up = choice == 1

    xs = torch.zeros(n_valid, max_steps, 13, device=device)
    xs[:, :, 3] = stimulus_t
    xs[:, :, 4] = torch.arange(max_steps, device=device) * dt  # time_elapsed (additional_input, fed to drift)
    xs[:, :, 5] = drift_t
    xs[:, :, 6] = threshold_t
    xs[:, :, 7] = evidence_t 
    xs[:, :, 8] = 0  # time_trial metadata slot (unused)
    xs[:, :, 9] = torch.arange(max_steps, device=device)  # trial metadata: per-step trial index
    xs[:, :, 10] = 0
    xs[:, :, 11] = 0
    xs[:, :, 12] = participant_id

    step_idx = torch.arange(max_steps, device=device).unsqueeze(0)  # [1, max_steps]
    decision_step_col = decision_step.unsqueeze(1)  # [n_valid, 1]
    before_decision = step_idx < decision_step_col
    at_decision = step_idx == decision_step_col
    after_decision = step_idx > decision_step_col

    ys = torch.zeros(n_valid, max_steps, 3, device=device)
    ys[:, :, 0] = before_decision.float()
    ys[:, :, 1] = (at_decision & is_up.unsqueeze(1)).float()
    ys[:, :, 2] = (at_decision & ~is_up.unsqueeze(1)).float()

    # NaN the action-placeholder columns for steps after the decision -- the standard SPICE
    # masking convention (_run_batch_training) drops these from the loss entirely, so a
    # trial's episode "ends" at its decision step exactly like a variable-length bandit session.
    for c in range(3):
        xs[:, :, c] = torch.where(after_decision, torch.full_like(xs[:, :, c], float('nan')), xs[:, :, c])

    xs = xs.unsqueeze(2)  # (n_valid, max_steps, 1, 10)
    ys = ys.unsqueeze(2)  # (n_valid, max_steps, 1, 3)

    dataset = SpiceDataset(xs, ys, n_reward_features=0)
    return dataset


def get_dataset(
    drift_rates,
    collapsing_bound_rate: list[float],
    leak: list[float],
    flip_time_range: list[float],
    drift_update_kwargs: list[dict],
    threshold: float,
    n_trials: int,
    t_max: float,
    max_steps: int,
    device: torch.device,
    **kwargs,
    ):
    """drift_rates: (n_participants,) -- one fixed stimulus/drift-rate condition
    per participant. n_trials: trials per participant."""
    if isinstance(drift_rates, (float, int)):
        drift_rates = [drift_rates]
    n_participants = len(drift_rates)

    if isinstance(collapsing_bound_rate, (float, int)):
        collapsing_bound_rate = [collapsing_bound_rate] * n_participants
    if isinstance(leak, (float, int)):
        leak = [leak] * n_participants
    if isinstance(drift_update_kwargs, dict):
        drift_update_kwargs = [drift_update_kwargs]* n_participants

    datasets = [
        simulate_ddm(
            n_trials=n_trials,
            t_max=t_max,
            max_steps=max_steps,

            drift_rate=drift_rates[i],
            diffusion_rate=1.0,
            leak=leak[i],
            threshold=threshold,
            collapsing_bound_rate=collapsing_bound_rate[i],
            flip_time_range=flip_time_range,
            drift_update_kwargs=drift_update_kwargs[i],

            participant_id=i,
            device=device,
        )
        for i in range(n_participants)
    ]
    xs = torch.cat([d.xs for d in datasets], dim=0)
    ys = torch.cat([d.ys for d in datasets], dim=0)
    dataset = SpiceDataset(xs, ys, n_reward_features=0)

    return dataset, dataset
