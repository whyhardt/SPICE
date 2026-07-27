import torch
from spice import SpiceDataset

from weinhardt2026.studies.archive.rtify2024.analysis_rtify2024 import decode_choice_rt, estimate_non_decision_time

def simulate_ddm(
    n_trials: int = 1000,
    t_max: float = 5.0,
    max_steps: int = 100,
    drift_rate: float = 1.0,
    diffusion_rate: float = 1.0,
    leak: float = 0.0,
    threshold: float = 1.0,
    collapsing_bound_rate: float = 0.0,
    threshold_min: float = 0.1,
    non_decision_time: float = 0.2,
    flip_time_range: tuple = None,
    participant_id: int = 0,
    device=None,
) -> SpiceDataset:
    """One general ground-truth simulator: a two-boundary accumulator with each
    generalization toggled by its own parameter, all defaulting to "off" so this
    reduces to the plain DDM unless you explicitly turn something on.

        dE/dt = -leak*E + drift(t) + diffusion*xi(t)
        threshold(t) = max(threshold - collapsing_bound_rate*t, threshold_min)

    - `leak=0` -> perfect integrator (no leak).
    - `flip_time_range=None` -> drift(t) is constant at `drift_rate` (no flip). Given
      a range (as a fraction of `t_max`), drift(t) = +drift_rate before a per-trial
      random flip time drawn from that range, -drift_rate after -- the classic
      change-of-mind / reversal paradigm. A perfect (leak=0) integrator carries a
      large pre-flip evidence pileup that takes a long time to unwind after the
      flip; a leaky one forgets stale evidence faster and can change its mind sooner.
    - `collapsing_bound_rate=0` -> constant threshold (no collapsing bound).

    The (possibly flipping) drift value is also what's fed to the model as
    `stimulus` -- the participant directly observes it, same as in a real experiment.

    xs: (n_trials, 1, max_steps, 9) -- [action_0, action_1 (unused), stimulus,
        time_elapsed, time_trial (metadata, unused), trial, block, experiment, participant].
    ys: (n_trials, 1, max_steps, 2) -- one-hot indicator over (boundary, RT bin):
        ys[i, w, 0] = 1 iff trial i's observed decision is (up, bin w); ys[i, w, 1]
        likewise for "down". Exactly one of the `2 * max_steps` entries is 1 per trial.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dt = t_max / max_steps

    if flip_time_range is None:
        drift_t = torch.full((n_trials, max_steps), drift_rate, device=device)
    else:
        flip_frac = torch.empty(n_trials, device=device).uniform_(*flip_time_range)
        flip_step = (flip_frac * max_steps).long().clamp(1, max_steps - 1)
        time_idx = torch.arange(max_steps, device=device).unsqueeze(0)  # [1, max_steps]
        drift_t = torch.where(time_idx < flip_step.unsqueeze(1), drift_rate, -drift_rate)  # [n_trials, max_steps]

    noise = torch.randn(n_trials, max_steps, device=device)
    decay = max(1. - leak * dt, 0.)

    evidence_step = torch.zeros(n_trials, device=device)
    evidence = torch.zeros(n_trials, max_steps, device=device)
    for w in range(max_steps):
        evidence_step = decay * evidence_step + drift_t[:, w] * dt + diffusion_rate * (dt ** 0.5) * noise[:, w]
        evidence[:, w] = evidence_step

    threshold_t = torch.clamp(threshold - collapsing_bound_rate * torch.arange(max_steps, device=device) * dt, min=threshold_min)  # [max_steps]

    upper_mask = evidence >= threshold_t.unsqueeze(0)
    lower_mask = evidence <= -threshold_t.unsqueeze(0)
    has_upper = upper_mask.any(dim=1)
    has_lower = lower_mask.any(dim=1)
    first_upper = torch.where(has_upper, upper_mask.int().argmax(dim=1), torch.full_like(has_upper, max_steps, dtype=torch.long))
    first_lower = torch.where(has_lower, lower_mask.int().argmax(dim=1), torch.full_like(has_lower, max_steps, dtype=torch.long))

    decision_step = torch.minimum(first_upper, first_lower)
    # Trials that never cross a boundary within t_max are timeouts/non-responses --
    # drop them rather than fabricating a censored RT (forcing them into one bin
    # creates an artificial delta-spike in the "observed" distribution). A leaky
    # accumulator with a strong leak relative to threshold/t_max can genuinely fail
    # to resolve often; if that fraction is large, `leak`/`threshold`/`t_max` are
    # likely mismatched for the task, not just a detail to paper over.
    responded = decision_step < max_steps
    n_valid = int(responded.sum().item())
    if n_valid < n_trials:
        print(f"simulate_ddm: dropped {n_trials - n_valid}/{n_trials} non-response (timeout) trials")

    choice = (first_upper <= first_lower).long()[responded]
    decision_step = decision_step[responded]
    drift_t = drift_t[responded]
    response_time = decision_step.float() * dt + non_decision_time

    xs = torch.zeros(n_valid, max_steps, 9, device=device)
    xs[:, :, 2] = drift_t
    xs[:, :, 3] = torch.arange(max_steps, device=device) * dt  # time_elapsed (additional_input, fed to drift)
    xs[:, :, 4] = 0  # time_trial metadata slot (unused)
    xs[:, :, 5] = 0
    xs[:, :, 6] = 0
    xs[:, :, 7] = 0
    xs[:, :, 8] = participant_id

    rt_bin = torch.clamp((response_time / dt).long(), 0, max_steps - 1)
    row_idx = torch.arange(n_valid, device=device)
    is_up = choice == 1

    ys = torch.zeros(n_valid, max_steps, 2, device=device)
    ys[row_idx[is_up], rt_bin[is_up], 0] = 1.
    ys[row_idx[~is_up], rt_bin[~is_up], 1] = 1.

    xs = xs.unsqueeze(1)  # (n_valid, 1, max_steps, 9)
    ys = ys.unsqueeze(1)  # (n_valid, 1, max_steps, 2)

    return SpiceDataset(xs, ys, n_reward_features=0)

def get_dataset(
    drift_rates: list[float], 
    collapsing_bound_rate: list[float],
    leak: list[float],
    flip_time_range: list[float],
    n_trials: int, 
    t_max: float, 
    max_steps: int, 
    device: torch.device,
    ):
    dt = t_max / max_steps
    
    datasets = [
        simulate_ddm(
            n_trials=n_trials,
            t_max=t_max,
            max_steps=max_steps,

            drift_rate=rate,
            diffusion_rate=1.0,
            leak=leak[i],
            threshold=1.0,
            collapsing_bound_rate=collapsing_bound_rate[i],
            non_decision_time=0.2,
            flip_time_range=flip_time_range,

            participant_id=i,
            device=device,
        )
        for i, rate in enumerate(drift_rates)
    ]
    xs = torch.cat([d.xs for d in datasets], dim=0)
    ys = torch.cat([d.ys for d in datasets], dim=0)
    dataset = SpiceDataset(xs, ys, n_reward_features=0)

    n_train = int(0.8 * dataset.xs.shape[0])
    perm = torch.randperm(dataset.xs.shape[0])
    dataset_train = SpiceDataset(dataset.xs[perm[:n_train]], dataset.ys[perm[:n_train]], n_reward_features=0)
    dataset_test = SpiceDataset(dataset.xs[perm[n_train:]], dataset.ys[perm[n_train:]], n_reward_features=0)

    _, rt_train = decode_choice_rt(dataset_train.ys[:, 0], dt)
    non_decision_time = estimate_non_decision_time(rt_train)
    
    info_dataset = {}
    info_dataset['non_decision_time'] = non_decision_time
    
    return dataset_train, dataset_test, info_dataset