"""
Synthetic DDM study: pure response-time + decision dynamics, no belief updating.

Ground truth: a classic two-boundary drift-diffusion process (spice/precoded/ddm.py's
`simulate_ddm`, adapted), including a flipping-stimulus / leaky-accumulator variant
(`simulate_ddm_flip`). Fitted model: a single 'drift' RNN submodule that receives the
trial's (stimulus, elapsed_time) repeated across all `max_steps` within-trial timesteps
in ONE `call_module` call -- the underlying EnsembleRNNModule already loops over
within-trial timesteps internally (compiled via torch.compile), so the resulting
`self.state['drift']` is a genuinely time-varying trajectory over the decision.

`drift` *is* the evidence -- there's no separate external integration step. The
module's own state naturally accumulates via its residual update (h[t+1] = h[t] +
dt*n[t], now that `dt` is threaded through `setup_module`/`EnsembleRNNModule`/the SINDy
fit so per-step increments read as per-unit-time rates, not tiny per-step deltas that
shrink as `max_steps` grows). "Leak" isn't a separate hand-designed parameter: it's
whatever self-coefficient SINDy discovers on `drift[t]` in `drift[t+1] = a*drift[t] +
b*stimulus[t] + c*time_elapsed[t]` -- a<1 is decay, a=1 is a perfect integrator, both
expressible by the same equation instead of two different code paths.

That evidence trajectory is turned into a discrete-time two-boundary hazard/survival
distribution (RTify's example2.ipynb idea) instead of ddm.py's noisy SDE simulation +
custom-autograd threshold detection:

    p_stop_up[w]   = sigmoid(gain * ( drift[w] - threshold))
    p_stop_down[w] = sigmoid(gain * (-drift[w] - threshold))
    p_stop[w]      = p_stop_up[w] + p_stop_down[w] - p_stop_up[w] * p_stop_down[w]
    p_decision_*[w] = p_stop_*[w] * prod_{k<w}(1 - p_stop[k])

`p_decision_up` + `p_decision_down` is a proper joint distribution over (choice, RT bin),
so choice and RT are fit by a single closed-form negative log-likelihood term -- fully
differentiable, no custom autograd Function, no distribution-matching loss.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator, BaseModel, SpiceConfig, SpiceDataset


CONFIG = SpiceConfig(
    library_setup={
        'drift': ['stimulus', 'time_elapsed'],
    },
    memory_state={
        'drift': 0.,
    },
    additional_inputs=('stimulus', 'time_elapsed'),
)


class DDMRNN(BaseModel):
    """Two-boundary DDM with a hazard-based (RTify-style) decision-time likelihood."""

    def __init__(self, max_steps: int = 100, t_max: float = 5.0, non_decision_time: float = 0., **kwargs):
        super().__init__(**kwargs)

        self.max_steps = max_steps
        self.t_max = t_max
        self.dt = t_max / max_steps
        self.non_decision_time = non_decision_time

        self.hazard_gain = torch.nn.Parameter(torch.tensor(0.))
        self.hazard_threshold = torch.nn.Parameter(torch.tensor(0.))

        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=self.dropout)

        # Re-setup with dt: BaseModel's automatic setup_modules_from_config() already
        # created 'drift' with the default dt=1. -- override so its residual update and
        # SINDy fit both scale by the trial's actual per-step time interval, letting
        # discovered coefficients read as per-second rates rather than per-step deltas
        # that shrink as `max_steps` grows.
        self.setup_module(key_module='drift', dt=self.dt, dropout=self.dropout)

    def forward(self, inputs: torch.Tensor, prev_state: torch.Tensor = None):
        spice_signals = self.init_forward_pass(inputs, prev_state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # Single outer trial (T=1): additional_inputs are [W=max_steps, E, B, 1].
        # One call_module invocation lets the RNN process the whole within-trial
        # sequence internally (its own compiled loop over W), instead of us looping.
        stimulus = spice_signals.additional_inputs['stimulus'][0]
        time_elapsed = spice_signals.additional_inputs['time_elapsed'][0]

        self.call_module(
            key_module='drift',
            key_state='drift',
            action_mask=None,
            inputs=(stimulus, time_elapsed),
            participant_index=spice_signals.participant_ids,
            participant_embedding=participant_embedding,
        )  # self.state['drift']: [W=max_steps, E, B, n_items]

        # `drift` *is* the evidence: no separate external integration step. Its own
        # residual state already accumulates (h[t+1] = h[t] + dt*n[t], now dt-scaled
        # via setup_module(dt=...)); "leak" is whatever self-coefficient SINDy finds
        # on drift[t] itself, not a separately hand-designed decay parameter.
        drift = self.state['drift'][..., 0:1]  # single accumulator: [W, E, B, 1]

        gain = torch.nn.functional.softplus(self.hazard_gain)
        threshold = torch.nn.functional.softplus(self.hazard_threshold)

        p_stop_up = torch.sigmoid(gain * (drift - threshold))
        p_stop_down = torch.sigmoid(gain * (-drift - threshold))

        # Gate the hazard so a decision can't be *reported* before non-decision
        # time has elapsed, even if evidence already crossed a boundary -- a
        # fixed stand-in for sensory/motor delay (ddm.py's `init_time`), set from
        # data via `estimate_non_decision_time()` rather than learned end-to-end.
        # Uses its own fixed, sharp steepness -- independent of the learned
        # `gain` -- so the cutoff stays a real cutoff even if `gain` shrinks.
        if self.non_decision_time > 0:
            time_axis = (torch.arange(1, self.max_steps + 1, device=self.device, dtype=drift.dtype) * self.dt).view(-1, 1, 1, 1)
            ndt_gate_steepness = 4. / self.dt
            report_gate = torch.sigmoid(ndt_gate_steepness * (time_axis - self.non_decision_time))
            p_stop_up = p_stop_up * report_gate
            p_stop_down = p_stop_down * report_gate

        p_stop = p_stop_up + p_stop_down - p_stop_up * p_stop_down

        survival = torch.cumprod(
            torch.cat((torch.ones_like(p_stop[:1]), 1. - p_stop[:-1]), dim=0),
            dim=0,
        )  # [W, E, B, 1]

        p_decision_up = p_stop_up * survival
        p_decision_down = p_stop_down * survival

        # Normalize (RTify's example2.ipynb convention): rescale proportionally so
        # up+down sums to 1, rather than dumping leftover (never-crossed) mass onto
        # the last bin -- avoids an artificial delta-spike at t_max.
        total = p_decision_up.sum(dim=0, keepdim=True) + p_decision_down.sum(dim=0, keepdim=True)
        p_decision_up = p_decision_up / (total + 1e-8)
        p_decision_down = p_decision_down / (total + 1e-8)

        # [W, E, B, 1] -> [E, B, 1(T), W, 2]: genuine per-timestep (up, down)
        # probabilities, no replication across a dummy axis -- O(max_steps) per
        # session, not O(max_steps^2). This also keeps `forward()` cheap enough
        # for the SINDy ridge-solve stage, which runs it over the whole dataset
        # flattened into one batch.
        output = torch.stack((p_decision_up, p_decision_down), dim=-1)  # [W, E, B, 1, 2]
        output = output.squeeze(-2).permute(1, 2, 0, 3).unsqueeze(2)  # [E, B, 1(T), W, 2]

        return output, self.get_state()


def make_ddm_loss():
    """Joint negative log-likelihood of (choice, RT) under the two-boundary hazard model.

    Both `prediction` and `target` are per-timestep: `prediction[..., w, :]` =
    [p_up[w], p_down[w]]; `target[..., w, :]` is a one-hot indicator that is 1 at
    exactly the (boundary, bin) pair actually observed for that trial, 0 elsewhere.
    Rows with no indicator (every `w` except the observed one) carry no loss --
    only the one row per trial matching the observed outcome contributes.
    """

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_obs = (prediction * target).sum(dim=-1)
        valid = target.sum(dim=-1) > 0.5
        return -torch.log(p_obs[valid].clamp_min(1e-8)).mean()

    return loss_fn


def decode_choice_rt(ys: torch.Tensor, dt: float) -> tuple:
    """Recover (is_up, rt_seconds) from the per-timestep one-hot indicator target."""
    bin_up = ys[..., 0].argmax(dim=-1)
    bin_down = ys[..., 1].argmax(dim=-1)
    is_up = ys[..., 0].sum(dim=-1) > 0.5
    rt_bin = torch.where(is_up, bin_up, bin_down)
    rt_seconds = (rt_bin.float() + 0.5) * dt
    return is_up, rt_seconds


def estimate_non_decision_time(rt_seconds: torch.Tensor, quantile: float = 0.05, safety_margin: float = 0.9) -> float:
    """Data-derived non-decision-time estimate: a low quantile of observed RTs.

    Standard practice (e.g. EZ-diffusion): the fastest responses are assumed to be
    close to pure sensory/motor delay, so a low percentile of the RT distribution
    is a reasonable fixed Ter estimate -- cheaper and more stable than learning it
    jointly with the evidence-accumulation dynamics.
    """
    return torch.quantile(rt_seconds.flatten(), quantile).item() * safety_margin


def simulate_ddm(
    n_trials: int = 1000,
    t_max: float = 5.0,
    max_steps: int = 100,
    drift_rate: float = 1.0,
    diffusion_rate: float = 1.0,
    threshold: float = 1.0,
    non_decision_time: float = 0.2,
    participant_id: int = 0,
    device=None,
) -> SpiceDataset:
    """Simulate a two-boundary DDM: `n_trials` sessions sharing one ground-truth condition.

    xs: (n_trials, 1, max_steps, 9) -- [action_0, action_1 (unused), stimulus,
        time_elapsed, time_trial (metadata, unused), trial, block, experiment, participant].
    ys: (n_trials, 1, max_steps, 2) -- one-hot indicator over (boundary, RT bin):
        ys[i, w, 0] = 1 iff trial i's observed decision is (up, bin w); ys[i, w, 1]
        likewise for "down". Exactly one of the `2 * max_steps` entries is 1 per trial.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dt = t_max / max_steps

    noise = torch.randn(n_trials, max_steps, device=device)
    increments = drift_rate * dt + diffusion_rate * (dt ** 0.5) * noise
    evidence = torch.cumsum(increments, dim=1)

    upper_mask = evidence >= threshold
    lower_mask = evidence <= -threshold
    has_upper = upper_mask.any(dim=1)
    has_lower = lower_mask.any(dim=1)
    first_upper = torch.where(has_upper, upper_mask.int().argmax(dim=1), torch.full_like(has_upper, max_steps, dtype=torch.long))
    first_lower = torch.where(has_lower, lower_mask.int().argmax(dim=1), torch.full_like(has_lower, max_steps, dtype=torch.long))

    decision_step = torch.minimum(first_upper, first_lower)
    # Trials that never cross a boundary within t_max are timeouts/non-responses --
    # drop them rather than fabricating a censored RT (forcing them into one bin
    # creates an artificial delta-spike in the "observed" distribution).
    responded = decision_step < max_steps
    n_valid = int(responded.sum().item())
    if n_valid < n_trials:
        print(f"simulate_ddm: dropped {n_trials - n_valid}/{n_trials} non-response (timeout) trials")

    choice = (first_upper <= first_lower).long()[responded]
    decision_step = decision_step[responded]
    response_time = decision_step.float() * dt + non_decision_time

    xs = torch.zeros(n_valid, max_steps, 9, device=device)
    xs[:, :, 2] = drift_rate
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


def simulate_ddm_flip(
    n_trials: int = 1000,
    t_max: float = 5.0,
    max_steps: int = 100,
    drift_rate: float = 1.0,
    diffusion_rate: float = 1.0,
    leak: float = 0.0,
    threshold: float = 1.0,
    non_decision_time: float = 0.2,
    flip_time_range: tuple = (0.3, 0.7),
    participant_id: int = 0,
    device=None,
) -> SpiceDataset:
    """Flipping-stimulus paradigm: ground truth is a leaky accumulator whose drift
    reverses sign at a random time (drawn per trial, as a fraction of `t_max`).

    dE/dt = -leak*E + drift(t) + diffusion*xi(t), where drift(t) = +drift_rate before
    the trial's flip time and -drift_rate after. A perfect (leak=0) integrator carries
    a large pre-flip evidence pileup that takes a long time to unwind after the flip;
    a leaky one forgets stale evidence faster and can change its mind sooner -- the
    classic motivation for leaky/competing accumulator models over perfect DDMs.

    The (flipping) drift value is also what's fed to the model as `stimulus` -- the
    participant directly observes the stimulus reversing, same as in a real experiment.

    xs/ys: same layout as `simulate_ddm`.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dt = t_max / max_steps

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

    upper_mask = evidence >= threshold
    lower_mask = evidence <= -threshold
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
        print(f"simulate_ddm_flip: dropped {n_trials - n_valid}/{n_trials} non-response (timeout) trials")

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

    xs = xs.unsqueeze(1)
    ys = ys.unsqueeze(1)

    return SpiceDataset(xs, ys, n_reward_features=0)


def _sanitize_predictions(p_up: torch.Tensor, p_down: torch.Tensor, label: str = '') -> tuple:
    """Drop trials with non-finite or degenerate (all-zero) predicted probabilities.

    A discovered SINDy equation is fit for one-step accuracy via ridge regression;
    nothing guarantees it stays bounded when iterated `max_steps` times (unlike the
    RNN, which was trained end-to-end and empirically stays well-behaved). If an
    equation diverges under rollout, `evidence` can hit +/-inf and the hazard math
    downstream produces NaN -- this excludes those trials rather than crashing
    (e.g. inside `torch.distributions.Categorical`) or silently corrupting stats.
    """
    total = p_up.sum(dim=-1) + p_down.sum(dim=-1)
    good = torch.isfinite(p_up).all(dim=-1) & torch.isfinite(p_down).all(dim=-1) & (total > 0)
    n_bad = int((~good).sum().item())
    if n_bad > 0:
        print(f"{label}: excluded {n_bad}/{p_up.shape[0]} trials with non-finite/degenerate probabilities "
              f"(likely SINDy rollout instability)")
    return p_up[good], p_down[good], good


@torch.no_grad()
def evaluate(estimator: SpiceEstimator, dataset: SpiceDataset, dt: float, max_steps: int) -> dict:
    estimator.model.eval()
    xs = dataset.xs.to(estimator.model.device)
    ys = dataset.ys.to(estimator.model.device)

    prediction, _ = estimator.model(xs)
    prediction = prediction.mean(dim=0)  # average over ensemble: [B, 1, W, 2]

    p_up = prediction[:, 0, :, 0]
    p_down = prediction[:, 0, :, 1]
    is_up, rt_obs = decode_choice_rt(ys[:, 0], dt)
    p_up, p_down, good = _sanitize_predictions(p_up, p_down, label='evaluate')
    is_up, rt_obs = is_up[good], rt_obs[good]

    pred_choice = p_up.sum(dim=-1) > p_down.sum(dim=-1)
    accuracy = (pred_choice == is_up).float().mean().item()

    bins = torch.arange(max_steps, device=p_up.device, dtype=p_up.dtype) * dt + dt / 2
    rt_pred_mean = ((p_up + p_down) * bins).sum(dim=-1).mean().item()
    rt_obs_mean = rt_obs.mean().item()

    return {'choice_accuracy': accuracy, 'rt_pred_mean': rt_pred_mean, 'rt_obs_mean': rt_obs_mean}


@torch.no_grad()
def plot_rt_distribution(
    estimator: SpiceEstimator,
    dataset: SpiceDataset,
    dt: float,
    max_steps: int,
    use_sindy: bool = False,
    output_path: str = None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    estimator.model.eval()
    prev_use_sindy = estimator.model.use_sindy
    estimator.use_sindy(use_sindy)

    xs = dataset.xs.to(estimator.model.device)
    ys = dataset.ys.to(estimator.model.device)

    prediction, _ = estimator.model(xs)
    estimator.use_sindy(prev_use_sindy)
    prediction = prediction.mean(dim=0)  # [B, 1, W, 2]

    p_up = prediction[:, 0, :, 0]
    p_down = prediction[:, 0, :, 1]
    is_up_obs, rt_obs = decode_choice_rt(ys[:, 0], dt)
    p_up, p_down, good = _sanitize_predictions(p_up, p_down, label='plot_rt_distribution')
    is_up_obs, rt_obs = is_up_obs[good], rt_obs[good]
    p_decision = p_up + p_down

    signed_rt_obs = torch.where(is_up_obs, rt_obs, -rt_obs).cpu().numpy()

    sampled_bin = torch.distributions.Categorical(probs=p_decision.clamp_min(1e-8)).sample()
    is_up_pred = torch.rand(p_decision.shape[0], device=p_decision.device) < (p_up.sum(dim=-1) / p_decision.sum(dim=-1).clamp_min(1e-8))
    rt_pred = (sampled_bin.float() + 0.5) * dt
    signed_rt_pred = torch.where(is_up_pred, rt_pred, -rt_pred).cpu().numpy()

    bins = np.linspace(-max_steps * dt, max_steps * dt, 2 * max_steps + 1)
    model_label = 'SPICE (SINDy, sampled)' if use_sindy else 'Model (RNN, sampled)'

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(signed_rt_obs, bins=50, alpha=0.5, density=True, label='Observed', color='tab:blue')
    ax.hist(signed_rt_pred, bins=50, alpha=0.5, density=True, label=model_label, color='tab:orange')
    ax.set_xlabel('Signed RT (s); sign = boundary')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    plt.show()

    return fig


def print_spice_models(estimator: SpiceEstimator, participant_ids=(0, 1)):
    for pid in participant_ids:
        print(f"\n--- Participant {pid} ---")
        estimator.print_spice_model(participant_id=pid)


@torch.no_grad()
def plot_drift_traces(
    estimator: SpiceEstimator,
    dataset: SpiceDataset,
    dt: float,
    participant_ids=(0, 1),
    n_examples: int = 3,
    use_sindy: bool = False,
    output_path: str = None,
):
    """Example raw `drift` trajectories (the module's own state, before integration
    into evidence) for a few individual trials per participant, overlaid with the
    true generative drift/stimulus (dashed) for the same trials. In the flip
    paradigm each trial has its own random flip time, so -- unlike the integrated
    evidence -- this shows directly whether `drift` tracks the stimulus reversal,
    and at what (per-trial-varying) point each example trace actually flips.
    """
    import matplotlib.pyplot as plt

    estimator.model.eval()
    prev_use_sindy = estimator.model.use_sindy
    estimator.use_sindy(use_sindy)

    xs = dataset.xs.to(estimator.model.device)
    participant_col = xs[:, 0, 0, -1]
    ground_truth_drift = xs[:, 0, :, 2].transpose(0, 1)  # [W, B] -- true (flipping) stimulus

    _, state = estimator.model(xs)
    estimator.use_sindy(prev_use_sindy)

    drift = state['drift'][..., 0:1].mean(dim=1).squeeze(-1)  # mean over ensemble: [W, B]
    time_axis = (torch.arange(1, drift.shape[0] + 1, device=drift.device) * dt).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10.colors
    for i, pid in enumerate(participant_ids):
        idx = (participant_col == pid).nonzero(as_tuple=True)[0][:n_examples]
        for j, trial_idx in enumerate(idx):
            ax.plot(
                time_axis,
                drift[:, trial_idx].cpu().numpy(),
                color=colors[i % len(colors)],
                alpha=0.7,
                label=f'participant {pid} (fit)' if j == 0 else None,
            )
            ax.plot(
                time_axis,
                ground_truth_drift[:, trial_idx].cpu().numpy(),
                color=colors[i % len(colors)],
                linestyle='--',
                alpha=0.4,
                label=f'participant {pid} (true)' if j == 0 else None,
            )

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drift')
    ax.set_title('SPICE (SINDy)' if use_sindy else 'RNN')
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    plt.show()

    return fig


if __name__ == '__main__':

    path_spice = 'weinhardt2026/studies/archive/rtify2024/params/rtify2024.pkl'

    max_steps = 60
    t_max = 3.0
    dt = t_max / max_steps
    n_trials = 1000

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Flipping-stimulus paradigm: drift_rates are magnitudes now -- the flip mechanic
    # supplies the sign reversal, so participants differ in stimulus strength, not sign.
    # Kept low relative to threshold/flip_time_range so the flip actually matters:
    # stronger rates (e.g. 1.5-2.0) resolve the decision before the earliest possible
    # flip, making the "down" boundary almost unreachable regardless of the leak.
    drift_rates = [0.1, 0.2, 0.3, 0.5]
    n_participants = len(drift_rates)
    ground_truth_leak = 1.0  # shared across participants; recovered via `estimator.model.leak_raw`

    datasets = [
        simulate_ddm_flip(
            n_trials=n_trials,
            t_max=t_max,
            max_steps=max_steps,
            drift_rate=rate,
            diffusion_rate=1.0,
            leak=ground_truth_leak,
            threshold=1.0,
            non_decision_time=0.2,
            flip_time_range=(0.3, 0.7),
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
    print(f"Estimated non-decision time: {non_decision_time:.3f}s")

    estimator = SpiceEstimator(
        spice_class=DDMRNN,
        spice_config=CONFIG,
        kwargs_spice_class={'max_steps': max_steps, 't_max': t_max, 'non_decision_time': non_decision_time},
        n_reward_features=0,

        n_actions=2,
        n_participants=n_participants,

        loss_fn=make_ddm_loss(),
        loss_fn_kwargs={},

        sindy_weight=0.01,
        sindy_refit=False,

        epochs=0,
        warmup_steps=500,
        
        device=device,
        verbose=True,
        save_path_spice=path_spice,
    )

    if estimator.epochs == 0:
        estimator.load_spice(path_spice)
    if estimator.epochs > 0 or estimator.sindy_refit:
        estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
        estimator.save_spice(path_spice)
    
    print("\n--- Train ---")
    print(evaluate(estimator, dataset_train, dt, max_steps))
    print("\n--- Test ---")
    print(evaluate(estimator, dataset_test, dt, max_steps))

    recovered_leak = torch.nn.functional.softplus(estimator.model.leak_raw).item()
    print(f"\nGround-truth leak: {ground_truth_leak}, recovered leak: {recovered_leak:.3f}")

    print_spice_models(estimator, participant_ids=(0, 2))

    plot_rt_distribution(estimator, dataset_test, dt, max_steps, use_sindy=False)
    plot_rt_distribution(estimator, dataset_test, dt, max_steps, use_sindy=True)

    plot_drift_traces(estimator, dataset_test, dt, participant_ids=(0, 2), use_sindy=False)
    plot_drift_traces(estimator, dataset_test, dt, participant_ids=(0, 2), use_sindy=True)
