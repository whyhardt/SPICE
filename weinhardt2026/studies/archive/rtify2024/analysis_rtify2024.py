import torch

from spice import SpiceEstimator, SpiceDataset


def decode_choice_rt(ys: torch.Tensor, dt: float) -> tuple:
    """Recover (is_up, rt_seconds) from the per-trial per-step one-hot target.

    `ys`: (..., T, 3) -- [no_decision, up, down] per step; the first step where column 1 or 2
    is set is the trial's decision step (steps after are masked-out zeros, per
    benchmark_rtify2024_f.py). Returns the RAW (non_decision_time-free) decision time --
    callers add non_decision_time themselves when they need a real-world-comparable RT (see
    module docstring in benchmark_rtify2024_f.py for why that split exists).
    """
    is_decision = (ys[..., 1] + ys[..., 2]) > 0.5  # [..., T]
    bin_idx = is_decision.float().argmax(dim=-1)  # first True index
    is_up = ys[..., 1].gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1) > 0.5
    rt_seconds = (bin_idx.float() + 0.5) * dt
    return is_up, rt_seconds


def _step_probs_to_marginal(logits: torch.Tensor) -> tuple:
    """[..., T, 3] raw per-step [no_decision, up, down] logits -> per-step CONDITIONAL
    probabilities (softmax) -> marginal (survival-weighted) p_up/p_down, purely for
    post-hoc reporting/plotting (not used anywhere in training).

    survival[t] = P(not yet decided before step t) = cumulative product of
    p_no_decision_cond over steps < t (1.0 at t=0, nothing decided yet).
    """
    step_probs = torch.softmax(logits, dim=-1)
    p_no_decision_cond = step_probs[..., 0]
    p_up_cond = step_probs[..., 1]
    p_down_cond = step_probs[..., 2]

    ones = torch.ones_like(p_no_decision_cond[..., :1])
    survival = torch.cumprod(torch.cat((ones, p_no_decision_cond[..., :-1]), dim=-1), dim=-1)

    return survival * p_up_cond, survival * p_down_cond


@torch.no_grad()
def evaluate(estimator: SpiceEstimator, dataset: SpiceDataset, dt: float, max_steps: int, non_decision_time: float = 0.0) -> dict:
    estimator.model.eval()
    xs = dataset.xs.to(estimator.model.device)
    ys = dataset.ys.to(estimator.model.device)

    logits, _ = estimator.model(xs)
    logits = logits.mean(dim=0)[:, :, 0, :]  # average over ensemble, drop W=1: [B, T, 3]

    p_up, p_down = _step_probs_to_marginal(logits)

    is_up_obs, rt_obs_raw = decode_choice_rt(ys[:, :, 0], dt)
    rt_obs = rt_obs_raw + non_decision_time

    pred_choice = p_up.sum(dim=-1) > p_down.sum(dim=-1)
    accuracy = (pred_choice == is_up_obs).float().mean().item()

    bins = torch.arange(max_steps, device=p_up.device, dtype=p_up.dtype) * dt + dt / 2 + non_decision_time
    rt_pred_mean = ((p_up + p_down) * bins).sum(dim=-1).mean().item()
    rt_obs_mean = rt_obs.mean().item()

    return {'choice_accuracy': accuracy, 'rt_pred_mean': rt_pred_mean, 'rt_obs_mean': rt_obs_mean}


def print_spice_models(estimator: SpiceEstimator, participant_ids=(0, 1)):
    for pid in participant_ids:
        print(f"\n--- Participant {pid} ---")
        estimator.print_spice_model(participant_id=pid)


@torch.no_grad()
def plot_participant_fit(
    estimator: SpiceEstimator,
    dataset: SpiceDataset,
    dt: float,
    t_max: float,
    participant_id: int,
    non_decision_time: float = 0.0,
    use_sindy: bool = False,
    output_path: str = None,
):
    """One participant: observed RT histogram (signed, density-normalized) vs. the model's own
    predicted per-bin likelihood, overlaid as a red line -- up on the positive side, down on
    the negative side. The likelihood is the analytic marginal (survival-weighted per-step
    conditional probabilities, see `_step_probs_to_marginal`) -- exact, no sampling -- so this
    is a more direct fit check than a histogram of simulated trajectories.
    """
    import matplotlib.pyplot as plt

    estimator.model.eval()
    xs = dataset.xs.to(estimator.model.device)
    ys = dataset.ys.to(estimator.model.device)
    prev_use_sindy = estimator.model.use_sindy
    estimator.use_sindy(use_sindy)

    logits, _ = estimator.model(xs)
    logits = logits.mean(dim=0)[:, :, 0, :]  # [B, T, 3]
    estimator.use_sindy(prev_use_sindy)

    participant_col = xs[:, 0, 0, -1]
    idx = (participant_col == participant_id).nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        raise ValueError(f"No trials found for participant {participant_id}")

    is_up_obs, rt_obs_raw = decode_choice_rt(ys[idx][:, :, 0], dt)
    rt_obs = rt_obs_raw + non_decision_time
    signed_rt_obs = torch.where(is_up_obs, rt_obs, -rt_obs).cpu().numpy()

    # `drift`/`evidence` are deterministic given a participant's (constant) stimulus, so every
    # trial belonging to `participant_id` shares the exact same predicted trajectory.
    trial_idx = idx[0]
    max_steps = logits.shape[1]
    p_up, p_down = _step_probs_to_marginal(logits[trial_idx])
    p_up, p_down = p_up.cpu().numpy(), p_down.cpu().numpy()

    time_axis = (torch.arange(max_steps, dtype=torch.float32) * dt + dt / 2).numpy() + non_decision_time

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(signed_rt_obs, bins=50, range=(-t_max, t_max), density=True, alpha=0.5, color='tab:blue', label='observed RTs')
    ax.plot(time_axis, p_up / dt, color='red', label='model likelihood (up/down)')
    ax.plot(-time_axis, p_down / dt, color='red')
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Signed RT (s); sign = boundary')
    ax.set_ylabel('Density')
    ax.set_title(f'Participant {participant_id} ({"sindy" if use_sindy else "rnn"})')
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    plt.show()

    return fig


@torch.no_grad()
def _rollout_state_trajectory(model, xs: torch.Tensor) -> dict:
    """Recover the full per-trial drift/evidence/threshold/logits history by calling `model()`
    once per trial-step and reading `get_state()` (plus the step's own return value) after
    each call -- a single batched `model(xs)` call only returns the LAST step's state snapshot
    (`self.state` is overwritten every iteration of `forward()`'s trial loop). State carries
    correctly across calls via `prev_state`, so rolling step-by-step and concatenating
    reconstructs the exact same per-step values a single within-forward() loop would see.

    `logits` is captured straight from each step's own model() return value -- NOT re-derived
    from evidence/threshold via a hand-rolled formula in analysis code, which is what
    previously went silently stale (see _simulate_trial_by_trial's docstring).

    rtify2024_f's DDMRNN has no scalar `evidence`/softplus(`threshold_raw`) state -- `evidence`
    is a full probability density over a grid (`state['evidence_pdf']`), and `threshold` is
    bounded via sigmoid, not softplus. `model.mean_evidence()`/`model.effective_threshold()`
    read the model's own definitions of those (its density's mean, its actual bounded
    threshold) rather than re-deriving them here -- same anti-staleness reasoning as `logits`.
    """
    T = xs.shape[1]
    state = None
    history = {'drift': [], 'evidence': [], 'threshold': [], 'logits': []}
    for t in range(T):
        logits_t, state = model(xs[:, t:t + 1], state)
        history['drift'].append(state['drift'][..., 0:1])
        history['evidence'].append(model.mean_evidence().unsqueeze(-1))
        history['threshold'].append(model.effective_threshold().unsqueeze(-1))
        history['logits'].append(logits_t[:, :, 0, 0, :])  # [E, B, 3] -- this step's T=1, W=1 squeezed out

    trajectory = {key: torch.cat(values, dim=0) for key, values in history.items() if key != 'logits'}
    trajectory['logits'] = torch.stack(history['logits'], dim=0)  # [T, E, B, 3]
    return trajectory


@torch.no_grad()
def _simulate_trial_by_trial(logits: torch.Tensor) -> tuple:
    """RL-agent-style stepwise simulation: at each step, sample one of [no_decision, up, down]
    from the model's own per-step conditional probabilities -- `no_decision` continues the
    episode, `up`/`down` ends it. This replays forward()'s own generative process directly.

    `logits`: [max_steps, B, 3] -- raw per-step logits straight from the model's own forward()
    (via _rollout_state_trajectory), softmaxed here to get probabilities. Deliberately NOT
    re-derived from evidence/threshold via a hand-rolled formula: an earlier version of this
    function hardcoded rtify2024_d's softmax([threshold, evidence, -evidence]) hazard, which
    silently went stale the moment spice_rtify2024_f.py switched to a Gaussian-CDF hazard --
    the simulated RT distribution quietly kept sampling from the wrong (old) formula while
    training (which reads the model's actual logits) was fine. Reading logits straight from
    the model avoids that class of bug regardless of what the hazard formula is.

    Returns (decided_bin, decided_up, decided): all [B]. `decided` is False for trials that
    never leave `no_decision` within the horizon -- mirrors simulate_ddm's own handling of
    timeouts (dropped, not force-assigned).
    """
    max_steps, n_trials, _ = logits.shape
    probs = torch.softmax(logits, dim=-1)

    decided_bin = torch.zeros(n_trials, dtype=torch.long, device=logits.device)
    decided_up = torch.zeros(n_trials, dtype=torch.bool, device=logits.device)
    decided = torch.zeros(n_trials, dtype=torch.bool, device=logits.device)

    for t in range(max_steps):
        choice = torch.multinomial(probs[t], num_samples=1).squeeze(-1)  # 0=no_decision, 1=up, 2=down
        stop_now = ~decided & (choice != 0)
        decided_bin[stop_now] = t
        decided_up[stop_now] = choice[stop_now] == 1
        decided = decided | stop_now

    return decided_bin, decided_up, decided


@torch.no_grad()
def plot_summary(
    estimator: SpiceEstimator,
    dataset: SpiceDataset,
    dt: float,
    t_max: float,
    max_steps: int,
    participant_ids=(0, 1),
    n_examples: int = 3,
    true_threshold: float = None,
    non_decision_time: float = 0.0,
    output_path: str = None,
):
    """One figure, four subplots: drift (rate), evidence (accumulator), decision
    boundary (+/-threshold), RT distribution. Drift/evidence/boundary subplots:
    one color per participant, one linestyle per role (true=dotted, rnn=dashed,
    sindy=solid). RT subplot: one color per role (true=blue, rnn=orange,
    sindy=red), pooled across participants, sampled trial-by-trial via
    `_simulate_trial_by_trial`.
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    estimator.model.eval()
    xs = dataset.xs.to(estimator.model.device)
    ys = dataset.ys.to(estimator.model.device)
    prev_use_sindy = estimator.model.use_sindy

    colors = {'true': 'tab:blue', 'rnn': 'tab:orange', 'sindy': 'tab:red'}
    linestyles = {'true': ':', 'rnn': '--', 'sindy': '-'}
    participant_colors = plt.cm.tab10.colors

    # --- run the model once in RNN mode, once in SINDy mode ---
    drifts, evidences, thresholds, logits_traj = {}, {}, {}, {}
    has_boundary_module = True  # CONFIG always has a threshold_raw module in this study
    for key, use_sindy in (('rnn', False), ('sindy', True)):
        estimator.use_sindy(use_sindy)
        trajectory = _rollout_state_trajectory(estimator.model, xs)
        drifts[key] = trajectory['drift'][..., 0].mean(dim=1)      # [max_steps, B]
        evidences[key] = trajectory['evidence'][..., 0].mean(dim=1)  # [max_steps, B]
        thresholds[key] = trajectory['threshold'][..., 0].mean(dim=1)  # [max_steps, B]
        logits_traj[key] = trajectory['logits'].mean(dim=1)  # [max_steps, B, 3]
    estimator.use_sindy(prev_use_sindy)

    fig, (ax_drift, ax_evidence, ax_boundary, ax_rt) = plt.subplots(1, 4, figsize=(22, 4))

    participant_col = xs[:, 0, 0, -1]
    ground_truth_drift = xs[:, :, 0, 3].transpose(0, 1)  # [T, B] -- true (possibly flipping) stimulus

    # --- drift: example rate traces (color = participant, linestyle = role) ---
    for i, pid in enumerate(participant_ids):
        pcolor = participant_colors[i % len(participant_colors)]
        idx = (participant_col == pid).nonzero(as_tuple=True)[0][:n_examples]
        for trial_idx in idx:
            time_axis_true = (torch.arange(1, ground_truth_drift.shape[0] + 1, device=xs.device) * dt).cpu().numpy()
            ax_drift.plot(
                time_axis_true, ground_truth_drift[:, trial_idx].cpu().numpy(),
                color=pcolor, linestyle=linestyles['true'], alpha=0.7,
            )
            for key in ('rnn', 'sindy'):
                d = drifts[key][:, trial_idx].cpu().numpy()
                time_axis = (torch.arange(1, len(d) + 1, device=xs.device) * dt).cpu().numpy()
                ax_drift.plot(time_axis, d, color=pcolor, linestyle=linestyles[key], alpha=0.7)

    ax_drift.axhline(0, color='gray', linewidth=0.5)
    ax_drift.set_xlabel('Time (s)')
    ax_drift.set_ylabel('Drift (rate)')

    # --- evidence: the accumulator. No ground-truth line -- simulate_ddm doesn't store the
    # true simulated evidence trajectory, only the observed choice/RT and the stimulus. ---
    for i, pid in enumerate(participant_ids):
        pcolor = participant_colors[i % len(participant_colors)]
        idx = (participant_col == pid).nonzero(as_tuple=True)[0][:n_examples]
        for trial_idx in idx:
            for key in ('rnn', 'sindy'):
                e = evidences[key][:, trial_idx].cpu().numpy()
                time_axis = (torch.arange(1, len(e) + 1, device=xs.device) * dt).cpu().numpy()
                ax_evidence.plot(time_axis, e, color=pcolor, linestyle=linestyles[key], alpha=0.7)

    ax_evidence.axhline(0, color='gray', linewidth=0.5)
    ax_evidence.set_xlabel('Time (s)')
    ax_evidence.set_ylabel('Evidence (accumulator)')

    # --- decision boundary (+/-threshold), same color/linestyle convention ---
    if has_boundary_module:
        for i, pid in enumerate(participant_ids):
            pcolor = participant_colors[i % len(participant_colors)]
            idx = (participant_col == pid).nonzero(as_tuple=True)[0][:n_examples]
            for trial_idx in idx:
                for key in ('rnn', 'sindy'):
                    th = thresholds[key][:, trial_idx].cpu().numpy()
                    time_axis = (torch.arange(1, len(th) + 1, device=xs.device) * dt).cpu().numpy()
                    ax_boundary.plot(time_axis, th, color=pcolor, linestyle=linestyles[key], alpha=0.7)
                    ax_boundary.plot(time_axis, -th, color=pcolor, linestyle=linestyles[key], alpha=0.7)

    if true_threshold is not None:
        ax_boundary.axhline(true_threshold, color=colors['true'], linestyle=linestyles['true'], alpha=0.7)
        ax_boundary.axhline(-true_threshold, color=colors['true'], linestyle=linestyles['true'], alpha=0.7)

    ax_boundary.axhline(0, color='gray', linewidth=0.5)
    ax_boundary.set_xlabel('Time (s)')
    ax_boundary.set_ylabel('Boundary (+/-threshold)')

    participant_handles = [
        mlines.Line2D([], [], color=participant_colors[i % len(participant_colors)], label=f'participant {pid}')
        for i, pid in enumerate(participant_ids)
    ]
    role_handles = [
        mlines.Line2D([], [], color='black', linestyle=linestyles[key], label=key)
        for key in ('true', 'rnn', 'sindy')
    ]
    ax_drift.legend(handles=participant_handles + role_handles, fontsize='small')

    # --- right: RT distribution ---
    is_up_obs, rt_obs_raw = decode_choice_rt(ys[:, :, 0], dt)
    signed_rt_obs = torch.where(is_up_obs, rt_obs_raw + non_decision_time, -(rt_obs_raw + non_decision_time)).cpu().numpy()

    if estimator.sindy_weight == 0 and not estimator.sindy_refit:
        spice_models = ('rnn',)
    else:
        spice_models = ('rnn', 'sindy')

    for key in spice_models:
        logits_key = logits_traj[key]
        finite = torch.isfinite(logits_key).all(dim=(0, 2))
        n_bad = int((~finite).sum().item())
        if n_bad > 0:
            print(f"plot_summary[{key}]: excluded {n_bad}/{finite.shape[0]} trials with non-finite "
                  f"logits trajectory (likely SINDy rollout instability)")

        decided_bin, decided_up, decided = _simulate_trial_by_trial(logits_key[:, finite])
        n_timeout = int((~decided).sum().item())
        if n_timeout > 0:
            print(f"plot_summary[{key}]: dropped {n_timeout}/{decided.shape[0]} trials with no decision "
                  f"within the {max_steps}-step horizon (timeout)")

        rt_pred = (decided_bin[decided].float() + 0.5) * dt + non_decision_time
        signed_rt_pred = torch.where(decided_up[decided], rt_pred, -rt_pred).cpu().numpy()

        ax_rt.hist(signed_rt_pred, bins=50, range=(-t_max, t_max), alpha=0.5, density=True, label=key, color=colors[key])

    ax_rt.hist(signed_rt_obs, bins=50, range=(-t_max, t_max), alpha=0.5, density=True, label='true', color=colors['true'])
    ax_rt.set_xlabel('Signed RT (s); sign = boundary')
    ax_rt.set_ylabel('Density')
    ax_rt.legend()

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    plt.show()

    return fig
