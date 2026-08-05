import torch

from spice import SpiceEstimator, SpiceDataset


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
    use_sindy: bool = False,
    output_path: str = None,
):
    """One participant: observed RT histogram (signed, density-normalized) vs.
    the model's own predicted per-bin likelihood, overlaid as a red line --
    up on the positive side, down on the negative side, mirroring the
    histogram's sign convention.

    `drift`/`evidence` are deterministic given a participant's (constant)
    stimulus, so every trial belonging to `participant_id` shares the exact
    same predicted p_decision_up/down(t) trajectory -- one trial's prediction
    *is* the model's claimed density for that participant, directly
    comparable to the aggregate empirical histogram across all of that
    participant's trials. This is a more direct fit check than plot_summary's
    RT subplot, which re-samples from the predicted distribution (adding
    sampling noise) rather than plotting the likelihood itself.

    `prediction[trial, 0, :, 0/1]` is a probability MASS per discrete time
    bin (summing to 1 over the trajectory, per post_forward_pass's
    normalization) -- dividing by `dt` converts it to a density comparable to
    matplotlib's `density=True` histogram scale (mass/bin_width).
    """
    import matplotlib.pyplot as plt

    estimator.model.eval()
    xs = dataset.xs.to(estimator.model.device)
    ys = dataset.ys.to(estimator.model.device)
    prev_use_sindy = estimator.model.use_sindy
    estimator.use_sindy(use_sindy)

    prediction, _ = estimator.model(xs)
    prediction = prediction.mean(dim=0)  # [B, 1, W, 2]
    estimator.use_sindy(prev_use_sindy)

    participant_col = xs[:, 0, 0, -1]
    idx = (participant_col == participant_id).nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        raise ValueError(f"No trials found for participant {participant_id}")

    is_up_obs, rt_obs = decode_choice_rt(ys[idx][:, 0], dt)
    signed_rt_obs = torch.where(is_up_obs, rt_obs, -rt_obs).cpu().numpy()

    # Every trial for this participant shares the same deterministic
    # prediction -- any single one (here, the first) is representative.
    trial_idx = idx[0]
    max_steps = prediction.shape[2]
    p_up = prediction[trial_idx, 0, :, 0].cpu().numpy()
    p_down = prediction[trial_idx, 0, :, 1].cpu().numpy()

    time_axis = (torch.arange(max_steps, dtype=torch.float32) * dt + dt / 2).numpy()

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
def plot_summary(
    estimator: SpiceEstimator,
    dataset: SpiceDataset,
    dt: float,
    t_max: float,
    max_steps: int,
    participant_ids=(0, 1),
    n_examples: int = 3,
    true_threshold: float = None,
    output_path: str = None,
):
    """One figure, four subplots: drift (rate), evidence (accumulator), decision
    boundary (+/-threshold), RT distribution. Drift/evidence/boundary subplots:
    one color per participant, one linestyle per role (true=dotted, rnn=dashed,
    sindy=solid). RT subplot: one color per role (true=blue, rnn=orange,
    sindy=red), pooled across participants.

    `true_threshold`: the constant boundary value passed to `simulate_ddm` (only
    meaningful if `collapsing_bound_rate=0` there -- pass None to skip the true
    boundary reference line if the ground truth is actually collapsing).
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import numpy as np

    estimator.model.eval()
    xs = dataset.xs.to(estimator.model.device)
    ys = dataset.ys.to(estimator.model.device)
    prev_use_sindy = estimator.model.use_sindy

    colors = {'true': 'tab:blue', 'rnn': 'tab:orange', 'sindy': 'tab:red'}
    linestyles = {'true': ':', 'rnn': '--', 'sindy': '-'}
    participant_colors = plt.cm.tab10.colors

    # --- run the model once in RNN mode, once in SINDy mode ---
    predictions, drifts, evidences, thresholds = {}, {}, {}, {}
    has_boundary_module = False
    for key, use_sindy in (('rnn', False), ('sindy', True)):
        estimator.use_sindy(use_sindy)
        prediction, state = estimator.model(xs)
        predictions[key] = prediction.mean(dim=0)  # [B, 1, W, 2]
        drifts[key] = state['drift'][..., 0:1].mean(dim=1).squeeze(-1)  # [max_steps, B]
        evidences[key] = state['evidence'][..., 0:1].mean(dim=1).squeeze(-1)  # [max_steps, B]
        if 'threshold_raw' in state:
            has_boundary_module = True
            thresholds[key] = torch.nn.functional.softplus(state['threshold_raw'][..., 0:1]).mean(dim=1).squeeze(-1)
    estimator.use_sindy(prev_use_sindy)

    fig, (ax_drift, ax_evidence, ax_boundary, ax_rt) = plt.subplots(1, 4, figsize=(22, 4))

    participant_col = xs[:, 0, 0, -1]
    ground_truth_drift = xs[:, 0, :, 2].transpose(0, 1)  # [W, B] -- true (possibly flipping) stimulus

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

    # --- evidence: the accumulator. No ground-truth line -- simulate_ddm
    # doesn't store the true simulated evidence trajectory, only the
    # observed choice/RT and the (possibly flipping) stimulus/drift signal
    # already shown in the drift subplot. ---
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

    # Two-part legend (color -> participant, linestyle -> role) shown once, on the
    # drift subplot; role legend uses black proxy lines since linestyle, not
    # color, carries the meaning there. Same convention applies to both subplots.
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
    is_up_obs, rt_obs = decode_choice_rt(ys[:, 0], dt)
    signed_rt_obs = None
    bins = np.linspace(-max_steps * dt, max_steps * dt, 2 * max_steps + 1)

    if estimator.sindy_weight == 0 and not estimator.sindy_refit:
        spice_models = ('rnn',)
    else:
        spice_models = ('rnn', 'sindy')
        
    for key in spice_models:
        p_up = predictions[key][:, 0, :, 0]
        p_down = predictions[key][:, 0, :, 1]
        is_up, rt = is_up_obs, rt_obs
        p_up, p_down, good = _sanitize_predictions(p_up, p_down, label=f'plot_summary[{key}]')
        is_up, rt = is_up[good], rt[good]
        p_decision = p_up + p_down

        if signed_rt_obs is None:
            signed_rt_obs = torch.where(is_up, rt, -rt).cpu().numpy()

        sampled_bin = torch.distributions.Categorical(probs=p_decision.clamp_min(1e-8)).sample()
        is_up_pred = torch.rand(p_decision.shape[0], device=p_decision.device) < (p_up.sum(dim=-1) / p_decision.sum(dim=-1).clamp_min(1e-8))
        rt_pred = (sampled_bin.float() + 0.5) * dt
        signed_rt_pred = torch.where(is_up_pred, rt_pred, -rt_pred).cpu().numpy()

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