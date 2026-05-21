import torch
import numpy as np
import pandas as pd

from spice import SpiceEstimator, SpiceDataset


@torch.no_grad()
def analysis_sindy_onestepahead(
    dataset: SpiceDataset,
    spice_model: SpiceEstimator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Diagnose whether SINDy's test-time performance gap is due to per-step
    approximation error or autoregressive error accumulation.

    Compares three evaluation modes:
      SPICE-RNN               — RNN autoregressive (ensemble mean)
      SPICE (autoregressive)  — standard SINDy evaluation (member 0)
      SPICE (one-step-ahead)  — SINDy receives RNN's state at each trial

    Interpretation:
      one-step-ahead ≈ RNN  but  autoregressive ≪ RNN  → error accumulation
      one-step-ahead ≪ RNN                              → poor per-step approximation

    Args:
        dataset: SpiceDataset with xs (B, T, W, F) and ys (B, T, W, A).
        spice_model: Fitted SpiceEstimator with SINDy coefficients.

    Returns:
        summary: Aggregate Trial Likelihood and NLL per mode.
        per_trial: Per-trial-position Trial Likelihood for plotting.
    """
    model = spice_model.model
    device = model.device
    xs = dataset.xs.to(device)
    ys = dataset.ys.cpu()

    B, T, W, F = xs.shape

    # Valid trial mask: (B, T) — True where data is not NaN-padded
    valid = ~torch.isnan(dataset.xs[:, :, 0, 0].cpu())

    # ------------------------------------------------------------------
    # 1. SPICE-RNN (autoregressive, ensemble mean)
    # ------------------------------------------------------------------
    print("  Computing SPICE-RNN predictions...")
    model.eval(use_sindy=False)
    logits_rnn, _ = model(xs)
    # (E, B, T, W, A) → ensemble mean → (B, T, W, A)
    probs_rnn = torch.softmax(logits_rnn.mean(dim=0), dim=-1).cpu()

    # ------------------------------------------------------------------
    # 2. SPICE SINDy autoregressive (member 0)
    # ------------------------------------------------------------------
    print("  Computing SPICE (autoregressive) predictions...")
    model.eval(use_sindy=True)
    logits_sindy_ar, _ = model(xs)
    # (E, B, T, W, A) → member 0 → (B, T, W, A)
    probs_sindy_ar = torch.softmax(logits_sindy_ar[0], dim=-1).cpu()

    # ------------------------------------------------------------------
    # 3. SPICE SINDy one-step-ahead (teacher-forced with RNN states)
    #    At each trial, SINDy receives the RNN's actual pre-trial state
    #    instead of its own previous predictions.
    # ------------------------------------------------------------------
    print("  Computing SPICE (one-step-ahead) predictions...")
    logits_osa_parts = []
    rnn_state = None

    for t in range(T):
        xs_t = xs[:, t:t+1]  # (B, 1, W, F)

        # Clone the RNN state before this trial
        pre_state = (
            {k: v.clone() for k, v in rnn_state.items()}
            if rnn_state is not None else None
        )

        # Advance RNN by one trial (produces correct state trajectory)
        model.eval(use_sindy=False)
        _, rnn_state = model(xs_t, rnn_state)

        # Run SINDy for this one trial starting from the RNN's pre-trial state
        model.eval(use_sindy=True)
        logits_t, _ = model(xs_t, pre_state)
        # (E, B, 1, W, A) → member 0 → (B, 1, W, A)
        logits_osa_parts.append(logits_t[0].cpu())

    probs_sindy_osa = torch.softmax(
        torch.cat(logits_osa_parts, dim=1), dim=-1,
    )  # (B, T, W, A)

    # Restore model to SINDy eval mode
    model.eval(use_sindy=True)

    # ------------------------------------------------------------------
    # Per-trial log-likelihood: (B, T)
    # ------------------------------------------------------------------
    def _trial_ll(probs):
        eps = 1e-9
        probs = probs.clamp(eps, 1 - eps)
        # sum over actions (dim=-1), then over within-trial steps (dim=-1)
        return (ys * torch.log(probs)).sum(dim=-1).sum(dim=-1)

    ll_rnn = _trial_ll(probs_rnn).where(valid, torch.tensor(float('nan')))
    ll_ar = _trial_ll(probs_sindy_ar).where(valid, torch.tensor(float('nan')))
    ll_osa = _trial_ll(probs_sindy_osa).where(valid, torch.tensor(float('nan')))

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    n_valid = valid.sum().item()

    nll_rnn = -torch.nansum(ll_rnn).item()
    nll_ar = -torch.nansum(ll_ar).item()
    nll_osa = -torch.nansum(ll_osa).item()

    summary = pd.DataFrame({
        'Trial Lik.': [
            np.exp(-nll_rnn / n_valid),
            np.exp(-nll_ar / n_valid),
            np.exp(-nll_osa / n_valid),
        ],
        'NLL': [nll_rnn, nll_ar, nll_osa],
    }, index=['SPICE-RNN', 'SPICE (autoregressive)', 'SPICE (one-step-ahead)'])

    # ------------------------------------------------------------------
    # Per-trial-position breakdown (for plotting)
    # ------------------------------------------------------------------
    per_trial = pd.DataFrame({
        'trial': np.arange(T),
        'SPICE-RNN': torch.exp(torch.nanmean(ll_rnn, dim=0)).numpy(),
        'SPICE (autoregressive)': torch.exp(torch.nanmean(ll_ar, dim=0)).numpy(),
        'SPICE (one-step-ahead)': torch.exp(torch.nanmean(ll_osa, dim=0)).numpy(),
    })

    return summary, per_trial
