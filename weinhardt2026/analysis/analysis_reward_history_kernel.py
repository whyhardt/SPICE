"""
Reward-history kernel (Lau & Glimcher 2005 / Ito & Doya 2009 style).

Model-free behavioral axis: does reward received i trials ago predict staying
with the action taken at t-1, and does that predictive weight differ
depending on whether that historical reward was earned by the *same* action
(own-history stream) or the *other* action (other-history stream)?

This is deliberately parameterized to line up 1:1 with the dezfouli2019
SPICE architecture (`spice.precoded.workingmemory`), which has explicit,
separate coefficients for `reward[t-i]` in `value_reward_chosen` (own-action
history) vs `value_reward_not_chosen` (other-action history) for i=1..3.
A single-exponential model (e.g. GQL) can only produce a smooth, monotonic
geometric decay in both streams -- this kernel is the model-free ground
truth to check that claim against.

Works on any SpiceDataset (human or model-generated), so the same function
scores human behavior and simulated behavior from any fitted model.
"""

from typing import Dict, Optional

import numpy as np
import torch
import statsmodels.api as sm

from spice.resources.spice_utils import SpiceDataset


def compute_reward_history_kernel(
    dataset: SpiceDataset,
    max_lag: int = 6,
) -> Dict[str, np.ndarray]:
    """Joint logistic regression of stay-with-(t-1)-action on reward history.

    For each trial t with valid history back to t-max_lag:
        y = 1 if action[t] == action[t-1] (stay), else 0
        for each lag i in 1..max_lag:
            reward_own[i]   = reward[t-i] if action[t-i] == action[t-1] else 0
            reward_other[i] = reward[t-i] if action[t-i] != action[t-1] else 0
    Fits y ~ reward_own[1..max_lag] + reward_other[2..max_lag] jointly
    (pooled across all sessions/participants). ``reward_other`` at lag 1 is
    dropped: the reference action *is* action[t-1] by definition, so "the
    other action was taken at lag 1" is structurally impossible (an
    identically-zero column, i.e. exactly collinear with the intercept).

    Args:
        dataset: SpiceDataset (human or model-generated). xs feature layout
            assumed `[actions (n_actions one-hot), reward (n_actions cols or
            1 col), ...]` -- reward_col is the offset of the reward block.
        max_lag: number of trial-lags to include as separate regressors.

    Returns
    -------
    dict with 'lags', 'coef_own', 'coef_other', 'se_own', 'se_other'
    (each length max_lag), plus the fitted statsmodels result object.
    """
    xs = dataset.xs[:, :, 0, :]  # (B, T, F) -- within_ts assumed 1
    n_actions = dataset.n_actions
    B, T, _ = xs.shape

    actions = xs[..., :n_actions].argmax(dim=-1)  # (B, T)
    valid = ~torch.isnan(xs[..., :n_actions].sum(dim=-1))  # (B, T)

    reward_block = xs[..., n_actions:n_actions + n_actions]  # (B, T, n_actions), NaN for unchosen
    reward_received = torch.nan_to_num(reward_block, nan=0.0).gather(-1, actions.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # (B, T)

    rows = []
    for t in range(max_lag, T):
        v = valid[:, t] & valid[:, t - 1]
        for i in range(1, max_lag + 1):
            v = v & valid[:, t - i]
        if v.sum() == 0:
            continue
        idx = v.nonzero(as_tuple=True)[0]
        a_ref = actions[idx, t - 1]
        y = (actions[idx, t] == a_ref).float()

        feat = {"y": y.numpy()}
        for i in range(1, max_lag + 1):
            same = (actions[idx, t - i] == a_ref)
            r = reward_received[idx, t - i]
            feat[f"own_{i}"] = torch.where(same, r, torch.zeros_like(r)).numpy()
            if i > 1:
                feat[f"other_{i}"] = torch.where(~same, r, torch.zeros_like(r)).numpy()
        rows.append(feat)

    import pandas as pd
    df = pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)

    X_cols = [f"own_{i}" for i in range(1, max_lag + 1)] + [f"other_{i}" for i in range(2, max_lag + 1)]
    X = sm.add_constant(df[X_cols])
    y = df["y"]

    model = sm.Logit(y, X).fit(disp=0)

    lags = np.arange(1, max_lag + 1)
    coef_own = np.array([model.params[f"own_{i}"] for i in lags])
    coef_other = np.array([np.nan] + [model.params[f"other_{i}"] for i in lags[1:]])
    se_own = np.array([model.bse[f"own_{i}"] for i in lags])
    se_other = np.array([np.nan] + [model.bse[f"other_{i}"] for i in lags[1:]])

    return dict(lags=lags, coef_own=coef_own, coef_other=coef_other,
                se_own=se_own, se_other=se_other, n_trials=len(df), result=model)
