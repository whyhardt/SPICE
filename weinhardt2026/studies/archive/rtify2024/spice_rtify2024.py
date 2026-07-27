import torch
from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        'drift': [
            'stimulus', 
            ],
        'boundary': [ 
            'time_elapsed',
            ],
    },
    memory_state={
        'drift': None,
        'boundary': None,
    },
    additional_inputs=('stimulus', 'time_elapsed'),
)


class DDMRNN(BaseModel):
    """Two-boundary DDM with a hazard-based (RTify-style) decision-time likelihood."""

    def __init__(self, dt: float = 0.05, **kwargs):
        super().__init__(**kwargs)

        self.dt = dt

        self.hazard_threshold = torch.nn.Parameter(torch.tensor(0.))
        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=self.dropout)

        # Re-setup with dt: BaseModel's automatic setup_modules_from_config() already
        # created 'drift' with the default dt=1. -- override so its residual update and
        # SINDy fit both scale by the trial's actual per-step time interval, letting
        # discovered coefficients read as per-second rates rather than per-step deltas
        # that shrink as `max_steps` grows.
        self.setup_module(key_module='drift', dt=self.dt, dropout=self.dropout, within_trial_timesteps=True)
        self.setup_module(key_module='boundary', include_state=False, dt=self.dt, dropout=self.dropout, within_trial_timesteps=True)
        
    def forward(self, inputs: torch.Tensor, prev_state: torch.Tensor = None):
        spice_signals = self.init_forward_pass(inputs, prev_state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # No non-decision-time shift: the decision process runs over the full
        # max_steps range from t=0. True RTs include a real ndt offset (see
        # simulate_ddm), so a handful of trials with true RT < ndt will have
        # their earliest bins essentially unreachable -- accepted tradeoff for
        # a model with no input-length-dependent special-casing, which lets
        # any W-length window (not just a full max_steps trial) be run through
        # forward() directly, e.g. for one-step or partial-rollout SINDy fitting.

        # Single outer trial (T=1): additional_inputs are [W=max_steps, E, B, 1].
        # One call_module invocation lets the RNN process the whole within-trial
        # sequence internally (its own compiled loop over W), instead of us looping.
        stimulus = spice_signals.additional_inputs['stimulus'][0]
        time_elapsed = spice_signals.additional_inputs['time_elapsed'][0]

        self.call_module(
            key_module='drift',
            key_state='drift',
            action_mask=None,
            inputs=(
                stimulus,
                ),
            participant_index=spice_signals.participant_ids,
            participant_embedding=participant_embedding,
        )  # self.state['drift']: [W=max_steps, E, B, n_items]
        
        self.call_module(
            key_module='boundary',
            key_state='boundary',
            action_mask=None,
            inputs=(
                time_elapsed,
                ),
            participant_index=spice_signals.participant_ids,
            participant_embedding=participant_embedding,
        )  # self.state['boundary']: [W=max_steps, E, B, n_items]

        # `drift` *is* the evidence: no separate external integration step. Its own
        # residual state already accumulates (h[t+1] = h[t] + dt*n[t], now dt-scaled
        # via setup_module(dt=...)); "leak" is whatever self-coefficient SINDy finds
        # on drift[t] itself, not a separately hand-designed decay parameter.
        drift = self.state['drift'][..., 0:1]  # single accumulator: [W, E, B, 1]
        threshold = torch.nn.functional.softplus(self.state['boundary'][..., 0:1])

        p_stop_up = torch.sigmoid(drift - threshold)
        p_stop_down = torch.sigmoid(-drift - threshold)

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