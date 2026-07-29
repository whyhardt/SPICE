import torch
from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        'drift': [
            'stimulus',
            ],
        # 'integrator': [
        #     'drift',
        #     ],
        # 'threshold_raw': [
        #     'time_elapsed',
        #     ],
    },
    memory_state={
        'drift': None,     # learnable initial drift rate -- a genuine parameter
        'evidence': 0.0,   # always starts at 0: no information accumulated yet
        'threshold_raw': None,
    },
    states_in_logit=[
        'drift',
    ],
    additional_inputs=('stimulus', 'time_elapsed'),
)


class DDMRNN(BaseModel):
    """Two-boundary DDM with a hazard-based (RTify-style) decision-time likelihood."""

    def __init__(self, dt: float = 0.05, non_decision_time: float = 0.2, **kwargs):
        super().__init__(**kwargs)

        self.dt = dt
        self.non_decision_time = non_decision_time

        # self.threshold_raw = torch.nn.Parameter(torch.tensor(0.))
        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=self.dropout)

        # Re-setup with dt: BaseModel's automatic setup_modules_from_config() already
        # created 'drift' with the default dt=1. -- override so its residual update and
        # SINDy fit both scale by the trial's actual per-step time interval, letting
        # discovered coefficients read as per-second rates rather than per-step deltas
        # that shrink as `max_steps` grows.
        self.setup_module(key_module='drift', dt=self.dt, dropout=self.dropout, within_trial_timesteps=True)
        
        # include_state=True: evidence's own value is a valid SINDy regressor,
        # so a self-coefficient a=1-dt*l can be discovered -- a=1 is a perfect
        # integrator, a<1 is leak, both expressible by the same equation
        # instead of a hand-designed decay parameter. Leak belongs on evidence
        # (the thing that decays back toward 0), not on drift (the rate).
        # include_bias=False: without this, a constant contribution to
        # evidence's increment is collinear with drift whenever drift is
        # ~constant (b*drift and c*1 give the identical evidence trajectory),
        # so training has no pressure to route the true rate through drift at
        # all -- it can bake it into evidence's own bias term instead, leaving
        # drift free to be arbitrary. Removing the bias forces any systematic
        # drive through drift specifically. polynomial_degree=2 (not 1): every
        # remaining candidate (evidence, drift, evidence^2, evidence*drift,
        # drift^2) still genuinely depends on evidence's/drift's actual values,
        # so this doesn't reopen a bypass -- capping at degree=1 just because
        # the synthetic ground truth here is linear would assume the answer
        # instead of letting SPICE discover whatever structure the data has.
        # self.setup_module(key_module='integrator', include_state=True, include_bias=False, dt=self.dt, dropout=self.dropout, within_trial_timesteps=True, polynomial_degree=2)
        
        # self.setup_module(key_module='threshold_raw', include_state=False, dt=self.dt, dropout=self.dropout, within_trial_timesteps=True)
        
    def forward(self, inputs: torch.Tensor, prev_state: torch.Tensor = None):
        spice_signals = self.init_forward_pass(inputs, prev_state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # Non-decision time is applied only to the final output below, never to
        # the internal computation: drift/evidence/threshold always run over
        # the full input range unchanged, so self.state[...] stays full-length
        # and forward() stays input-length-agnostic (any W-length window still
        # works for Stage 2's one-step/rollout fitting). The decision process
        # accumulates on its own internal clock starting at t=0 regardless of
        # ndt; only the reported response time is shifted by a constant delay.

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
        )  # self.state['drift']: [W=max_steps, E, B, n_items] -- the drift rate

        # evidence accumulates dt*drift[t] each step, starting from a fixed 0
        # (memory_state=0.0, never learnable). drift's full trajectory is
        # already computed above, so this is a second independent vectorized
        # call, not a step-by-step Python loop -- drift doesn't depend on
        # evidence, so there's no feedback coupling forcing interleaving.
        # self.call_module(
        #     key_module='integrator',
        #     key_state='evidence',
        #     action_mask=None,
        #     inputs=(
        #         self.state['drift'],
        #         ),
        #     participant_index=spice_signals.participant_ids,
        #     participant_embedding=participant_embedding,
        # )  # self.state['evidence']: [W=max_steps, E, B, n_items]
        self.state['evidence'] = self.dt * torch.cumsum(self.state['drift'], dim=0)
        evidence = self.state['evidence'][..., 0:1]  # single accumulator: [W, E, B, 1]
        
        # boundary_updates = self.call_module(
        #     key_module='threshold_raw',
        #     # key_state='threshold_raw',
        #     action_mask=None,
        #     inputs=(
        #         time_elapsed,
        #         ),
        #     participant_index=spice_signals.participant_ids,
        #     participant_embedding=participant_embedding,
        # )
        
        # boundary was called with key_state=None, so self.state['threshold_raw'] still
        # holds the learned initial value (untouched by call_module) rather than
        # the zero it would've been reset to -- combine it with the state-blind
        # cumulative update here, then write the combined trajectory back so
        # get_state() reports what's actually used (plotting, Stage 2 SINDy
        # refit's _vectorize_state_sequential both read state['threshold_raw']).
        # self.state['threshold_raw'] = self.state['threshold_raw'] + boundary_updates
        threshold = torch.nn.functional.softplus(self.state['threshold_raw'][..., 0:1])
        # threshold = torch.nn.functional.softplus(self.threshold_raw)

        p_stop_up = torch.sigmoid(evidence - threshold)
        p_stop_down = torch.sigmoid(-evidence - threshold)

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

        # Shift the reported response time by a constant non-decision delay --
        # output only, self.state[...] above is untouched. Drops the last
        # ndt_bins bins to keep the array length fixed (negligible probability
        # mass lost for the slowest responses, as long as ndt << t_max).
        W = p_decision_up.shape[0]
        ndt_bins = min(max(int(round(self.non_decision_time / self.dt)), 0), W - 1)
        if ndt_bins > 0:
            pad = torch.zeros(ndt_bins, *p_decision_up.shape[1:], device=self.device, dtype=p_decision_up.dtype)
            p_decision_up = torch.cat((pad, p_decision_up[:-ndt_bins]), dim=0)
            p_decision_down = torch.cat((pad, p_decision_down[:-ndt_bins]), dim=0)

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
        loss = -torch.log(p_obs[valid].clamp_min(1e-8)).mean()
        return loss

    return loss_fn