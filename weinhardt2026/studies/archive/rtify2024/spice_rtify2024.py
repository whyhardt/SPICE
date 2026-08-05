import math

import torch
import torch.nn.functional as F
from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    # library_setup declares which SINDy-fittable RNN submodules exist. DDMRNN.forward() calls
    # this 'drift' submodule every step via call_module(inputs=(stimulus,), ...) -- drift is a
    # learned function of 'stimulus', not a fixed scalar (see forward()'s per-step call_module
    # block). 'stimulus' here is the only candidate input this submodule's tiny polynomial RNN
    # is allowed to read.
    library_setup={
        'drift': [
            'stimulus',
            ],
    },
    # Both entries are None -> BaseModel creates a genuine nn.Parameter (one learnable scalar
    # per participant, per ensemble member) for each, rather than treating it as a fixed
    # constant. See docs/training.md: a concrete float here (e.g. 'drift': 0.5) would freeze
    # the value with no gradient -- exactly the mistake rtify2024_c made when it hardcoded
    # these for its "assume ground truth" diagnostic run. For 'drift', this is just the
    # INITIAL value the 'drift' submodule starts from at step 0 -- every step after that,
    # call_module() overwrites self.state['drift'] with the submodule's own output.
    memory_state={
        'drift': None,          # initial drift rate before the 'drift' submodule starts updating it
        'threshold_raw': None,  # pre-transform threshold; DDMRNN.effective_threshold() maps this
                                 # through a bounded sigmoid (not softplus, see that method's docstring)
    },
    # Cosmetic/interpretability metadata only in this model (which state(s) "drive" the
    # decision) -- DDMRNN.forward() builds its own logits directly and doesn't consult this list.
    states_in_logit=[
        'drift',
    ],
    # Must match benchmark_rtify2024_f.py's xs column layout (stimulus, time_elapsed are read
    # out of xs by init_forward_pass() into spice_signals.additional_inputs) even though this
    # particular forward() doesn't use spice_signals.additional_inputs at all.
    additional_inputs=('stimulus', 'time_elapsed'),
)


class DDMRNN(BaseModel):
    """Two-boundary DDM whose state is the actual (discretized) probability DENSITY over
    evidence -- not a single scalar trajectory -- evolved via the Fokker-Planck / Kolmogorov
    forward equation, with absorbing boundaries at +/-threshold.

    Why this exists (short version -- full chain in this variant's chat history):
    rtify2024_e's Gaussian-CDF hazard was the mathematically correct one-step boundary-crossing
    formula, but it evaluated that formula against a deterministic MEAN evidence trajectory
    (no noise ever added to `self.state['evidence']`) identical across every trial sharing a
    drift. Verified directly: the TRUE ground-truth (drift, threshold) scored a *worse*
    per-trial log-likelihood under that model than the (wrong) values gradient descent
    converged to, on the same data -- so it wasn't a training bug, the model structurally
    couldn't represent the true process. That's "frailty bias" in survival-analysis terms:
    fitting a single shared hazard curve to a population with unobserved individual-level
    heterogeneity (each trial's own noise history) biases the recovered parameters.

    Fixing this properly would normally mean giving each trial its own simulated noise
    history -- but sampling noise per trial makes the per-step decision non-differentiable
    (you'd need REINFORCE/particle-filter machinery, high-variance gradients). This class
    sidesteps that entirely by tracking the exact evolution of the *population* density
    instead of any individual trial's trajectory -- exactly what real DDM-fitting software
    (fast-dm, HDDM, pyDDM) does numerically:

      1. `evidence_pdf`: a probability MASS vector over a fixed grid of evidence values,
         conditional on the trial having survived (not yet decided) to the current step.
         Starts as a delta spike at evidence=0 (nothing accumulated yet).
      2. Each step: diffuse+drift-shift the density (see forward()'s step 1 -- split into a
         shared "spread" convolution and a per-trial "shift" interpolation, mean drift*dt, std
         sigma=diffusion_rate*sqrt(dt), the TRUE per-step noise scale).
      3. Whatever density crosses +/-threshold this step IS the exact per-step conditional
         [up, down] probability -- literally the fraction of the surviving population whose
         evidence moved past the boundary this step. Soft (sigmoid, not hard-cutoff) absorption
         at the boundary, so there's a well-defined gradient w.r.t. `threshold`. The remaining
         interior density, renormalized back to sum to 1, becomes the conditional distribution
         given survival into the next step.

    Because step 3 works entirely with the CONDITIONAL (survival-renormalized) density, the
    resulting [no_decision, up, down] triple sums to 1 at every step BY CONSTRUCTION -- so the
    same `log(probs)`-as-raw-logits convention rtify2024_e introduced still applies here
    unchanged, and so does everything downstream that consumes it: the per-step cross-entropy
    training loop, `_simulate_trial_by_trial`, `_step_probs_to_marginal`.

    Fully vectorized, no Python-level loop over trials/participants/ensemble members: a naive
    implementation would build one [G, G] transition matrix per trial and matmul it in
    (mathematically exact for one step, but either O(B*G*G) memory if done for the whole batch
    at once, or a Python loop grouping trials by participant to avoid that). Instead, the
    Gaussian transition kernel's mean (drift*dt, different per trial) and spread (sigma, THE
    SAME for every trial) are handled separately:
      - spread: one small, fixed, shared convolution kernel, applied to the whole batch in a
        single `conv1d` call (see `diffusion_kernel` in __init__).
      - shift: each trial's density translated by its own `drift*dt`, in one batched call to
        `grid_sample` (see `_shift_pdf`), which natively supports a different shift per batch
        element with no loop.
    This also generalizes for free to genuinely per-trial-varying drift (not just
    per-participant-constant), unlike the grouped-matmul approach it replaces.
    """

    def __init__(
        self,
        dt: float = 0.05,
        diffusion_rate: float = 1.0,
        grid_half_width: float = 4.0,
        grid_points: int = 161,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dt = dt
        # The single most important number in this file: the true per-step noise standard
        # deviation. This is what rtify2024_e's softmax hazard was missing entirely, and what
        # makes both that variant's Gaussian-CDF hazard AND this variant's diffusion kernel
        # correct discretizations of the real continuous-time DDM.
        self.sigma = diffusion_rate * (dt ** 0.5)
        self.grid_half_width = grid_half_width  # how far the evidence grid extends past 0 in each direction

        # The fixed set of possible evidence values every trial's density lives on, e.g.
        # grid_points=161 points evenly spaced across [-grid_half_width, +grid_half_width].
        # register_buffer (not a plain attribute) so it moves with the model on .to(device) and
        # gets saved/loaded with the model's state_dict, like a non-trainable constant should.
        x_grid = torch.linspace(-grid_half_width, grid_half_width, grid_points)
        self.register_buffer('x_grid', x_grid)
        self.dx = float(x_grid[1] - x_grid[0])  # spacing between adjacent grid points
        # Width of the soft (sigmoid) absorption transition at each boundary, in evidence
        # units -- one grid cell wide is sharp enough to closely approximate a hard absorbing
        # boundary while keeping a well-behaved (non-vanishing) gradient w.r.t. threshold.
        self.absorb_temperature = self.dx

        # Initial density: everything (probability mass 1.0) concentrated at the grid point
        # closest to evidence=0 -- a discrete delta spike, since no information has been
        # accumulated at the very start of a trial.
        initial_pdf = torch.zeros(grid_points)
        initial_pdf[torch.argmin(x_grid.abs())] = 1.0
        self.register_buffer('initial_pdf', initial_pdf)

        # Fixed "spread" kernel for the diffusion half of each step -- a small, zero-mean
        # discretized Gaussian, span +/-4 sigma (captures ~99.994% of the mass), reused as-is
        # for every trial/participant/ensemble member/step, since it doesn't depend on drift at
        # all (only sigma, which is fixed). This is what makes step 1 of forward() a single
        # ordinary conv1d over the whole batch instead of a per-trial matmul.
        kernel_radius = max(1, math.ceil(4 * self.sigma / self.dx))
        offsets = torch.arange(-kernel_radius, kernel_radius + 1, dtype=torch.float32) * self.dx
        diffusion_kernel = torch.exp(-0.5 * (offsets / self.sigma) ** 2)
        diffusion_kernel = diffusion_kernel / diffusion_kernel.sum()  # mass-conserving: rows sum to 1
        self.register_buffer('diffusion_kernel', diffusion_kernel.view(1, 1, -1))  # conv1d weight shape [out_ch, in_ch, K]
        self.kernel_radius = kernel_radius

        # Drift is computed by a real learned RNN submodule (drift = f(stimulus)) each step,
        # not a fixed per-participant scalar -- setup_module()/call_module() need a participant
        # embedding to condition that submodule on who's currently being simulated.
        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=self.dropout)

    def effective_threshold(self) -> torch.Tensor:
        """The actual decision boundary magnitude, read from the raw learnable parameter.

        Bounded into the open interval (0, grid_half_width) via a sigmoid -- deliberately
        different from rtify2024_c/_d/_e's unbounded softplus. Those variants didn't need a
        bound because their hazard formulas had no finite domain to respect; here, threshold
        has to stay strictly inside the finite evidence grid or the boundary becomes
        unreachable (the sigmoid absorption weights would collapse to ~0 everywhere), silently
        breaking training. Sigmoid guarantees that can't happen, at any parameter value.
        """
        # self.state['threshold_raw'] has shape [W, E, B, n_items=1]; [..., 0] drops the
        # trailing size-1 "items" axis, leaving [W, E, B].
        return self.grid_half_width * torch.sigmoid(self.state['threshold_raw'][..., 0])

    def mean_evidence(self) -> torch.Tensor:
        """E[evidence | survived to now] -- a single scalar summary of the current density,
        purely for plotting (analysis_rtify2024_f.py's _rollout_state_trajectory calls this to
        draw an "evidence" line, standing in for the scalar `evidence` state earlier variants
        had). Never used inside forward() itself -- the full density is what actually matters
        for the decision computation, this is just a human-readable projection of it.
        """
        pdf = self.state['evidence_pdf']  # [W, E, B, G]
        # Expectation = sum over grid points of (probability at that point) * (value at that point).
        return (pdf * self.x_grid.view(1, 1, 1, -1)).sum(dim=-1)

    def _shift_pdf(self, pdf: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Translate `pdf` along its last (grid) axis by `shift` evidence-units, per batch
        element, via bilinear interpolation -- the "drift" half of each step (see __init__ and
        forward()'s step 1). Density that shifts past either grid edge is dropped (zero-padded),
        matching the assumption that `grid_half_width` gives enough margin past `threshold` for
        this to essentially never matter in practice.

        pdf: [..., G]. shift: same leading shape as pdf minus the last dim (one shift value per
        batch element). Implemented via `grid_sample` rather than a hand-rolled gather so the
        interpolation/edge-clamping logic is PyTorch's well-tested code, not new bespoke code:
        `grid_sample` natively supports a different sampling location per batch element in one
        vectorized call, which is exactly "shift every trial by its own drift*dt, no loop."
        """
        leading_shape = pdf.shape[:-1]
        G = pdf.shape[-1]
        N = pdf.numel() // G  # flatten every leading dim (W, E, B, ...) into one batch axis

        # grid_sample needs a 4D input [N, C=1, H=1, W=G] -- a batch of N single-row "images".
        pdf_flat = pdf.reshape(N, 1, 1, G)
        shift_flat = shift.reshape(N)

        # Output position j should read its value from input position (j - shift_cells): a
        # positive shift moves the density toward higher x, so each output position pulls its
        # mass from a lower (behind it) source position -- same derivation as a moving-average
        # filter, just with a continuous (sub-grid-cell) offset instead of an integer one.
        j = torch.arange(G, device=pdf.device, dtype=pdf.dtype)  # [G]
        shift_cells = shift_flat.view(N, 1) / self.dx             # [N, 1] -- shift in units of grid cells
        src = j.view(1, G) - shift_cells                          # [N, G] -- source position per output position

        # grid_sample expects sampling coordinates normalized to [-1, 1] (with align_corners=True,
        # -1 <-> index 0 and +1 <-> index G-1); a "row" (height) coordinate is required too even
        # though there's only one row, so it's fixed at 0.
        coord_x = 2 * src / (G - 1) - 1
        coord_y = torch.zeros_like(coord_x)
        grid = torch.stack((coord_x, coord_y), dim=-1).unsqueeze(1)  # [N, 1, G, 2]

        shifted = F.grid_sample(pdf_flat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # [N, 1, 1, G]
        return shifted.reshape(*leading_shape, G)

    def forward(self, inputs: torch.Tensor, prev_state: torch.Tensor = None):
        # Standard SPICE boilerplate: unpacks `inputs` into spice_signals (actions, additional
        # inputs, participant/block/trial ids, a pre-zeroed logits buffer), and either restores
        # `self.state` from `prev_state` (mid-sequence BPTT-truncation chunk) or initializes it
        # fresh from CONFIG.memory_state + this model's learnable_initial_values (first chunk).
        spice_signals = self.init_forward_pass(inputs, prev_state)

        # W = within-trial timesteps (always 1 here -- this study uses the T=step/W=1 axis
        # reframing shared by rtify2024_b through _f, where the "outer" trial axis T actually
        # indexes DDM discretization steps, not separate behavioral trials).
        # E = ensemble size (1 in every config in this study -- SINDy ensembling isn't used).
        # B = batch size, i.e. number of (synthetic) trials being processed in parallel.
        W, E, B = self.state['drift'].shape[0], self.state['drift'].shape[1], self.state['drift'].shape[2]
        assert W == 1, "DDMRNN assumes the T=step/W=1 trial-axis reframing used throughout rtify2024_b/c/d/e/f."
        G = self.x_grid.shape[0]  # number of grid points

        # On the very first chunk of a sequence (prev_state is None), inject the initial
        # density -- a fresh delta spike at 0 for every (ensemble, batch) trial -- as an
        # ADDITIONAL entry in self.state alongside 'drift'/'threshold_raw' (which
        # init_forward_pass already populated above). On later BPTT chunks, self.state was
        # already restored wholesale from prev_state, so 'evidence_pdf' carries over correctly
        # without needing to be touched here.
        if prev_state is None:
            self.state['evidence_pdf'] = self.initial_pdf.view(1, 1, 1, G).expand(W, E, B, G).clone()

        threshold = self.effective_threshold()  # [W, E, B] -- this trial's boundary magnitude
        # `drift` itself is now computed by call_module() fresh every step below (from
        # 'stimulus' via the learned 'drift' RNN submodule), not read once here -- only
        # `threshold` stays step-invariant, so only it gets hoisted out of the loop.
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # Precompute the soft absorption weight for every grid point, ONCE per forward() call
        # (not per step) -- valid because `threshold` doesn't change across the max_steps loop
        # in this constant-threshold setup. sigmoid((x - threshold)/temperature) is ~0 for x far
        # below threshold, ~1 for x far above it, and transitions smoothly across ~1 grid cell
        # right at the boundary -- a differentiable stand-in for the hard indicator "x >= threshold".
        x_row = self.x_grid.view(1, 1, 1, G)  # [1, 1, 1, G], broadcasts against threshold's [W, E, B]
        absorb_up = torch.sigmoid((x_row - threshold.unsqueeze(-1)) / self.absorb_temperature)     # [W, E, B, G]
        absorb_down = torch.sigmoid((-x_row - threshold.unsqueeze(-1)) / self.absorb_temperature)  # [W, E, B, G]
        # A grid point survives this step only if it's absorbed by NEITHER boundary.
        survive_weight = (1. - absorb_up) * (1. - absorb_down)  # [W, E, B, G]

        # Main loop: one DDM discretization step per iteration (spice_signals.trials is
        # arange(T), T = max_steps for a full, un-chunked call). No loop over
        # ensemble/participant/trial anywhere in here -- both operations inside are single
        # batched calls covering every (W, E, B) trial at once.
        for timestep in spice_signals.trials:
            pdf = self.state['evidence_pdf']  # [1, E, B, G] -- density BEFORE this step, already
                                               # conditional on survival up to (not including) this step

            # Compute this step's drift from this step's stimulus via the learned RNN
            # submodule -- call_module() writes its result straight into self.state['drift'].
            stimulus = spice_signals.additional_inputs['stimulus'][timestep]  # [W, E, B, 1]
            self.call_module(
                key_module='drift',
                key_state='drift',
                action_mask=None,
                inputs=(stimulus,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            drift = self.state['drift'][..., 0]  # [W, E, B] -- this step's drift rate
            drift_shift = drift * self.dt        # [W, E, B] -- this step's mean evidence displacement

            # --- 1. Diffuse + drift-shift the density (spread, then shift) ---
            # Spread: convolve every trial's density with the SAME small fixed kernel in one
            # call -- valid because sigma (and hence this kernel) doesn't depend on drift/
            # participant/trial at all. conv1d needs a [N, C, L] input; flatten (W=1, E, B)
            # into one batch axis N and treat the grid axis G as the 1D spatial length L.
            pdf_flat = pdf.reshape(1 * E * B, 1, G)
            pdf_spread = F.conv1d(pdf_flat, self.diffusion_kernel, padding=self.kernel_radius).reshape(1, E, B, G)
            # Shift: translate each trial's (now-spread) density by its own drift*dt, all in
            # one batched call -- see _shift_pdf's docstring for why grid_sample handles the
            # "different shift per trial" part without a Python loop.
            pdf_diffused = self._shift_pdf(pdf_spread, drift_shift)  # [1, E, B, G]

            # --- 2. Absorb whatever density crossed a boundary this step ---
            # Total probability mass (out of the surviving population) that landed past each
            # boundary after diffusing -- this directly IS the per-step conditional [up]/[down]
            # probability, no separate hazard-rate formula needed.
            absorbed_up = (pdf_diffused * absorb_up).sum(dim=-1)      # [1, E, B]
            absorbed_down = (pdf_diffused * absorb_down).sum(dim=-1)  # [1, E, B]
            # Density that survived both boundaries this step -- not yet renormalized.
            interior = pdf_diffused * survive_weight                  # [1, E, B, G]
            p_no_decision = interior.sum(dim=-1)                      # [1, E, B]

            # These three sum to (very close to) 1 by construction: every unit of density
            # either got absorbed up, absorbed down, or stayed interior. log(), not softmax --
            # cross_entropy_loss applies its own log_softmax internally, and softmax(log(p)) == p
            # exactly when p already sums to 1, so this is the correct "raw logits" to hand it.
            probs = torch.stack((p_no_decision, absorbed_up, absorbed_down), dim=-1)  # [1, E, B, 3]
            spice_signals.logits[timestep] = torch.log(probs.clamp_min(1e-8))

            # --- 3. Renormalize the surviving density for the next step ---
            # `interior` currently holds "probability of (this evidence value) AND (survived)".
            # Dividing by its own total turns it back into "probability of (this evidence value)
            # GIVEN survived" -- the correct conditional distribution to diffuse again next step.
            self.state['evidence_pdf'] = interior / interior.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Standard SPICE boilerplate: permutes the logits tensor back to the (E, B, T, W, A)
        # layout the rest of the framework (loss functions, evaluation code) expects.
        return self.post_forward_pass(spice_signals).logits, self.get_state()

