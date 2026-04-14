"""
Diagnostic toolkit for analyzing SPICE model architectural bottlenecks.

All diagnostics work by externally probing fitted model modules
via forward hooks — no modifications to the SPICE backend required.

Usage:
    from spice.utils.diagnostics import SpiceDiagnostics

    diag = SpiceDiagnostics(estimator, dataset)

    # Per-module polynomial R² test
    r2_report = diag.polynomial_adequacy()

    # Which modules have the largest SINDy approximation gap?
    sindy_report = diag.sindy_loss_per_module()

    # Which modules hurt behavioral prediction when swapped to SINDy?
    swap_report = diag.module_swap_test()

    # Is the RNN relying on embeddings that SINDy can't access?
    emb_report = diag.embedding_dependence()

    # Are hidden states in polynomial-friendly ranges?
    state_report = diag.state_range()

    # What's the polynomial missing?
    residual_report = diag.residual_structure()

    # Combined summary table
    summary_df = diag.summary()
"""

import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple

from spice.resources.sindy_differentiable import compute_polynomial_library


class SpiceDiagnostics:
    """Diagnostic toolkit for analyzing SPICE model bottlenecks.

    Probes fitted GRU modules externally to identify where polynomial
    approximation fails, without modifying the backend.

    Args:
        estimator: Fitted SpiceEstimator instance.
        dataset: SpiceDataset used for probing.
    """

    def __init__(self, estimator, dataset):
        self.estimator = estimator
        self.model = estimator.model
        self.dataset = dataset
        self.config = estimator.spice_config

    def _collect_module_data(self) -> Dict[str, List[Tuple[torch.Tensor, ...]]]:
        """Run forward pass with hooks to collect per-module GRU I/O.

        Returns:
            Dict mapping module_name -> list of (inputs, state, output) tuples,
            one per call_module invocation (one per trial in forward pass).
            - inputs: (W, E, B, I, F) — full GRU input (controls + embedding)
            - state:  (W, E, B, I) or None — hidden state before update
            - output: (W, E, B, I, 1) — GRU output (before action masking)
        """
        model = self.model
        model.eval()

        hook_data = {name: [] for name in model.submodules_rnn}

        def make_hook(name):
            def hook_fn(module, args, kwargs, output):
                state = kwargs.get('state', None)
                hook_data[name].append((
                    args[0].detach().cpu(),
                    state.detach().cpu() if state is not None else None,
                    output.detach().cpu(),
                ))
            return hook_fn

        handles = []
        for name, module in model.submodules_rnn.items():
            handles.append(module.register_forward_hook(make_hook(name), with_kwargs=True))

        was_sindy = model.use_sindy
        model.use_sindy = False
        with torch.no_grad():
            xs = self.dataset.xs.to(model.device)
            model(xs)
        model.use_sindy = was_sindy

        for h in handles:
            h.remove()

        return hook_data

    def polynomial_adequacy(self, degree: Optional[int] = None) -> pd.DataFrame:
        """Per-module polynomial R² test.

        Hooks GRU modules during a forward pass, collects (h_in, controls, h_out)
        per timestep, builds polynomial library, fits OLS, reports R².

        | R² value   | Interpretation                           |
        |------------|------------------------------------------|
        | > 0.95     | SINDy-friendly                           |
        | 0.90-0.95  | Marginal — consider minor restructuring  |
        | < 0.90     | Needs restructuring                      |
        | < 0.80     | Fundamental expressiveness mismatch      |

        Args:
            degree: Polynomial degree for regression. If None, uses each
                    module's configured degree from sindy_specs.

        Returns:
            DataFrame with columns: module, r2, n_samples, n_terms, delta_h_std
        """
        hook_data = self._collect_module_data()
        results = []

        for module_name in self.model.submodules_rnn:
            calls = hook_data[module_name]
            if not calls:
                continue

            mod_degree = (degree if degree is not None
                          else self.model.sindy_specs[module_name]['polynomial_degree'])
            n_controls = len(self.config.library_setup[module_name])
            feature_names = self.model.sindy_specs[module_name]['input_names']
            library_terms = self.model.sindy_candidate_terms[module_name]

            all_h_in = []
            all_controls = []
            all_h_out = []

            for inputs, state, output in calls:
                W, E, B, I, F = inputs.shape
                # h_in: hidden state at last within-trial step (GRU initial hidden)
                h_in = state[-1] if state is not None else torch.zeros(E, B, I)
                # controls: first n_controls dims of GRU input (before embedding)
                if n_controls > 0:
                    controls = inputs[-1, :, :, :, :n_controls]  # (E, B, I, n_controls)
                else:
                    controls = inputs[-1, :, :, :, :0]  # (E, B, I, 0)
                # h_out: GRU output at last within-trial step
                h_out = output[-1, :, :, :, 0]  # (E, B, I)

                all_h_in.append(h_in.reshape(E * B, I))
                all_controls.append(controls.reshape(E * B, I, n_controls))
                all_h_out.append(h_out.reshape(E * B, I))

            h_in_stacked = torch.stack(all_h_in)          # (T, E*B, I)
            controls_stacked = torch.stack(all_controls)   # (T, E*B, I, n_controls)
            h_out_stacked = torch.stack(all_h_out)         # (T, E*B, I)
            delta_h = h_out_stacked - h_in_stacked         # (T, E*B, I)

            # Filter out items whose state never varies (never updated by action mask)
            I = h_in_stacked.shape[2]
            active_items = [i for i in range(I)
                           if h_in_stacked[:, :, i].var() > 1e-6]
            if not active_items:
                active_items = list(range(I))  # fallback: keep all

            h_in_filtered = h_in_stacked[:, :, active_items]
            controls_filtered = controls_stacked[:, :, active_items, :]
            delta_filtered = delta_h[:, :, active_items]

            # Build polynomial library using the same function as forward_sindy
            library_matrix = compute_polynomial_library(
                h_in_filtered, controls_filtered, mod_degree,
                feature_names, library_terms,
            )  # (T, E*B, len(active_items), n_terms)

            # Flatten: all (trial, ensemble*batch, active_item) → rows
            n_terms = library_matrix.shape[-1]
            lib_flat = library_matrix.reshape(-1, n_terms).float()
            dh_flat = delta_filtered.reshape(-1, 1).float()

            # OLS via lstsq
            coeffs = torch.linalg.lstsq(lib_flat, dh_flat).solution
            predicted = lib_flat @ coeffs

            ss_res = ((dh_flat - predicted) ** 2).sum()
            ss_tot = ((dh_flat - dh_flat.mean()) ** 2).sum()
            r2 = (1 - ss_res / ss_tot).item() if ss_tot > 1e-12 else float('nan')

            results.append({
                'module': module_name,
                'r2': round(r2, 4),
                'n_samples': lib_flat.shape[0],
                'n_terms': n_terms,
                'n_active_items': len(active_items),
                'delta_h_std': round(dh_flat.std().item(), 6),
            })

        return pd.DataFrame(results)

    def gate_saturation(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Per-module GRU gate distribution analysis.

        Recomputes gate activations from module weights and collected
        input/state data. Returns raw gate values for plotting/analysis.

        | Gate distribution           | Interpretation                          |
        |-----------------------------|-----------------------------------------|
        | Unimodal, peak near 0.5     | GRU in linear regime (SINDy-friendly)   |
        | Bimodal, peaks near 0 and 1 | Hard mode-switching (split module)      |
        | Unimodal, peak near 0 or 1  | Gate effectively constant (simplify)    |

        Returns:
            Dict mapping module_name -> {
                'z': Tensor — flattened update gate values,
                'r': Tensor — flattened reset gate values,
                'z_mean': float, 'z_std': float,
                'r_mean': float, 'r_std': float,
            }
        """
        hook_data = self._collect_module_data()
        results = {}

        for module_name, calls in hook_data.items():
            if not calls:
                continue

            gru = self.model.submodules_rnn[module_name]
            w_linear = gru.weight_linear.detach().cpu()
            b_linear = gru.bias_linear.detach().cpu()
            w_ih = gru.weight_ih.detach().cpu()
            b_ih = gru.bias_ih.detach().cpu()
            w_hh = gru.weight_hh.detach().cpu()
            b_hh = gru.bias_hh.detach().cpu()
            H = gru.hidden_size

            all_z = []
            all_r = []

            for inputs, state, _ in calls:
                W, E, B, I, F = inputs.shape
                x = inputs.reshape(W, E, B * I, F)
                h = (state[-1].reshape(E, B * I, 1)
                     if state is not None
                     else torch.zeros(E, B * I, 1))

                # Linear projection
                y = (torch.einsum('eoi,webi->webo', w_linear, x)
                     + b_linear.unsqueeze(1))

                # Input-to-hidden (precomputed for all within-trial steps)
                gi_all = (torch.einsum('ego,webo->webg', w_ih, y)
                          + b_ih.unsqueeze(1))

                # Step through within-trial timesteps
                for t in range(W):
                    gi = gi_all[t]
                    gh = (torch.einsum('ego,ebo->ebg', w_hh, h)
                          + b_hh.unsqueeze(1))

                    r = torch.sigmoid(gi[..., :H] + gh[..., :H])
                    z = torch.sigmoid(gi[..., H:2*H] + gh[..., H:2*H])
                    n = torch.tanh(gi[..., 2*H:] + r * gh[..., 2*H:])
                    h = (1 - z) * n + z * h

                    all_r.append(r.reshape(-1))
                    all_z.append(z.reshape(-1))

            z_all = torch.cat(all_z)
            r_all = torch.cat(all_r)

            results[module_name] = {
                'z': z_all,
                'r': r_all,
                'z_mean': round(z_all.mean().item(), 4),
                'z_std': round(z_all.std().item(), 4),
                'r_mean': round(r_all.mean().item(), 4),
                'r_std': round(r_all.std().item(), 4),
            }

        return results

    def gate_conditioned_r2(self, module_name: str, degree: Optional[int] = None) -> pd.DataFrame:
        """Per-regime polynomial R² for a specific module, split by update gate z.

        Tests whether each gate regime individually is polynomial-amenable.
        If both regimes have high R² but the combined R² is low, the fix is
        to externalize whatever condition triggers the mode switch.

        Args:
            module_name: Name of the module to analyze.
            degree: Polynomial degree. If None, uses module's configured degree.

        Returns:
            DataFrame with columns: regime, r2, n_samples, n_terms
        """
        hook_data = self._collect_module_data()
        calls = hook_data[module_name]
        if not calls:
            return pd.DataFrame()

        mod_degree = (degree if degree is not None
                      else self.model.sindy_specs[module_name]['polynomial_degree'])
        n_controls = len(self.config.library_setup[module_name])
        feature_names = self.model.sindy_specs[module_name]['input_names']
        library_terms = self.model.sindy_candidate_terms[module_name]

        # GRU weights for gate recomputation
        gru = self.model.submodules_rnn[module_name]
        w_linear = gru.weight_linear.detach().cpu()
        b_linear = gru.bias_linear.detach().cpu()
        w_ih = gru.weight_ih.detach().cpu()
        b_ih = gru.bias_ih.detach().cpu()
        w_hh = gru.weight_hh.detach().cpu()
        b_hh = gru.bias_hh.detach().cpu()
        H = gru.hidden_size

        all_h_in = []
        all_controls = []
        all_h_out = []
        all_z = []

        for inputs, state, output in calls:
            W, E, B, I, F = inputs.shape

            # Polynomial data (last within-trial step)
            h_in = state[-1] if state is not None else torch.zeros(E, B, I)
            if n_controls > 0:
                controls = inputs[-1, :, :, :, :n_controls]
            else:
                controls = inputs[-1, :, :, :, :0]
            h_out = output[-1, :, :, :, 0]

            # Gate recomputation
            x = inputs.reshape(W, E, B * I, F)
            h_gate = (state[-1].reshape(E, B * I, 1)
                      if state is not None
                      else torch.zeros(E, B * I, 1))

            y = (torch.einsum('eoi,webi->webo', w_linear, x)
                 + b_linear.unsqueeze(1))
            gi_all = (torch.einsum('ego,webo->webg', w_ih, y)
                      + b_ih.unsqueeze(1))

            for t in range(W):
                gi = gi_all[t]
                gh = (torch.einsum('ego,ebo->ebg', w_hh, h_gate)
                      + b_hh.unsqueeze(1))
                r = torch.sigmoid(gi[..., :H] + gh[..., :H])
                z = torch.sigmoid(gi[..., H:2*H] + gh[..., H:2*H])
                n = torch.tanh(gi[..., 2*H:] + r * gh[..., 2*H:])
                h_gate = (1 - z) * n + z * h_gate

            # z from last within-trial step → (E, B, I)
            z_last = z.reshape(E, B, I)

            all_h_in.append(h_in.reshape(E * B, I))
            all_controls.append(controls.reshape(E * B, I, n_controls))
            all_h_out.append(h_out.reshape(E * B, I))
            all_z.append(z_last.reshape(E * B, I))

        h_in_stacked = torch.stack(all_h_in)
        controls_stacked = torch.stack(all_controls)
        h_out_stacked = torch.stack(all_h_out)
        z_stacked = torch.stack(all_z)
        delta_h = h_out_stacked - h_in_stacked

        # Filter inactive items
        I = h_in_stacked.shape[2]
        active_items = [i for i in range(I)
                        if h_in_stacked[:, :, i].var() > 1e-6]
        if not active_items:
            active_items = list(range(I))

        h_in_f = h_in_stacked[:, :, active_items]
        controls_f = controls_stacked[:, :, active_items, :]
        delta_f = delta_h[:, :, active_items]
        z_f = z_stacked[:, :, active_items]

        # Build polynomial library
        library_matrix = compute_polynomial_library(
            h_in_f, controls_f, mod_degree, feature_names, library_terms,
        )

        # Flatten
        n_terms = library_matrix.shape[-1]
        lib_flat = library_matrix.reshape(-1, n_terms).float()
        dh_flat = delta_f.reshape(-1).float()
        z_flat = z_f.reshape(-1).float()

        # R² per regime
        results = []
        for regime_name, mask in [('z <= 0.5', z_flat <= 0.5),
                                   ('z > 0.5', z_flat > 0.5)]:
            n_regime = mask.sum().item()
            if n_regime < n_terms + 1:
                results.append({
                    'regime': regime_name,
                    'r2': float('nan'),
                    'n_samples': n_regime,
                    'n_terms': n_terms,
                })
                continue

            lib_regime = lib_flat[mask]
            dh_regime = dh_flat[mask].unsqueeze(-1)

            coeffs = torch.linalg.lstsq(lib_regime, dh_regime).solution
            predicted = lib_regime @ coeffs

            ss_res = ((dh_regime - predicted) ** 2).sum()
            ss_tot = ((dh_regime - dh_regime.mean()) ** 2).sum()
            r2 = (1 - ss_res / ss_tot).item() if ss_tot > 1e-12 else float('nan')

            results.append({
                'regime': regime_name,
                'r2': round(r2, 4),
                'n_samples': n_regime,
                'n_terms': n_terms,
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # New diagnostics for identifying SINDy bottleneck modules
    # ------------------------------------------------------------------

    def sindy_loss_per_module(self) -> pd.DataFrame:
        """Per-module SINDy approximation error using trained coefficients.

        Monkey-patches ``compute_vectorized_sindy_loss`` during a single
        forward pass to record the MSE between each module's RNN output
        and its SINDy prediction (using the *current* trained coefficients).

        Useful after training with ``sindy_weight > 0`` to see which
        modules contribute most to the total ``sindy_loss`` gap.

        Returns:
            DataFrame with columns: module, sindy_mse, n_trials
        """
        model = self.model
        device = model.device

        module_losses: Dict[str, float] = {}
        module_n_trials: Dict[str, int] = {}

        # Capture per-module loss by intercepting the vectorized computation
        original_compute = type(model).compute_vectorized_sindy_loss

        def recording_compute(self_model):
            # Compute per-module loss from buffers without accumulating
            for module_name, buffers in self_model._sindy_buffers.items():
                if module_name not in self_model.sindy_coefficients:
                    continue

                n_trials = len(buffers['h_current'])
                h_current = torch.cat(buffers['h_current'], dim=0)
                h_next_rnn = torch.cat(buffers['h_next_rnn'], dim=0)
                controls = torch.cat(buffers['controls'], dim=0)

                W_per_trial = buffers['h_current'][0].shape[0]
                masks = torch.stack(buffers['action_mask'], dim=0)
                if W_per_trial > 1:
                    masks = masks.unsqueeze(1).expand(-1, W_per_trial, -1, -1, -1)
                    masks = masks.reshape(-1, *masks.shape[2:])

                h_next_sindy = self_model.forward_sindy(
                    h_current=h_current,
                    key_module=module_name,
                    participant_ids=buffers['participant_ids'],
                    experiment_ids=buffers['experiment_ids'],
                    controls=controls,
                    polynomial_degree=self_model.sindy_polynomial_degree,
                )

                diff = (h_next_rnn - h_next_sindy) ** 2
                masked_diff = torch.where(masks == 1, diff, torch.zeros_like(diff))
                n_masked = masks.sum(dim=-1).clamp(min=1)
                per_sample_loss = masked_diff.sum(dim=-1) / n_masked
                module_losses[module_name] = per_sample_loss.mean().item()
                module_n_trials[module_name] = n_trials

        # Save model state
        was_training = model.training
        was_sindy = model.use_sindy
        was_fit_sindy = getattr(model, 'fit_sindy', True)
        was_rnn_finished = getattr(model, 'rnn_training_finished', False)

        # eval() disables dropout; override training flag so the sindy_loss
        # gate check in call_module fires
        model.eval()
        model.training = True
        model.use_sindy = False
        model.fit_sindy = True
        model.rnn_training_finished = False
        type(model).compute_vectorized_sindy_loss = recording_compute

        with torch.no_grad():
            xs = self.dataset.xs.to(device)
            model(xs)

        # Restore
        type(model).compute_vectorized_sindy_loss = original_compute
        model.use_sindy = was_sindy
        model.fit_sindy = was_fit_sindy
        model.rnn_training_finished = was_rnn_finished
        if was_training:
            model.train()
        else:
            model.eval()

        results = []
        for name in model.submodules_rnn:
            if name in module_losses:
                results.append({
                    'module': name,
                    'sindy_mse': round(module_losses[name], 6),
                    'n_trials': module_n_trials[name],
                })

        return pd.DataFrame(results)

    def module_swap_test(self, loss_fn=None) -> pd.DataFrame:
        """Behavioral loss impact of swapping each module individually to SINDy.

        For each module, swaps only that module from RNN to SINDy while
        keeping all other modules as RNN. Measures the behavioral cross-
        entropy loss change to identify which module causes the largest
        prediction degradation.

        Requires trained SINDy coefficients (``sindy_weight > 0`` during
        training, or a completed Stage 2 ridge refit).

        Also reports the all-RNN baseline and all-SINDy loss for context.

        Args:
            loss_fn: Loss function ``(prediction, target) -> scalar``.
                     Defaults to cross-entropy.

        Returns:
            DataFrame with columns: module, loss_rnn, loss_sindy_swap,
                                    loss_delta, loss_all_sindy
        """
        if loss_fn is None:
            from spice.resources.spice_training import cross_entropy_loss
            loss_fn = cross_entropy_loss

        model = self.model
        device = model.device
        model.eval()

        xs = self.dataset.xs.to(device)
        ys = self.dataset.ys.to(device)

        def _compute_loss(logits):
            # logits: (E, B, T, W, A) from post_forward_pass
            # ys:     (B, T, W, A)
            if logits.dim() == 5:
                logits = logits.mean(dim=0)  # average over ensemble
            # NaN masking (same logic as _run_batch_training)
            mask = ~torch.isnan(xs[..., :model.n_actions].sum(dim=-1))
            return loss_fn(logits[mask], ys[mask]).item()

        was_sindy = model.use_sindy

        # Baseline: all RNN
        model.use_sindy = False
        with torch.no_grad():
            logits_rnn, _ = model(xs)
        loss_rnn = _compute_loss(logits_rnn)

        # All SINDy
        model.use_sindy = True
        with torch.no_grad():
            logits_sindy, _ = model(xs)
        loss_all_sindy = _compute_loss(logits_sindy)

        # Per-module swap: temporarily override call_module so that only
        # the target module uses SINDy, all others remain RNN.
        original_call = model.call_module  # bound method

        results = []
        for target_module in model.submodules_rnn:
            def _make_wrapper(target):
                def wrapper(**kwargs):
                    key_module = kwargs.get('key_module')
                    model.use_sindy = (key_module == target)
                    return original_call(**kwargs)
                return wrapper

            model.call_module = _make_wrapper(target_module)
            with torch.no_grad():
                logits_swap, _ = model(xs)
            swap_loss = _compute_loss(logits_swap)

            results.append({
                'module': target_module,
                'loss_rnn': round(loss_rnn, 6),
                'loss_sindy_swap': round(swap_loss, 6),
                'loss_delta': round(swap_loss - loss_rnn, 6),
                'loss_all_sindy': round(loss_all_sindy, 6),
            })

        # Restore
        del model.call_module  # remove instance override -> class method restored
        model.use_sindy = was_sindy

        return pd.DataFrame(results)

    def embedding_dependence(self) -> pd.DataFrame:
        """Weight-norm analysis of control vs embedding vs state importance.

        For each RNN module, computes the L2 norm of the first-layer
        weights (``weight_linear``) partitioned by input group:

        - **controls**: the SINDy-visible control signals
        - **embedding**: participant/experiment embeddings (SINDy-invisible)
        - **state**: the module's own hidden state

        High ``emb_frac`` indicates the RNN relies on information that
        SINDy cannot access, pointing to a fundamental expressiveness gap.

        Returns:
            DataFrame with columns: module, control_norm, emb_norm,
                                    state_norm, emb_frac
        """
        model = self.model
        results = []

        for module_name in model.submodules_rnn:
            rnn = model.submodules_rnn[module_name]
            n_controls = len(self.config.library_setup[module_name])

            # weight_linear: (E, proj_size, input_size+1)
            # Columns: [controls(n_c), embedding(emb_dim), state(1)]
            W = rnn.weight_linear.detach().cpu().float()
            total_cols = W.shape[2]  # input_size + 1
            emb_dim = total_cols - n_controls - 1  # embedding columns

            # Per-ensemble L2 norm of each group, then average over E
            if n_controls > 0:
                ctrl_norm = W[:, :, :n_controls].norm(dim=(1, 2)).mean().item()
            else:
                ctrl_norm = 0.0

            if emb_dim > 0:
                emb_norm = W[:, :, n_controls:n_controls + emb_dim].norm(dim=(1, 2)).mean().item()
            else:
                emb_norm = 0.0

            state_norm = W[:, :, -1].norm(dim=1).mean().item()

            total_norm = ctrl_norm + emb_norm + state_norm
            emb_frac = emb_norm / total_norm if total_norm > 1e-8 else 0.0

            results.append({
                'module': module_name,
                'control_norm': round(ctrl_norm, 4),
                'emb_norm': round(emb_norm, 4),
                'state_norm': round(state_norm, 4),
                'emb_frac': round(emb_frac, 4),
            })

        return pd.DataFrame(results)

    def state_range(self) -> pd.DataFrame:
        """Distribution statistics of hidden states and updates per module.

        Collects the state entering each module (``h_in``) and the residual
        update (``delta_h = h_out - h_in``) across all trials. Reports
        range, std, and fraction of values outside [-1, 1] and [-2, 2].

        States with large magnitude or high variance degrade polynomial
        approximation quality (higher-degree terms blow up).

        Returns:
            DataFrame with columns: module, metric, min, max, mean, std,
                                    pct_outside_1, pct_outside_2
        """
        hook_data = self._collect_module_data()
        results = []

        for module_name in self.model.submodules_rnn:
            calls = hook_data[module_name]
            if not calls:
                continue

            all_h_in = []
            all_delta = []

            for inputs, state, output in calls:
                E, B, I = output.shape[1], output.shape[2], output.shape[3]
                h_in = state[-1] if state is not None else torch.zeros(E, B, I)
                h_out = output[-1, :, :, :, 0]
                all_h_in.append(h_in.reshape(-1))
                all_delta.append((h_out - h_in).reshape(-1))

            for metric_name, values in [('state_in', torch.cat(all_h_in)),
                                         ('delta_h', torch.cat(all_delta))]:
                results.append({
                    'module': module_name,
                    'metric': metric_name,
                    'min': round(values.min().item(), 4),
                    'max': round(values.max().item(), 4),
                    'mean': round(values.mean().item(), 4),
                    'std': round(values.std().item(), 4),
                    'pct_outside_1': round(
                        (values.abs() > 1).float().mean().item() * 100, 1),
                    'pct_outside_2': round(
                        (values.abs() > 2).float().mean().item() * 100, 1),
                })

        return pd.DataFrame(results)

    def residual_structure(self, degree: Optional[int] = None) -> pd.DataFrame:
        """Correlation between polynomial-fit residuals and input features.

        After fitting an OLS polynomial to each module's dynamics, computes
        the Pearson correlation between the residuals and each library
        input (state, control signals). High correlation indicates
        systematic signal the polynomial cannot capture — suggesting
        missing terms, insufficient degree, or fundamentally non-polynomial
        dynamics.

        Args:
            degree: Polynomial degree. If None, uses each module's
                    configured degree.

        Returns:
            DataFrame with columns: module, feature, correlation, abs_correlation
        """
        hook_data = self._collect_module_data()
        results = []

        for module_name in self.model.submodules_rnn:
            calls = hook_data[module_name]
            if not calls:
                continue

            mod_degree = (degree if degree is not None
                          else self.model.sindy_specs[module_name]['polynomial_degree'])
            n_controls = len(self.config.library_setup[module_name])
            feature_names = list(self.model.sindy_specs[module_name]['input_names'])
            library_terms = self.model.sindy_candidate_terms[module_name]

            all_h_in = []
            all_controls = []
            all_delta = []

            for inputs, state, output in calls:
                W, E, B, I, F = inputs.shape
                h_in = state[-1] if state is not None else torch.zeros(E, B, I)
                controls = (inputs[-1, :, :, :, :n_controls] if n_controls > 0
                            else inputs[-1, :, :, :, :0])
                h_out = output[-1, :, :, :, 0]

                all_h_in.append(h_in.reshape(E * B, I))
                all_controls.append(controls.reshape(E * B, I, n_controls))
                all_delta.append((h_out - h_in).reshape(E * B, I))

            h_in_stacked = torch.stack(all_h_in)        # (T, EB, I)
            controls_stacked = torch.stack(all_controls)  # (T, EB, I, C)
            delta_stacked = torch.stack(all_delta)        # (T, EB, I)

            # Filter inactive items
            I = h_in_stacked.shape[2]
            active_items = [i for i in range(I)
                           if h_in_stacked[:, :, i].var() > 1e-6]
            if not active_items:
                active_items = list(range(I))

            h_in_f = h_in_stacked[:, :, active_items]
            controls_f = controls_stacked[:, :, active_items, :]
            delta_f = delta_stacked[:, :, active_items]

            # OLS polynomial fit
            library_matrix = compute_polynomial_library(
                h_in_f, controls_f, mod_degree, feature_names, library_terms,
            )
            n_terms = library_matrix.shape[-1]
            lib_flat = library_matrix.reshape(-1, n_terms).float()
            dh_flat = delta_f.reshape(-1, 1).float()

            coeffs = torch.linalg.lstsq(lib_flat, dh_flat).solution
            residuals = (dh_flat - lib_flat @ coeffs).squeeze(-1)  # (N,)

            # Correlate residuals with each raw input feature
            h_in_flat = h_in_f.reshape(-1).float()
            raw_features: List[Tuple[str, torch.Tensor]] = [('state', h_in_flat)]
            control_names = list(self.config.library_setup[module_name])
            for c_idx, c_name in enumerate(control_names):
                raw_features.append(
                    (c_name, controls_f[:, :, :, c_idx].reshape(-1).float()))

            for feat_name, feat_values in raw_features:
                if feat_values.std() < 1e-8 or residuals.std() < 1e-8:
                    corr = 0.0
                else:
                    corr = torch.corrcoef(
                        torch.stack([residuals, feat_values])
                    )[0, 1].item()

                results.append({
                    'module': module_name,
                    'feature': feat_name,
                    'correlation': round(corr, 4),
                    'abs_correlation': round(abs(corr), 4),
                })

        return pd.DataFrame(results)

    def summary(self, degree: Optional[int] = None) -> pd.DataFrame:
        """Combined per-module diagnostic report.

        Merges polynomial R², embedding dependence fraction, and state
        range into a single table for quick triage.

        | Column             | What it tells you                       |
        |--------------------|-----------------------------------------|
        | r2                 | Can a polynomial express the dynamics?  |
        | delta_h_std        | Scale of state updates                  |
        | emb_frac           | SINDy-invisible information reliance    |
        | state_max          | Polynomial-unfriendly magnitude?        |
        | state_pct_outside_1| Fraction of states outside [-1, 1]      |

        Args:
            degree: Polynomial degree override (passed to polynomial_adequacy).

        Returns:
            DataFrame indexed by module.
        """
        r2_df = self.polynomial_adequacy(degree=degree)
        emb_df = self.embedding_dependence()
        state_df = self.state_range()

        # Pivot state_range to get per-module state_in stats
        state_in = state_df[state_df['metric'] == 'state_in'].set_index('module')

        rows = []
        for _, r2_row in r2_df.iterrows():
            name = r2_row['module']
            row = {
                'module': name,
                'r2': r2_row['r2'],
                'delta_h_std': r2_row['delta_h_std'],
            }

            emb_match = emb_df[emb_df['module'] == name]
            if not emb_match.empty:
                row['emb_frac'] = emb_match.iloc[0]['emb_frac']

            if name in state_in.index:
                s = state_in.loc[name]
                row['state_max'] = max(abs(s['min']), abs(s['max']))
                row['state_pct_outside_1'] = s['pct_outside_1']

            rows.append(row)

        return pd.DataFrame(rows)
