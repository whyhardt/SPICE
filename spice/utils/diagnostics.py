"""
Diagnostic toolkit for analyzing SPICE model architectural bottlenecks.

All diagnostics work by externally probing fitted model modules
via forward hooks — no modifications to the SPICE backend required.

Usage:
    from spice.utils.diagnostics import SpiceDiagnostics

    diag = SpiceDiagnostics(estimator, dataset)

    # Per-module polynomial R² test
    r2_report = diag.polynomial_adequacy()

    # Per-module GRU gate distributions
    gate_report = diag.gate_saturation()
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
            model(xs, batch_first=True)
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
