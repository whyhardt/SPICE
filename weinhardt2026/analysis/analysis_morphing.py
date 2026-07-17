"""Model Morphing Analysis

Morphs participant embeddings along a target metric direction (e.g., avg_reward)
and refits SINDy equations to reveal how cognitive model structure and parameters
change along that dimension.

Runs independently for each ensemble member (each has its own RNN + embedding
space), then aggregates results to produce mean ± SE coefficient curves.

Pipeline (per ensemble member):
    1. Find the morphing direction in that member's embedding space via linear
       regression of embeddings onto the target metric.
    2. Create an expanded model (E=1) with P × M virtual participants, where
       each real participant's embedding is shifted by M steps along the direction.
    3. Duplicate the dataset so each virtual participant uses the original
       participant's behavioral trajectory.
    4. Run Stage 2 SINDy refit (with pruning) on the expanded model.
    5. Extract per-step coefficients and inclusion probabilities.

After all members are processed, aggregate across members for plotting.

Usage:
    from weinhardt2026.analysis.analysis_morphing import run_morphing, get_morphed_coefficients

    result = run_morphing(
        estimator=estimator,
        dataset=dataset,
        metric_values=avg_reward_per_participant,  # shape (n_participants,)
        n_steps=20,
        save_dir='params/morphing/',
    )
    coeffs = get_morphed_coefficients(result)
"""

import math
import os

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from spice import SpiceEstimator
from spice.resources.spice_training import _ridge_solve_sindy
from spice.resources.spice_utils import SpiceDataset


def _find_morphing_direction(
    estimator: SpiceEstimator,
    metric_values: np.ndarray,
    ensemble_member: int = 0,
) -> tuple:
    """Find the direction in embedding space that best predicts the target metric.

    Args:
        estimator: Fitted SpiceEstimator with participant embeddings.
        metric_values: Target metric per participant, shape (n_participants,).
        ensemble_member: Which ensemble member's embeddings to use.

    Returns:
        direction: Unit vector in embedding space, shape (embedding_dim,).
        projections: Projection of each participant's embedding onto direction, shape (n_participants,).
    """
    if not hasattr(estimator.model, 'participant_embedding'):
        raise ValueError("Model has no participant embeddings.")

    # Access weight tensor directly to avoid EnsembleEmbedding forward-pass shape issues
    emb_weight = estimator.model.participant_embedding.weight.data  # (E, P, D)
    emb_matrix = emb_weight[ensemble_member].detach().cpu().numpy()  # (P, D)

    reg = LinearRegression()
    reg.fit(emb_matrix, metric_values)

    direction = reg.coef_  # (D,)
    direction = direction / np.linalg.norm(direction)  # unit vector

    projections = emb_matrix @ direction  # (P,)

    return direction, projections


def _create_morphed_dataset(dataset: SpiceDataset, n_participants: int, n_steps: int) -> SpiceDataset:
    """Duplicate dataset so each participant appears n_steps times with unique IDs.

    Virtual participant layout: for original participant i and morphing step j,
    the virtual ID is i * n_steps + j.

    Args:
        dataset: Original SpiceDataset.
        n_participants: Number of original participants.
        n_steps: Number of morphing steps per participant.

    Returns:
        Expanded SpiceDataset with n_participants * n_steps sessions.
    """
    xs = dataset.xs  # (B, T, W, F)
    ys = dataset.ys  # (B, T, W, A)

    # Group sessions by participant ID
    pid_col = xs[:, 0, 0, -1].long()  # participant ID per session

    xs_parts = []
    ys_parts = []

    for pid in range(n_participants):
        session_mask = pid_col == pid
        xs_pid = xs[session_mask]  # (n_blocks, T, W, F)
        ys_pid = ys[session_mask]  # (n_blocks, T, W, A)

        for step in range(n_steps):
            virtual_pid = pid * n_steps + step
            xs_step = xs_pid.clone()
            xs_step[..., -1] = virtual_pid  # remap participant ID
            xs_parts.append(xs_step)
            ys_parts.append(ys_pid.clone())

    xs_new = torch.cat(xs_parts, dim=0)
    ys_new = torch.cat(ys_parts, dim=0)

    return SpiceDataset(xs_new, ys_new)


def _create_morphed_estimator(
    estimator: SpiceEstimator,
    direction: np.ndarray,
    n_steps: int,
    step_range: tuple,
    ensemble_member: int = 0,
) -> SpiceEstimator:
    """Create a new E=1 estimator from a single ensemble member with morphed embeddings.

    Takes the RNN weights, embeddings, and SINDy coefficients from one specific
    ensemble member of the original estimator. Embeddings are shifted along
    the morphing direction.

    Args:
        estimator: Original fitted estimator.
        direction: Unit direction vector in embedding space, shape (D,).
        n_steps: Number of morphing steps.
        step_range: (min_projection, max_projection) defining the morphing range.
        ensemble_member: Index of the ensemble member to use.

    Returns:
        New SpiceEstimator with P * n_steps participants (ensemble_size=1).
    """
    model = estimator.model
    n_participants = estimator.n_participants
    n_virtual = n_participants * n_steps
    E_orig = model.ensemble_size

    # Create new estimator with E=1
    new_estimator = SpiceEstimator(
        spice_class=estimator.spice_class,
        spice_config=estimator.spice_config,
        n_actions=estimator.n_actions,
        n_items=estimator.n_items,
        n_participants=n_virtual,
        n_experiments=estimator.n_experiments,
        sindy_library_polynomial_degree=estimator.sindy_library_polynomial_degree,
        ensemble_size=1,
        use_sindy=True,
        embedding_size=estimator.embedding_size,
        dropout=estimator.dropout,
        sindy_alpha=estimator.sindy_alpha,
        sindy_pruning_frequency=estimator.sindy_pruning_frequency,
        sindy_threshold_pruning=estimator.sindy_threshold_pruning,
        sindy_shooting_steps=estimator.sindy_shooting_steps,
    )
    new_model = new_estimator.model

    # ── Copy RNN weights from the specified ensemble member ──
    # Participant-dependent parameters (SINDy coeffs, embeddings, initial values)
    # have different participant counts and are handled by the expansion code below.
    e = ensemble_member
    src_state = model.state_dict()
    dst_state = new_model.state_dict()
    compatible_state = {}
    for key in dst_state:
        if key not in src_state:
            continue
        src_shape = src_state[key].shape
        dst_shape = dst_state[key].shape
        if src_shape == dst_shape:
            compatible_state[key] = src_state[key].clone()
        elif (len(src_shape) == len(dst_shape)
              and src_shape[0] == E_orig and dst_shape[0] == 1
              and src_shape[1:] == dst_shape[1:]):
            # Only ensemble dim differs, rest matches — take specified member
            compatible_state[key] = src_state[key][e:e+1].clone()
    new_model.load_state_dict(compatible_state, strict=False)

    # ── Morphed embeddings ──
    direction_t = torch.tensor(direction, dtype=torch.float32)
    step_values = torch.linspace(step_range[0], step_range[1], n_steps)

    old_emb = model.participant_embedding.weight.data  # (E_orig, P, D)
    emb_e = old_emb[e]  # (P, D)
    new_emb = torch.zeros(1, n_virtual, emb_e.shape[1])

    for pid in range(n_participants):
        orig_emb = emb_e[pid]  # (D,)
        orig_proj = (orig_emb @ direction_t).item()

        for step in range(n_steps):
            virtual_pid = pid * n_steps + step
            shift = step_values[step] - orig_proj
            new_emb[0, virtual_pid, :] = orig_emb + shift * direction_t

    new_model.participant_embedding.weight.data = new_emb

    # ── Expand SINDy coefficients from specified ensemble member ──
    for module in model.get_modules():
        old_coeffs = model.sindy_coefficients[module].data  # (E_orig, P, X, C)
        X, C = old_coeffs.shape[2], old_coeffs.shape[3]
        coeffs_e = old_coeffs[e]  # (P, X, C)
        new_coeffs = torch.zeros(1, n_virtual, X, C)

        old_presence = model.sindy_coefficients_presence[module]  # (E_orig, P, X, C)
        presence_e = old_presence[e]  # (P, X, C)
        new_presence = torch.ones(1, n_virtual, X, C, dtype=old_presence.dtype)

        for pid in range(n_participants):
            for step in range(n_steps):
                virtual_pid = pid * n_steps + step
                new_coeffs[0, virtual_pid] = coeffs_e[pid]
                new_presence[0, virtual_pid] = presence_e[pid]

        new_model.sindy_coefficients[module] = torch.nn.Parameter(new_coeffs)
        new_model.sindy_coefficients_presence[module] = new_presence

        # Reset patience counters
        if module in new_model.sindy_pruning_patience_counters:
            new_model.sindy_pruning_patience_counters[module] = torch.zeros(1, n_virtual, X, C)

        # Expand prior masks
        if module in model.sindy_coefficients_prior_mask:
            old_mask = model.sindy_coefficients_prior_mask[module]
            mask_e = old_mask[e]  # (P, X, C)
            new_mask = torch.ones(1, n_virtual, X, C, dtype=old_mask.dtype)
            for pid in range(n_participants):
                for step in range(n_steps):
                    new_mask[0, pid * n_steps + step] = mask_e[pid]
            new_model.sindy_coefficients_prior_mask[module] = new_mask

        # Expand damping parameters
        if module in model.sindy_damping_raw:
            old_damp = model.sindy_damping_raw[module].data  # (E_orig, P, X)
            damp_e = old_damp[e]  # (P, X)
            new_damp = torch.zeros(1, n_virtual, X)
            for pid in range(n_participants):
                for step in range(n_steps):
                    new_damp[0, pid * n_steps + step] = damp_e[pid]
            new_model.sindy_damping_raw[module] = torch.nn.Parameter(new_damp)

    return new_estimator


def _extract_coefficients_single(
    estimator: SpiceEstimator,
    n_participants: int,
    n_steps: int,
) -> dict:
    """Extract per-step averaged coefficients from a single morphed estimator (E=1).

    Returns:
        Dict[module_name -> dict] with:
            'mean_coefficients': (n_steps, n_terms) mean coefficient across participants.
            'inclusion_probability': (n_steps, n_terms) fraction of participants with term present.
    """
    coeffs_dict = estimator.get_sindy_coefficients()
    modules = estimator.get_modules()

    output = {}
    for module in modules:
        coeffs = coeffs_dict[module]  # (1, P_virtual, X, C)
        presence = estimator.model.sindy_coefficients_presence[module]  # (1, P_virtual, X, C)

        c = coeffs[0]    # (P_virtual, X, C)
        p = presence[0]  # (P_virtual, X, C)

        # Average over experiments (X dimension)
        if isinstance(c, np.ndarray):
            c = c.mean(axis=1)
            p = p.astype(float).mean(axis=1)
        else:
            c = c.mean(dim=1)
            p = p.float().mean(dim=1)

        # Convert to numpy
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()

        # Reshape: (P_virtual, C) → (P, M, C) then average across participants → (M, C)
        c = c.reshape(n_participants, n_steps, -1)
        p = p.reshape(n_participants, n_steps, -1)

        output[module] = {
            'mean_coefficients': np.nanmean(c, axis=0),  # (M, C)
            'inclusion_probability': (p > 0).astype(float).mean(axis=0),  # (M, C)
        }

    return output


def _run_morphing_single_member(
    estimator: SpiceEstimator,
    dataset: SpiceDataset,
    metric_values: np.ndarray,
    n_steps: int,
    morphing_range_sd: float,
    ensemble_member: int,
    n_pruning_rounds: int = 20,
    pruning_threshold: float = 0.05,
    alpha: float = 0.01,
    save_path: str = None,
    verbose: bool = True,
) -> dict:
    """Run morphing for a single ensemble member.

    Returns dict with 'direction', 'step_values', 'estimator', 'coefficients'.
    """
    n_participants = estimator.n_participants

    # Step 1: Find morphing direction in this member's embedding space
    direction, projections = _find_morphing_direction(
        estimator, metric_values, ensemble_member=ensemble_member,
    )

    proj_mean = projections.mean()
    proj_std = projections.std()
    step_range = (proj_mean - morphing_range_sd * proj_std,
                  proj_mean + morphing_range_sd * proj_std)
    step_values = np.linspace(step_range[0], step_range[1], n_steps)

    if verbose:
        reg_r2 = LinearRegression().fit(
            projections.reshape(-1, 1), metric_values
        ).score(projections.reshape(-1, 1), metric_values)
        print(f"    Direction R² with metric: {reg_r2:.3f}")

    # Step 2: Create morphed estimator from this member
    morphed_estimator = _create_morphed_estimator(
        estimator, direction, n_steps, step_range,
        ensemble_member=ensemble_member,
    )

    # Step 3: Create expanded dataset (shared across members — same data)
    morphed_dataset = _create_morphed_dataset(dataset, n_participants, n_steps)

    # Step 4: Fast ridge-prune cycle
    # Since we only need coefficient values and sparsity patterns (not a model
    # for prediction), we skip SGD entirely and iterate: ridge → prune → ridge.
    morphed_model = morphed_estimator.model
    xs = morphed_dataset.xs
    ys = morphed_dataset.ys
    xs_5d = xs.unsqueeze(0)
    ys_5d = ys.unsqueeze(0)

    # Reset presence masks for full exploration (respect prior masks)
    for module in morphed_model.get_modules():
        morphed_model.sindy_coefficients_presence[module].fill_(True)
        morphed_model.sindy_coefficients_presence[module] &= (
            morphed_model.sindy_coefficients_prior_mask[module]
        )
        morphed_model.sindy_pruning_patience_counters[module].zero_()

    # Determine pruning threshold
    total_terms = sum(
        morphed_model.sindy_coefficients[m].shape[-1]
        for m in morphed_model.submodules_rnn
    )
    threshold = pruning_threshold

    if verbose:
        print(f"    Ridge-prune cycle: up to {n_pruning_rounds} rounds, "
              f"threshold={threshold}, {total_terms} total terms")

    for round_i in range(n_pruning_rounds):
        # Ridge solve: closed-form optimal coefficients given current sparsity
        success = _ridge_solve_sindy(morphed_model, xs_5d, ys_5d, alpha=alpha)
        if not success:
            if verbose:
                print(f"    Ridge solve failed at round {round_i + 1}")
            break

        # Count total active entries across all participants (not union)
        n_active = sum(
            morphed_model.sindy_coefficients_presence[m].sum().item()
            for m in morphed_model.submodules_rnn
        )

        # Mark terms below threshold (increments patience counter), then prune
        # all terms whose counter reached 1 (i.e., below threshold this round).
        morphed_model.sindy_coefficient_patience(threshold)
        morphed_model.sindy_coefficient_pruning(patience=1, n_terms_pruning=total_terms)

        n_active_after = sum(
            morphed_model.sindy_coefficients_presence[m].sum().item()
            for m in morphed_model.submodules_rnn
        )

        if verbose and (round_i + 1) % max(1, n_pruning_rounds // 10) == 0:
            print(f"      Round {round_i + 1}/{n_pruning_rounds}: "
                  f"{n_active} → {n_active_after} active coefficients")

        pruned_frac = (n_active - n_active_after) / max(n_active, 1)
        if pruned_frac < 0.01:
            if verbose:
                print(f"    Converged at round {round_i + 1} "
                      f"({n_active_after} active coefficients, "
                      f"{pruned_frac:.4%} pruned)")
            break

    # Final ridge solve with the discovered sparsity pattern
    _ridge_solve_sindy(morphed_model, xs_5d, ys_5d, alpha=estimator.sindy_alpha)

    # Save individual member model if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        morphed_estimator.save_spice(save_path)

    # Extract coefficients
    coefficients = _extract_coefficients_single(
        morphed_estimator, n_participants, n_steps,
    )

    return {
        'direction': direction,
        'step_values': step_values,
        'estimator': morphed_estimator,
        'coefficients': coefficients,
    }


def run_morphing(
    estimator: SpiceEstimator,
    dataset: SpiceDataset,
    metric_values: np.ndarray,
    n_steps: int = 20,
    morphing_range_sd: float = 1.0,
    n_pruning_rounds: int = 20,
    pruning_threshold: float = 0.05,
    save_dir: str = None,
    verbose: bool = True,
) -> dict:
    """Run the full morphing analysis pipeline across all ensemble members.

    Each ensemble member has its own RNN and embedding space, so the morphing
    direction is found independently per member. Results are aggregated across
    members to produce mean ± SE coefficient curves.

    Uses a fast ridge-prune cycle instead of SGD:
        ridge → prune → ridge → prune → ... → ridge
    Each round is a closed-form solve, making the entire pipeline orders of
    magnitude faster than SGD-based training.

    Args:
        estimator: Fitted SpiceEstimator with participant embeddings.
        dataset: Original SpiceDataset used for training.
        metric_values: Target metric per participant (e.g., avg_reward), shape (n_participants,).
        n_steps: Number of morphing steps (resolution of the morphing axis).
        morphing_range_sd: Range of morphing in units of SD of the projected embeddings.
        n_pruning_rounds: Number of ridge-prune iterations for sparsity discovery.
        pruning_threshold: Minimum coefficient magnitude to survive pruning.
        save_dir: Directory to save per-member morphed models. If None, does not save.
        verbose: Print progress.

    Returns:
        Dict with keys:
            'member_results': List of per-member result dicts.
            'n_ensemble_members': Number of ensemble members processed.
            'n_steps': Number of morphing steps.
            'n_participants': Number of original participants.
            'metric_values': Original metric values per participant.
            'modules': Tuple of module names.
            'candidate_terms': Dict of term names per module.
    """
    n_participants = estimator.n_participants
    E = estimator.model.ensemble_size

    if len(metric_values) != n_participants:
        raise ValueError(
            f"metric_values has {len(metric_values)} entries but model has {n_participants} participants."
        )

    if verbose:
        print(f"Running morphing analysis across {E} ensemble members...")
        print(f"  Participants: {n_participants}, Steps: {n_steps}, "
              f"Range: ±{morphing_range_sd} SD\n")

    member_results = []
    for e in range(E):
        if verbose:
            print(f"  ── Ensemble member {e+1}/{E} ──")

        save_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'morphed_member_{e}.pkl')

        result_e = _run_morphing_single_member(
            estimator=estimator,
            dataset=dataset,
            metric_values=metric_values,
            n_steps=n_steps,
            morphing_range_sd=morphing_range_sd,
            ensemble_member=e,
            n_pruning_rounds=n_pruning_rounds,
            pruning_threshold=pruning_threshold,
            save_path=save_path,
            verbose=verbose,
        )
        member_results.append(result_e)

        if verbose:
            print()

    if verbose:
        print(f"Morphing analysis complete ({E} members).")

    return {
        'member_results': member_results,
        'n_ensemble_members': E,
        'n_steps': n_steps,
        'n_participants': n_participants,
        'metric_values': metric_values,
        'modules': estimator.get_modules(),
        'candidate_terms': estimator.get_candidate_terms(),
    }


def get_morphed_coefficients(result: dict) -> dict:
    """Aggregate per-step coefficients across ensemble members.

    Args:
        result: Dict returned by run_morphing().

    Returns:
        Dict with keys per module, each containing:
            'mean_coefficients': (n_steps, n_terms) mean across participants and ensemble members.
            'se_coefficients': (n_steps, n_terms) SE across ensemble members.
            'inclusion_probability': (n_steps, n_terms) mean inclusion probability.
            'se_inclusion_probability': (n_steps, n_terms) SE of inclusion probability.
            'term_names': List of SINDy term names.
            'step_values': (n_steps,) morphing axis values (from member 0).
            'all_coefficients': (E, n_steps, n_terms) per-member coefficient curves.
            'all_inclusion_probability': (E, n_steps, n_terms) per-member IP curves.
    """
    member_results = result['member_results']
    E = result['n_ensemble_members']
    modules = result['modules']
    candidate_terms = result['candidate_terms']

    # Use member 0's step_values as reference (all members use same SD-based range
    # but may have slightly different projection distributions)
    step_values = member_results[0]['step_values']

    output = {}
    for module in modules:
        # Stack per-member results: (E, n_steps, n_terms)
        all_mc = np.stack([
            member_results[e]['coefficients'][module]['mean_coefficients']
            for e in range(E)
        ], axis=0)
        all_ip = np.stack([
            member_results[e]['coefficients'][module]['inclusion_probability']
            for e in range(E)
        ], axis=0)

        output[module] = {
            'mean_coefficients': all_mc.mean(axis=0),
            'se_coefficients': all_mc.std(axis=0) / np.sqrt(E) if E > 1 else np.zeros_like(all_mc[0]),
            'inclusion_probability': all_ip.mean(axis=0),
            'se_inclusion_probability': all_ip.std(axis=0) / np.sqrt(E) if E > 1 else np.zeros_like(all_ip[0]),
            'term_names': candidate_terms[module],
            'step_values': step_values,
            'all_coefficients': all_mc,
            'all_inclusion_probability': all_ip,
        }

    return output
