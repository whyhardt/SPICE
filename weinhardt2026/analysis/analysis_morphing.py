"""Model Morphing Analysis

Morphs participant embeddings along a target metric direction (e.g., avg_reward)
and refits SINDy equations to reveal how cognitive model structure and parameters
change along that dimension.

Pipeline:
    1. Load a fitted SpiceEstimator and extract participant embeddings.
    2. Find the morphing direction in embedding space via linear regression
       of embeddings onto the target metric.
    3. Create an expanded model with P × M virtual participants, where each
       real participant's embedding is shifted by M steps along the direction.
    4. Duplicate the dataset so each virtual participant uses the original
       participant's behavioral trajectory.
    5. Run Stage 2 SINDy refit (with pruning) on the expanded model.
    6. Save the morphed model for downstream plotting/analysis.

Usage:
    from weinhardt2026.analysis.analysis_morphing import run_morphing

    run_morphing(
        estimator=estimator,
        dataset=dataset,
        metric_values=avg_reward_per_participant,  # shape (n_participants,)
        n_steps=20,
        save_path='params/spice_study_morph_reward.pkl',
    )
"""

import os

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from spice import SpiceEstimator
from spice.resources.spice_training import _run_sindy_training
from spice.resources.spice_utils import SpiceDataset


def _find_morphing_direction(estimator: SpiceEstimator, metric_values: np.ndarray) -> tuple:
    """Find the direction in embedding space that best predicts the target metric.

    Args:
        estimator: Fitted SpiceEstimator with participant embeddings.
        metric_values: Target metric per participant, shape (n_participants,).

    Returns:
        direction: Unit vector in embedding space, shape (embedding_dim,).
        projections: Projection of each participant's embedding onto direction, shape (n_participants,).
    """
    if not hasattr(estimator.model, 'participant_embedding'):
        raise ValueError("Model has no participant embeddings.")

    # Access weight tensor directly to avoid EnsembleEmbedding forward-pass shape issues
    emb_weight = estimator.model.participant_embedding.weight.data  # (E, P, D)
    # Use ensemble member 0 for direction finding
    emb_matrix = emb_weight[0].detach().cpu().numpy()  # (P, D)

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
) -> SpiceEstimator:
    """Create a new estimator with expanded participants and morphed embeddings.

    Uses ensemble_size=1 (mean across original ensemble) for memory efficiency.
    The RNN is frozen during Stage 2 and identical across ensemble members, so
    state trajectories don't benefit from multiple members. Pruning uses
    threshold-based pruning instead of CI-based ensemble pruning.

    Args:
        estimator: Original fitted estimator.
        direction: Unit direction vector in embedding space, shape (D,).
        n_steps: Number of morphing steps.
        step_range: (min_projection, max_projection) defining the morphing range.

    Returns:
        New SpiceEstimator with P * n_steps participants (ensemble_size=1).
    """
    model = estimator.model
    n_participants = estimator.n_participants
    n_virtual = n_participants * n_steps
    E_orig = model.ensemble_size

    # Create new estimator with E=1 for memory efficiency
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

    # ── Copy RNN weights from ensemble member 0 ──
    # Each ensemble member is an independently trained model — averaging weights
    # across members produces nonsense. Take member 0 as a coherent single model.
    # Participant-dependent parameters (SINDy coeffs, embeddings, initial values)
    # have different participant counts and are handled by the expansion code below.
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
            # Only ensemble dim differs, rest matches — take member 0
            compatible_state[key] = src_state[key][0:1].clone()
    new_model.load_state_dict(compatible_state, strict=False)

    # ── Morphed embeddings ──
    direction_t = torch.tensor(direction, dtype=torch.float32)
    step_values = torch.linspace(step_range[0], step_range[1], n_steps)

    old_emb = model.participant_embedding.weight.data  # (E_orig, P, D)
    emb_0 = old_emb[0]  # (P, D) — ensemble member 0
    new_emb = torch.zeros(1, n_virtual, emb_0.shape[1])

    for pid in range(n_participants):
        orig_emb = emb_0[pid]  # (D,)
        orig_proj = (orig_emb @ direction_t).item()

        for step in range(n_steps):
            virtual_pid = pid * n_steps + step
            shift = step_values[step] - orig_proj
            new_emb[0, virtual_pid, :] = orig_emb + shift * direction_t

    new_model.participant_embedding.weight.data = new_emb

    # ── Expand SINDy coefficients (from ensemble member 0) ──
    for module in model.get_modules():
        old_coeffs = model.sindy_coefficients[module].data  # (E_orig, P, X, C)
        X, C = old_coeffs.shape[2], old_coeffs.shape[3]
        coeffs_0 = old_coeffs[0]  # (P, X, C)
        new_coeffs = torch.zeros(1, n_virtual, X, C)

        old_presence = model.sindy_coefficients_presence[module]  # (E_orig, P, X, C)
        presence_0 = old_presence[0]  # (P, X, C)
        new_presence = torch.ones(1, n_virtual, X, C, dtype=old_presence.dtype)

        for pid in range(n_participants):
            for step in range(n_steps):
                virtual_pid = pid * n_steps + step
                new_coeffs[0, virtual_pid] = coeffs_0[pid]
                new_presence[0, virtual_pid] = presence_0[pid]

        new_model.sindy_coefficients[module] = torch.nn.Parameter(new_coeffs)
        new_model.sindy_coefficients_presence[module] = new_presence

        # Reset patience counters
        if module in new_model.sindy_pruning_patience_counters:
            new_model.sindy_pruning_patience_counters[module] = torch.zeros(1, n_virtual, X, C)

        # Expand prior masks
        if module in model.sindy_coefficients_prior_mask:
            old_mask = model.sindy_coefficients_prior_mask[module]
            mask_0 = old_mask[0]  # (P, X, C)
            new_mask = torch.ones(1, n_virtual, X, C, dtype=old_mask.dtype)
            for pid in range(n_participants):
                for step in range(n_steps):
                    new_mask[0, pid * n_steps + step] = mask_0[pid]
            new_model.sindy_coefficients_prior_mask[module] = new_mask

        # Expand damping parameters
        if module in model.sindy_damping_raw:
            old_damp = model.sindy_damping_raw[module].data  # (E_orig, P, X)
            damp_0 = old_damp[0]  # (P, X)
            new_damp = torch.zeros(1, n_virtual, X)
            for pid in range(n_participants):
                for step in range(n_steps):
                    new_damp[0, pid * n_steps + step] = damp_0[pid]
            new_model.sindy_damping_raw[module] = torch.nn.Parameter(new_damp)

    return new_estimator


def run_morphing(
    estimator: SpiceEstimator,
    dataset: SpiceDataset,
    metric_values: np.ndarray,
    n_steps: int = 20,
    morphing_range_sd: float = 1.0,
    save_path: str = None,
    verbose: bool = True,
) -> dict:
    """Run the full morphing analysis pipeline.

    Args:
        estimator: Fitted SpiceEstimator with participant embeddings.
        dataset: Original SpiceDataset used for training.
        metric_values: Target metric per participant (e.g., avg_reward), shape (n_participants,).
        n_steps: Number of morphing steps (resolution of the morphing axis).
        morphing_range_sd: Range of morphing in units of SD of the projected embeddings.
            E.g., 1.0 means morph from -1 SD to +1 SD of the projection distribution.
        save_path: Path to save the morphed model (.pkl). If None, does not save.
        verbose: Print progress.

    Returns:
        Dict with keys:
            'estimator': Morphed SpiceEstimator with P * n_steps virtual participants.
            'direction': Unit direction vector in embedding space.
            'step_values': Morphing step values (projection values along direction).
            'n_steps': Number of morphing steps.
            'n_participants': Number of original participants.
            'metric_values': Original metric values per participant.
    """
    n_participants = estimator.n_participants

    if len(metric_values) != n_participants:
        raise ValueError(
            f"metric_values has {len(metric_values)} entries but model has {n_participants} participants."
        )

    # Step 1: Find morphing direction
    if verbose:
        print("Step 1: Finding morphing direction in embedding space...")
    direction, projections = _find_morphing_direction(estimator, metric_values)

    proj_mean = projections.mean()
    proj_std = projections.std()
    step_range = (proj_mean - morphing_range_sd * proj_std,
                  proj_mean + morphing_range_sd * proj_std)
    step_values = np.linspace(step_range[0], step_range[1], n_steps)

    if verbose:
        reg_r2 = LinearRegression().fit(
            projections.reshape(-1, 1), metric_values
        ).score(projections.reshape(-1, 1), metric_values)
        print(f"  Direction R² with metric: {reg_r2:.3f}")
        print(f"  Projection range: [{projections.min():.3f}, {projections.max():.3f}]")
        print(f"  Morphing range: [{step_range[0]:.3f}, {step_range[1]:.3f}] "
              f"(±{morphing_range_sd} SD)")

    # Step 2: Create morphed estimator
    if verbose:
        print(f"\nStep 2: Creating morphed model ({n_participants} × {n_steps} = "
              f"{n_participants * n_steps} virtual participants)...")
    morphed_estimator = _create_morphed_estimator(
        estimator, direction, n_steps, step_range,
    )

    # Step 3: Create expanded dataset
    if verbose:
        print("Step 3: Expanding dataset...")
    morphed_dataset = _create_morphed_dataset(dataset, n_participants, n_steps)

    # Step 4: Run Stage 2 SINDy refit
    if verbose:
        print(f"\nStep 4: Running Stage 2 SINDy refit on morphed model...")
    morphed_model = morphed_estimator.model

    # Prepare 5D training tensors: (1, B, T, W, F) — single ensemble member
    xs = morphed_dataset.xs
    ys = morphed_dataset.ys

    xs_5d = xs.unsqueeze(0)
    ys_5d = ys.unsqueeze(0)

    # With E=1, ensemble CI pruning is not applicable — use threshold pruning.
    # Fall back to the original threshold if set, otherwise use a reasonable default.
    threshold_pruning = estimator.sindy_threshold_pruning
    if threshold_pruning is None:
        threshold_pruning = 0.0

    _run_sindy_training(
        model=morphed_model,
        xs_train=xs_5d,
        ys_train=ys_5d,
        xs_train_original=xs,
        ys_train_original=ys,
        epochs=1000,
        n_warmup_steps=100,
        sindy_alpha=estimator.sindy_alpha,
        sindy_pruning_frequency=estimator.sindy_pruning_frequency,
        sindy_ensemble_pruning=None,  # disabled — E=1
        sindy_threshold_pruning=threshold_pruning,
        shooting_steps=estimator.sindy_shooting_steps,
        sindy_ridge=estimator.sindy_ridge,
        sindy_contraction_weight=estimator.sindy_contraction_weight,
        verbose=verbose,
    )

    # Step 5: Save
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        morphed_estimator.save_spice(save_path)
        if verbose:
            print(f"\nMorphed model saved to: {save_path}")

    result = {
        'estimator': morphed_estimator,
        'direction': direction,
        'step_values': step_values,
        'n_steps': n_steps,
        'n_participants': n_participants,
        'metric_values': metric_values,
    }

    if verbose:
        print("\nMorphing analysis complete.")

    return result


def get_morphed_coefficients(result: dict, estimator_morphed: SpiceEstimator = None) -> dict:
    """Extract per-step averaged coefficients and presence from a morphing result.

    Args:
        result: Dict returned by run_morphing().
        estimator_morphed: Optionally pass a loaded morphed estimator (e.g., from saved .pkl).
            If None, uses result['estimator'].

    Returns:
        Dict with keys per module, each containing:
            'mean_coefficients': (n_steps, n_terms) mean coefficient across participants.
            'inclusion_probability': (n_steps, n_terms) fraction of participants with term present.
            'term_names': List of SINDy term names.
            'step_values': (n_steps,) morphing axis values.
    """
    est = estimator_morphed or result['estimator']
    n_steps = result['n_steps']
    n_participants = result['n_participants']

    coeffs_dict = est.get_sindy_coefficients()
    modules = est.get_modules()
    candidate_terms = est.get_candidate_terms()

    output = {}
    for module in modules:
        coeffs = coeffs_dict[module]  # (E, P_virtual, X, C)
        presence = est.model.sindy_coefficients_presence[module]  # (E, P_virtual, X, C)

        # Use ensemble member 0 (Stage 2 fits all members to consensus targets)
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

        mean_coeffs = np.nanmean(c, axis=0)
        inclusion_prob = (p > 0).astype(float).mean(axis=0)

        output[module] = {
            'mean_coefficients': mean_coeffs,
            'inclusion_probability': inclusion_prob,
            'term_names': candidate_terms[module],
            'step_values': result['step_values'],
        }

    return output
