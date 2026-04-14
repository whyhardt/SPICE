"""
Value trajectory visualization for SPICE models.

Visualizes model predictions (action probabilities, logits) and internal
value trajectories across time for SPICE-RNN and SPICE modes.

The function creates a multi-row plot showing:
  1. Action probabilities (row 1)
  2. Logits (row 2)
  3. Value trajectories for each memory state (remaining rows)

Each plot shows:
  - SPICE-RNN (use_sindy=False) in one color
  - SPICE (use_sindy=True) in another color
  - Optional benchmark and GRU models (action probs and logits only)

Usage examples:

  # From Python:
  from weinhardt2026.analysis.analysis_value_trajectories import (
      plot_value_trajectories,
  )

  plot_value_trajectories(
      dataset=test_data,
      spice_model=estimator,
      benchmark_model=benchmark,
      gru_model=gru,
      output_path="output/trajectories.png",
      participant_id=0,
      block_id=1,
      action_idx=0,
  )
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from spice import SpiceEstimator, SpiceDataset


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _get_session_idx(
    dataset: SpiceDataset,
    participant_id: int,
    block_id: int,
) -> int:
    """Convert participant ID and block ID to session index.

    Args:
        dataset: SpiceDataset
        participant_id: Participant ID from dataset (0-indexed after csv_to_dataset remapping)
        block_id: Block ID from dataset (raw from CSV, often 1-indexed)

    Returns:
        session_idx: Flat session index in the dataset
    """
    # Get metadata (participant IDs and block IDs)
    participant_ids = dataset.xs[:, 0, 0, -1].cpu().numpy()  # (n_sessions,)
    block_ids = dataset.xs[:, 0, 0, -3].cpu().numpy()  # (n_sessions,)

    # Find session that matches both participant and block
    session_mask = (participant_ids == participant_id) & (block_ids == block_id)
    matching_sessions = np.where(session_mask)[0]

    if len(matching_sessions) == 0:
        raise ValueError(f"No session found for participant_id={participant_id}, block_id={block_id}")

    session_idx = matching_sessions[0]

    return int(session_idx)


@torch.no_grad()
def extract_trajectories(
    dataset: SpiceDataset,
    spice_model: SpiceEstimator,
    participant_id: int = 0,
    block_id: int = 1,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Extract predictions and internal states for SPICE-RNN and SPICE.

    Args:
        dataset: Input dataset (SpiceDataset)
        spice_model: Trained SPICE model
        participant_id: Participant ID from dataset (0-indexed, default: 0)
        block_id: Block ID from dataset (raw from CSV, default: 1)

    Returns:
        rnn_data: dict with keys 'logits', 'probs', 'states'
        spice_data: dict with keys 'logits', 'probs', 'states'

    States dict has shape (T, n_items) for each memory state key.
    """
    spice_model.model.eval()
    device = spice_model.device

    # Convert participant + block to session index
    session_idx = _get_session_idx(dataset, participant_id, block_id)

    # Extract single session
    xs_session = dataset.xs[session_idx:session_idx+1].to(device)  # (1, T, W, F)

    # --- RNN mode ---
    spice_model.use_sindy(False)
    spice_model.model.init_state(batch_size=1)

    rnn_logits, _ = spice_model(xs_session)  # (E, 1, T, W, A) or (1, T, W, A)

    # Collect state trajectories during forward pass
    # We need to run forward again while capturing states
    spice_model.model.init_state(batch_size=1)
    rnn_states = _extract_state_trajectory(spice_model.model, xs_session)

    # --- SPICE mode ---
    spice_model.use_sindy(True)
    spice_model.model.init_state(batch_size=1)

    spice_logits, _ = spice_model(xs_session)  # (E, 1, T, W, A) or (1, T, W, A)

    spice_model.model.init_state(batch_size=1)
    spice_states = _extract_state_trajectory(spice_model.model, xs_session)

    # Process logits (average over ensemble if present)
    if rnn_logits.dim() == 5:  # (E, B, T, W, A)
        rnn_logits = rnn_logits.mean(dim=0)  # (B, T, W, A)
        spice_logits = spice_logits.mean(dim=0)

    # Squeeze session dimension and within-trial dimension
    rnn_logits = rnn_logits[0, :, 0, :].cpu()  # (T, A)
    spice_logits = spice_logits[0, :, 0, :].cpu()  # (T, A)

    # Compute probabilities
    rnn_probs = torch.softmax(rnn_logits, dim=-1)  # (T, A)
    spice_probs = torch.softmax(spice_logits, dim=-1)  # (T, A)

    rnn_data = {
        'logits': rnn_logits,
        'probs': rnn_probs,
        'states': rnn_states,
    }

    spice_data = {
        'logits': spice_logits,
        'probs': spice_probs,
        'states': spice_states,
    }

    return rnn_data, spice_data


def _extract_state_trajectory(model, xs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run forward pass and capture state trajectory.

    This function iterates through trials one at a time, running the model's
    forward pass cumulatively (trials 0:t) and capturing the state after each trial.

    Args:
        model: BaseModel instance
        xs: Input tensor (B=1, T, W, F) in batch-first format

    Returns:
        states: dict mapping state_key -> (T, n_items) tensor
    """
    # Store states at each timestep
    state_history = {key: [] for key in model.spice_config.memory_state.keys()}

    T = xs.shape[1]

    # Process cumulatively: run forward on [0:t] and capture state at t
    for t in range(T):
        # Re-initialize state for each cumulative run
        model.init_state(batch_size=1)

        # Get input up to and including trial t: (1, t+1, W, F)
        xs_cumulative = xs[:, :t+1, :, :]

        # Forward pass for trials 0:t
        _ = model(xs_cumulative, model.get_state())

        # Capture current state values (W, E, B, I)
        current_state = model.get_state()
        for key in state_history.keys():
            # Extract (W=1, E, B=1, I) -> (I,)
            # Average over ensemble dimension if present
            if current_state[key].shape[1] > 1:  # ensemble dimension exists
                state_val = current_state[key][0, :, 0, :].mean(dim=0).cpu()
            else:
                state_val = current_state[key][0, 0, 0, :].cpu()
            state_history[key].append(state_val)

    # Stack into (T, I) tensors
    states = {
        key: torch.stack(vals, dim=0) for key, vals in state_history.items()
    }

    return states


@torch.no_grad()
def extract_benchmark_predictions(
    dataset: SpiceDataset,
    benchmark_model: torch.nn.Module,
    participant_id: int = 0,
    block_id: int = 1,
) -> Dict[str, torch.Tensor]:
    """Extract predictions from benchmark model.

    Args:
        dataset: Input dataset
        benchmark_model: Hand-coded cognitive model
        participant_id: Participant ID from dataset (0-indexed)
        block_id: Block ID from dataset (raw from CSV)

    Returns:
        data: dict with keys 'logits', 'probs'
    """
    benchmark_model.eval()

    session_idx = _get_session_idx(dataset, participant_id, block_id)
    xs_session = dataset.xs[session_idx:session_idx+1]

    logits, _ = benchmark_model(xs_session)
    logits = logits[0, :, 0, :].cpu()  # (T, A)

    probs = torch.softmax(logits, dim=-1)

    return {'logits': logits, 'probs': probs}


@torch.no_grad()
def extract_gru_predictions(
    dataset: SpiceDataset,
    gru_model: torch.nn.Module,
    participant_id: int = 0,
    block_id: int = 1,
) -> Dict[str, torch.Tensor]:
    """Extract predictions from GRU model.

    Args:
        dataset: Input dataset
        gru_model: GRU baseline model
        participant_id: Participant ID from dataset (0-indexed)
        block_id: Block ID from dataset (raw from CSV)

    Returns:
        data: dict with keys 'logits', 'probs'
    """
    gru_model.eval()

    session_idx = _get_session_idx(dataset, participant_id, block_id)
    xs_session = dataset.xs[session_idx:session_idx+1]

    logits, _ = gru_model(xs_session)
    logits = logits[0, :, 0, :].cpu()  # (T, A)

    probs = torch.softmax(logits, dim=-1)

    return {'logits': logits, 'probs': probs}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_value_trajectories(
    dataset: SpiceDataset,
    spice_model: SpiceEstimator = None,
    benchmark_model: torch.nn.Module = None,
    gru_model: torch.nn.Module = None,
    participant_id: int = 0,
    block_id: int = 1,
    action_idx: int = 0,
    output_path: str = None,
    figsize: Tuple[int, int] = None,
    show_trials: int = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot action probabilities, logits, and value trajectories.

    Creates a multi-row figure with:
      - Row 1: Action probabilities (for specified action)
      - Row 2: Logits (for specified action)
      - Rows 3+: Value trajectories for each memory state

    Args:
        dataset: Input dataset (SpiceDataset)
        spice_model: Trained SPICE estimator
        benchmark_model: Optional benchmark model
        gru_model: Optional GRU baseline
        participant_id: Participant ID from dataset (0-indexed, default: 0)
        block_id: Block ID from dataset (raw from CSV, default: 1)
        action_idx: Which action to plot probabilities for (default: 0)
        output_path: Save path (if None, displays interactively)
        figsize: Figure size (auto-computed if None)
        show_trials: Number of trials to show (None = all)
        dpi: DPI for saved figure

    Returns:
        fig: matplotlib Figure object
    """

    # Extract data
    rnn_data, spice_data = None, None
    if spice_model is not None:
        rnn_data, spice_data = extract_trajectories(dataset, spice_model, participant_id, block_id)

    benchmark_data = None
    if benchmark_model is not None:
        benchmark_data = extract_benchmark_predictions(dataset, benchmark_model, participant_id, block_id)

    gru_data = None
    if gru_model is not None:
        gru_data = extract_gru_predictions(dataset, gru_model, participant_id, block_id)

    # Get session index for extracting actual actions
    session_idx = _get_session_idx(dataset, participant_id, block_id)

    # Get actual actions from dataset
    actual_actions = dataset.ys[session_idx, :, 0, :].cpu()  # (T, A)
    actual_action_idx = torch.argmax(actual_actions, dim=-1)  # (T,)

    # Determine number of rows
    n_value_states = len(rnn_data['states'])
    n_rows = 2 + n_value_states  # probs + logits + states

    # Auto-compute figure size
    if figsize is None:
        figsize = (12, 3 * n_rows)

    # Determine trial range
    T = rnn_data['probs'].shape[0]
    if show_trials is None:
        show_trials = T
    trial_range = slice(0, min(show_trials, T))

    # Create figure
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = [axes]

    # Time axis
    time_steps = np.arange(show_trials)

    # Color palette
    colors = {
        'spice_rnn': '#2E86AB',      # Blue
        'spice': '#A23B72',           # Purple
        'benchmark': '#F18F01',       # Orange
        'gru': '#6A994E',             # Green
        'actual': '#000000',          # Black
    }

    # ----- Row 1: Action probabilities -----
    ax = axes[0]

    if spice_data is not None:
        ax.plot(time_steps, rnn_data['probs'][trial_range, action_idx].numpy(),
                label='SPICE-RNN', color=colors['spice_rnn'], linewidth=1.5, alpha=0.8)
        ax.plot(time_steps, spice_data['probs'][trial_range, action_idx].numpy(),
                label='SPICE', color=colors['spice'], linewidth=1.5, alpha=0.8)

    if benchmark_data is not None:
        ax.plot(time_steps, benchmark_data['probs'][trial_range, action_idx].numpy(),
                label='Benchmark', color=colors['benchmark'], linewidth=1.5, linestyle='--', alpha=0.7)

    if gru_data is not None:
        ax.plot(time_steps, gru_data['probs'][trial_range, action_idx].numpy(),
                label='GRU', color=colors['gru'], linewidth=1.5, linestyle='--', alpha=0.7)

    # Mark actual choices
    actual_mask = (actual_action_idx[trial_range] == action_idx).numpy()
    if actual_mask.any():
        ax.scatter(time_steps[actual_mask],
                  rnn_data['probs'][trial_range, action_idx].numpy()[actual_mask],
                  color=colors['actual'], s=30, alpha=0.5, marker='o',
                  label='Actual choice', zorder=10)

    ax.set_ylabel(f'P(Action {action_idx})', fontsize=10)
    ax.set_title(f'Action Probabilities - Participant {participant_id}, Block {block_id}',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # ----- Row 2: Logits -----
    ax = axes[1]

    ax.plot(time_steps, rnn_data['logits'][trial_range, action_idx].numpy(),
            label='SPICE-RNN', color=colors['spice_rnn'], linewidth=1.5, alpha=0.8)
    ax.plot(time_steps, spice_data['logits'][trial_range, action_idx].numpy(),
            label='SPICE', color=colors['spice'], linewidth=1.5, alpha=0.8)

    if benchmark_data is not None:
        ax.plot(time_steps, benchmark_data['logits'][trial_range, action_idx].numpy(),
                label='Benchmark', color=colors['benchmark'], linewidth=1.5, linestyle='--', alpha=0.7)

    if gru_data is not None:
        ax.plot(time_steps, gru_data['logits'][trial_range, action_idx].numpy(),
                label='GRU', color=colors['gru'], linewidth=1.5, linestyle='--', alpha=0.7)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.set_ylabel(f'Logit(Action {action_idx})', fontsize=10)
    ax.set_title('Logits', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ----- Rows 3+: Value trajectories -----
    for i, state_key in enumerate(rnn_data['states']):
        ax = axes[2 + i]
        
        rnn_state_vals = rnn_data['states'][state_key]
        spice_state_vals = spice_data['states'][state_key]
        
        # Check dimensionality
        if rnn_state_vals.dim() == 1:
            # Single value per timestep
            ax.plot(time_steps, rnn_state_vals[trial_range].numpy(),
                   label='SPICE-RNN', color=colors['spice_rnn'], linewidth=1.5, alpha=0.8)
            ax.plot(time_steps, spice_state_vals[trial_range].numpy(),
                   label='SPICE', color=colors['spice'], linewidth=1.5, alpha=0.8)
        else:
            # Multiple items (n_items > 1)
            n_items = rnn_state_vals.shape[1]
            for item_idx in range(n_items):
                item_suffix = f' (Item {item_idx})' if n_items > 1 else ''

                ax.plot(time_steps, rnn_state_vals[trial_range, item_idx].numpy(),
                       label=f'SPICE-RNN{item_suffix}', color=colors['spice_rnn'],
                       linewidth=1.5, alpha=0.8, linestyle=['-', '--', '-.', ':'][item_idx % 4])
                ax.plot(time_steps, spice_state_vals[trial_range, item_idx].numpy(),
                       label=f'SPICE{item_suffix}', color=colors['spice'],
                       linewidth=1.5, alpha=0.8, linestyle=['-', '--', '-.', ':'][item_idx % 4])

        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
        ax.set_ylabel(f'{state_key}', fontsize=10)
        ax.set_title(f'Value: {state_key}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    # X-axis label on bottom plot
    axes[-1].set_xlabel('Trial', fontsize=11)

    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Multi-session comparison
# ---------------------------------------------------------------------------

def plot_value_trajectories_multi(
    dataset: SpiceDataset,
    spice_model: SpiceEstimator,
    participant_block_pairs: List[Tuple[int, int]],
    action_idx: int = 0,
    output_dir: str = None,
    figsize: Tuple[int, int] = None,
    show_trials: int = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """Plot value trajectories for multiple participant-block combinations.

    Args:
        dataset: Input dataset
        spice_model: Trained SPICE estimator
        participant_block_pairs: List of (participant_id, block_id) tuples
        action_idx: Which action to plot probabilities for
        output_dir: Directory to save plots (if None, displays interactively)
        figsize: Figure size (auto-computed if None)
        show_trials: Number of trials to show per session
        dpi: DPI for saved figures

    Returns:
        figs: List of matplotlib Figure objects
    """
    figs = []

    for participant_id, block_id in participant_block_pairs:
        output_path = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"trajectories_p{participant_id}_b{block_id}.png"
            )

        fig = plot_value_trajectories(
            dataset=dataset,
            spice_model=spice_model,
            participant_id=participant_id,
            block_id=block_id,
            action_idx=action_idx,
            output_path=output_path,
            figsize=figsize,
            show_trials=show_trials,
            dpi=dpi,
        )

        figs.append(fig)

    return figs
