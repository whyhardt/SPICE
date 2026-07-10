"""
Prototype: Dynamics trace — trial-by-trial latent state trajectories.

Shows RNN vs SINDy (symbolic) state traces for a single participant,
overlaid on actual choice behavior. Demonstrates that the sparse
symbolic equations capture the same dynamics as the full RNN.
"""

import os
import sys
import importlib
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spice import SpiceEstimator, SpiceConfig, SpiceDataset, csv_to_dataset


def extract_traces(estimator, dataset, participant_idx=0, session_idx=None):
    """Run forward pass twice (RNN mode and SINDy mode) and collect
    trial-by-trial state traces for a single participant/session.

    Returns:
        rnn_states: dict of state_name -> (T,) arrays
        sindy_states: dict of state_name -> (T,) arrays
        actions: (T,) array of chosen action indices
        rewards: (T,) array of received rewards
    """
    xs = dataset.xs
    ys = dataset.ys

    # Find sessions belonging to this participant
    participant_ids = xs[:, 0, 0, -1].int()
    participant_mask = participant_ids == participant_idx
    sessions = torch.where(participant_mask)[0]

    if session_idx is not None:
        session = sessions[session_idx]
    else:
        session = sessions[0]  # first session

    xs_single = xs[session:session+1]  # (1, T, W, F)
    ys_single = ys[session:session+1]

    # Determine valid trials (not NaN-padded)
    valid_mask = ~torch.isnan(xs_single[0, :, 0, 0])
    n_valid = valid_mask.sum().item()

    n_actions = dataset.n_actions
    actions = xs_single[0, :n_valid, 0, :n_actions].argmax(dim=-1).numpy()
    # Rewards: pick the reward of the chosen action
    reward_cols = xs_single[0, :n_valid, 0, n_actions:2*n_actions]
    rewards = np.array([
        reward_cols[t, actions[t]].item() if not torch.isnan(reward_cols[t, actions[t]]) else 0.0
        for t in range(n_valid)
    ])

    model = estimator.model
    state_names = list(model.spice_config.memory_state.keys())

    def _run_and_collect(use_sindy_mode):
        """Forward pass collecting logits per trial."""
        model.use_sindy = use_sindy_mode
        model.eval()

        with torch.no_grad():
            logits_out, _ = model(xs_single)

        # logits shape: (E, B, T, W, A) or (T, W, E, B, A)
        if logits_out.dim() == 5:
            if model.batch_first:
                # (E, B, T, W, A) -> average over ensemble -> (T, A)
                logits_np = logits_out[:, 0, :n_valid, 0, :].mean(dim=0).detach().cpu().numpy()
            else:
                logits_np = logits_out[:n_valid, 0, :, 0, :].mean(dim=1).detach().cpu().numpy()
        else:
            logits_np = logits_out[0, :n_valid, 0, :].detach().cpu().numpy()

        return logits_np

    rnn_logits = _run_and_collect(False)
    sindy_logits = _run_and_collect(True)

    return rnn_logits, sindy_logits, actions, rewards, n_valid


def plot_dynamics_trace(
    estimator_or_path,
    dataset_or_path,
    model_module=None,
    participant_idx=0,
    session_idx=None,
    output_path='figures/prototype_dynamics_trace',
    title='Discovered Dynamics',
    dataset_kwargs=None,
):
    """Create dynamics trace plot."""
    # Load model if path given
    if isinstance(estimator_or_path, str):
        if model_module is None:
            raise ValueError("model_module required when passing path")
        mod = importlib.import_module(model_module)
        rnn_class = mod.SpiceModel
        spice_config = mod.CONFIG

        dataset_kwargs = dataset_kwargs or {}
        dataset = csv_to_dataset(file=dataset_or_path, **dataset_kwargs)
        dataset.normalize_rewards()

        n_actions = dataset.ys.shape[-1]
        n_participants = len(dataset.xs[..., -1].int().unique())

        ckpt = torch.load(estimator_or_path, map_location="cpu")
        first_mod = next(iter(spice_config.library_setup))
        ensemble_size = ckpt["model"][f"sindy_coefficients.{first_mod}"].shape[0]
        del ckpt

        estimator = SpiceEstimator(
            spice_class=rnn_class,
            spice_config=spice_config,
            n_actions=n_actions,
            n_participants=n_participants,
            sindy_library_polynomial_degree=2,
            ensemble_size=ensemble_size,
            use_sindy=True,
        )
        estimator.load_spice(estimator_or_path)
    else:
        estimator = estimator_or_path
        if isinstance(dataset_or_path, str):
            dataset_kwargs = dataset_kwargs or {}
            dataset = csv_to_dataset(file=dataset_or_path, **dataset_kwargs)
            dataset.normalize_rewards()
        else:
            dataset = dataset_or_path

    rnn_logits, sindy_logits, actions, rewards, n_valid = extract_traces(
        estimator, dataset, participant_idx, session_idx
    )

    n_actions = rnn_logits.shape[1]
    trials = np.arange(n_valid)

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 2]})

    # Panel 1: Actual choices and rewards
    ax = axes[0]
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')

    # Choice markers
    for a in range(n_actions):
        mask = actions == a
        ax.scatter(trials[mask], np.full(mask.sum(), a),
                   c='#2ca02c' if a == 0 else '#d62728',
                   marker='|', s=30, linewidths=1.5, zorder=3)
    ax.set_yticks(range(n_actions))
    ax.set_yticklabels([f'Action {a}' for a in range(n_actions)], fontsize=9)
    ax.set_ylabel('Choice', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Reward as background shading
    ax_r = ax.twinx()
    ax_r.fill_between(trials, rewards, alpha=0.15, color='gold', step='mid')
    ax_r.set_ylabel('Reward', fontsize=9, color='goldenrod')
    ax_r.set_ylim(-0.1, 1.5)
    ax_r.spines['top'].set_visible(False)
    ax_r.tick_params(axis='y', colors='goldenrod')

    # Panel 2: RNN logits (action probabilities)
    ax = axes[1]
    rnn_probs = np.exp(rnn_logits) / np.exp(rnn_logits).sum(axis=1, keepdims=True)
    for a in range(n_actions):
        ax.plot(trials, rnn_probs[:, a], linewidth=1.5, alpha=0.8,
                color='#2ca02c' if a == 0 else '#d62728',
                linestyle='--', label=f'RNN P(action {a})')
    ax.set_ylabel('P(action)', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.7)
    ax.set_title('RNN Dynamics', fontsize=10, loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15)

    # Panel 3: SINDy logits (action probabilities)
    ax = axes[2]
    sindy_probs = np.exp(sindy_logits) / np.exp(sindy_logits).sum(axis=1, keepdims=True)
    for a in range(n_actions):
        ax.plot(trials, sindy_probs[:, a], linewidth=1.5, alpha=0.8,
                color='#2ca02c' if a == 0 else '#d62728',
                label=f'SPICE-EQ P(action {a})')
    ax.set_ylabel('P(action)', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Trial', fontsize=10)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.7)
    ax.set_title('Symbolic (SPICE-EQ) Dynamics', fontsize=10, loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(f'{output_path}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{output_path}.pdf', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}.png/.pdf")


if __name__ == '__main__':
    plot_dynamics_trace(
        estimator_or_path='weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019.pkl',
        dataset_or_path='weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv',
        model_module='spice.precoded.workingmemory',
        participant_idx=0,
        output_path='figures/prototype_dynamics_trace_dezfouli2019',
        title='Discovered Dynamics — Participant 0 (Dezfouli 2019)',
    )
