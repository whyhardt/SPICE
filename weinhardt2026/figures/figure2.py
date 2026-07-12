"""
Figure 2 Prototype: Equation Showcase

Panels (saved individually, one per cluster-representative participant):
  a) Discovered equations (rendered as text) — per participant
  b) Actual choices and rewards — per participant
  c) RNN vs SPICE-EQ action probabilities (overlaid) — per participant

Each panel saved as detailed + clean versions in figures/figure2/.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from spice import SpiceEstimator, csv_to_dataset
from weinhardt2026.figures.panel_utils import save_panel


# ---------------------------------------------------------------------------
# Equation formatting
# ---------------------------------------------------------------------------

def _format_equation_lines(estimator, participant_id, max_terms=5, min_coef=0.03):
    """Extract per-module equation strings for a participant."""
    model = estimator.model
    coefs_dict = model.get_sindy_coefficients(aggregate=True)
    modules = model.get_modules()
    terms_dict = model.sindy_candidate_terms

    lines = []
    for mod in modules:
        coefs = coefs_dict[mod][participant_id, 0].detach().cpu().numpy()
        terms = terms_dict[mod]

        term_entries = []
        for i, term in enumerate(terms):
            c = float(coefs[i])
            if term == mod:
                c += 1.0
            if abs(c) < min_coef:
                continue

            c_str = f"{c:+.2f}"
            t_display = term.replace("[t]", "").replace("[t-1]", "\u208b\u2081").replace("[t-2]", "\u208b\u2082").replace("[t-3]", "\u208b\u2083")

            if term == "1":
                entry = c_str
            elif term == mod:
                entry = f"{c_str}\u00b7{_short_mod(mod)}\u209c"
            else:
                t_display = _short_term(t_display)
                entry = f"{c_str}\u00b7{t_display}"

            term_entries.append((abs(c), entry))

        term_entries.sort(key=lambda x: x[0], reverse=True)
        kept = term_entries[:max_terms]
        n_omitted = len(term_entries) - len(kept)

        if not kept:
            eq = "0"
        else:
            eq = " ".join(e[1] for e in kept)
            if n_omitted > 0:
                eq += f" (+{n_omitted})"
            if eq.startswith("+"):
                eq = eq[1:]

        lhs = f"\u0394{_short_mod(mod)}"
        lines.append((lhs, eq))

    return lines


def _short_mod(mod_name):
    mapping = {
        'value_reward_chosen': 'V\u1d63(ch)',
        'value_reward_not_chosen': 'V\u1d63(unch)',
        'value_choice_chosen': 'V\u1d6a(ch)',
        'value_choice_not_chosen': 'V\u1d6a(unch)',
    }
    return mapping.get(mod_name, mod_name)


def _short_term(term):
    term = term.replace('value_reward_chosen', 'V\u1d63(ch)')
    term = term.replace('value_reward_not_chosen', 'V\u1d63(unch)')
    term = term.replace('value_choice_chosen', 'V\u1d6a(ch)')
    term = term.replace('value_choice_not_chosen', 'V\u1d6a(unch)')
    term = term.replace('reward', 'r')
    term = term.replace('choice', 'c')
    return term


# ---------------------------------------------------------------------------
# Dynamics extraction
# ---------------------------------------------------------------------------

def _extract_session_data(dataset, participant_idx, session_idx=None):
    """Get choices, rewards, and valid trial count for a single session."""
    xs = dataset.xs
    n_actions = dataset.n_actions

    pids = xs[:, 0, 0, -1].int()
    sessions = torch.where(pids == participant_idx)[0]

    if session_idx is None:
        lengths = []
        for s in sessions:
            valid = (~torch.isnan(xs[s, :, 0, 0])).sum().item()
            lengths.append(valid)
        session = sessions[np.argmax(lengths)]
    else:
        session = sessions[min(session_idx, len(sessions) - 1)]

    xs_single = xs[session:session + 1]

    valid_mask = ~torch.isnan(xs_single[0, :, 0, 0])
    n_valid = valid_mask.sum().item()

    actions = xs_single[0, :n_valid, 0, :n_actions].argmax(dim=-1).numpy()
    reward_cols = xs_single[0, :n_valid, 0, n_actions:2 * n_actions]
    rewards = np.array([
        reward_cols[t, actions[t]].item()
        if not torch.isnan(reward_cols[t, actions[t]]) else 0.0
        for t in range(n_valid)
    ])

    return xs_single, actions, rewards, n_valid


def _run_forward(model, xs_single, n_valid, use_sindy):
    """Run model forward pass and extract action probabilities."""
    model.use_sindy = use_sindy
    model.eval()

    with torch.no_grad():
        logits_out, _ = model(xs_single)

    if logits_out.dim() == 5:
        if model.batch_first:
            logits = logits_out[:, 0, :n_valid, 0, :].mean(dim=0)
        else:
            logits = logits_out[:n_valid, 0, :, 0, :].mean(dim=1)
    else:
        logits = logits_out[0, :n_valid, 0, :]

    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# Panel plotting functions
# ---------------------------------------------------------------------------

def _plot_panel_a_equations(estimator, participant_ids, cluster_labels=None):
    """Panel a: Discovered equations for each participant (side by side)."""
    n_participants = len(participant_ids)
    fig, axes = plt.subplots(1, n_participants, figsize=(5.5 * n_participants, 4))
    if n_participants == 1:
        axes = [axes]

    for col, pid in enumerate(participant_ids):
        ax = axes[col]
        ax.axis('off')

        eq_lines = _format_equation_lines(estimator, pid)

        if cluster_labels is not None:
            title = f"Cluster {cluster_labels[col]} \u2014 Participant {pid}"
        else:
            title = f"Participant {pid}"
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

        n_eq = len(eq_lines)
        y_spacing = min(0.20, 0.85 / max(n_eq, 1))
        y_pos = 0.92
        for lhs, rhs in eq_lines:
            ax.text(
                0.03, y_pos, f"{lhs}",
                transform=ax.transAxes,
                fontsize=8, fontfamily='monospace', fontweight='bold',
                verticalalignment='top', color='#333333',
            )
            ax.text(
                0.03, y_pos - 0.10, f"  = {rhs}",
                transform=ax.transAxes,
                fontsize=7.5, fontfamily='monospace',
                verticalalignment='top', color='#555555',
            )
            y_pos -= y_spacing

    fig.tight_layout()
    return fig


def _plot_panel_b_choices(dataset, model, participant_ids, session_idx, n_trials_show, cluster_labels=None):
    """Panel b: Choices and rewards for each participant."""
    n_participants = len(participant_ids)
    action_colors = ['#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, n_participants, figsize=(5.5 * n_participants, 2))
    if n_participants == 1:
        axes = [axes]

    for col, pid in enumerate(participant_ids):
        ax = axes[col]
        xs_single, actions, rewards, n_valid = _extract_session_data(dataset, pid, session_idx)
        n_show = min(n_trials_show, n_valid) if n_trials_show else n_valid
        trials = np.arange(n_show)
        actions = actions[:n_show]
        rewards = rewards[:n_show]

        rnn_probs = _run_forward(model, xs_single, n_valid, False)[:n_show]
        n_actions = rnn_probs.shape[1]

        for a in range(n_actions):
            mask = actions == a
            ax.scatter(
                trials[mask], np.full(mask.sum(), a),
                c=action_colors[a], marker='|', s=20, linewidths=1.2, zorder=3,
            )

        ax.set_yticks(range(n_actions))
        ax.set_yticklabels([f'A{a}' for a in range(n_actions)], fontsize=8)
        ax.set_ylabel('Choice', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Reward shading
        ax_r = ax.twinx()
        ax_r.fill_between(trials, rewards, alpha=0.12, color='goldenrod', step='mid')
        ax_r.set_ylim(-0.1, 1.5)
        if col == n_participants - 1:
            ax_r.set_ylabel('Reward', fontsize=8, color='goldenrod')
            ax_r.tick_params(axis='y', colors='goldenrod', labelsize=7)
        else:
            ax_r.set_yticks([])
        ax_r.spines['top'].set_visible(False)

        if col == 0:
            ax.set_title('Choices & Rewards', fontsize=9, loc='left', color='grey')

    fig.tight_layout()
    return fig


def _plot_panel_c_probs(dataset, model, participant_ids, session_idx, n_trials_show, cluster_labels=None):
    """Panel c: Action probabilities (RNN + EQ overlaid) for each participant."""
    n_participants = len(participant_ids)
    action_colors = ['#2ca02c', '#d62728']
    rnn_color = '#888888'

    fig, axes = plt.subplots(1, n_participants, figsize=(5.5 * n_participants, 3))
    if n_participants == 1:
        axes = [axes]

    for col, pid in enumerate(participant_ids):
        ax = axes[col]
        xs_single, actions, rewards, n_valid = _extract_session_data(dataset, pid, session_idx)
        n_show = min(n_trials_show, n_valid) if n_trials_show else n_valid
        trials = np.arange(n_show)

        rnn_probs = _run_forward(model, xs_single, n_valid, False)[:n_show]
        eq_probs = _run_forward(model, xs_single, n_valid, True)[:n_show]
        n_actions = rnn_probs.shape[1]

        for a in range(n_actions):
            ax.plot(trials, rnn_probs[:, a],
                    color=rnn_color, linewidth=1.0, alpha=0.6, linestyle='--',
                    label=f'RNN P(A{a})' if col == 0 and a == 0 else None)
            ax.plot(trials, eq_probs[:, a],
                    color=action_colors[a], linewidth=1.5, alpha=0.85,
                    label=f'SPICE-EQ P(A{a})' if col == 0 else None)

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Trial', fontsize=9)
        ax.set_ylabel('P(action)', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.12)

        if col == 0:
            ax.set_title('Action Probabilities', fontsize=9, loc='left', color='grey')
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=rnn_color, linestyle='--', linewidth=1, label='RNN'),
                Line2D([0], [0], color=action_colors[0], linewidth=1.5, label='SPICE-EQ (A0)'),
                Line2D([0], [0], color=action_colors[1], linewidth=1.5, label='SPICE-EQ (A1)'),
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc='upper right',
                      framealpha=0.8, borderpad=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_figure2(
    estimator,
    dataset,
    participant_ids,
    cluster_labels=None,
    session_idx=None,
    output_dir='figures/figure2',
    n_trials_show=None,
):
    """Create Figure 2 panels: Equation showcase.

    Each panel saved as detailed + clean versions in output_dir.
    """
    model = estimator.model

    # Panel a: Equations
    fig_a = _plot_panel_a_equations(estimator, participant_ids, cluster_labels)
    save_panel(fig_a, output_dir, 'panel_a_equations')

    # Panel b: Choices and rewards
    fig_b = _plot_panel_b_choices(dataset, model, participant_ids, session_idx, n_trials_show, cluster_labels)
    save_panel(fig_b, output_dir, 'panel_b_choices')

    # Panel c: Action probabilities
    fig_c = _plot_panel_c_probs(dataset, model, participant_ids, session_idx, n_trials_show, cluster_labels)
    save_panel(fig_c, output_dir, 'panel_c_action_probs')

    print(f"\nAll Figure 2 panels saved to {output_dir}/")
