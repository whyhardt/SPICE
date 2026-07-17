"""
Figure 2 Prototype: Equation Showcase

Panels (saved individually, one per cluster-representative participant):
  a) Discovered equations (rendered as text) — per participant
  b) Actual choices and rewards — per participant
  c) RNN vs SPICE-EQ action probabilities (overlaid) — per participant
  d) Combined dynamics: P(action), V_reward, V_choice for item 0 — per participant

Each panel saved as detailed + clean versions in figures/figure2/.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform

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


def _run_forward_with_states(model, xs_single, n_valid, use_sindy):
    """Run forward pass and record per-trial value_reward / value_choice for item 0.

    Returns:
        probs:  (n_valid, n_actions)
        vr:     (n_valid, n_actions)  value_reward per item per trial
        vc:     (n_valid, n_actions)  value_choice per item per trial
    """
    model.use_sindy = use_sindy
    model.eval()

    # We need per-trial states, so we hook into the forward pass.
    # Run init manually, iterate trials, and record states.
    with torch.no_grad():
        spice_signals = model.init_forward_pass(xs_single, prev_state=None)
        participant_embedding = model.participant_embedding(spice_signals.participant_ids)

        vr_list, vc_list = [], []
        for trial in spice_signals.trials:
            # --- replicate the forward() body from workingmemory.SpiceModel ---
            model.call_module(
                key_module='value_reward_chosen', key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(spice_signals.feedback[trial],
                        model.state['buffer_reward_1'],
                        model.state['buffer_reward_2'],
                        model.state['buffer_reward_3']),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            model.call_module(
                key_module='value_reward_not_chosen', key_state='value_reward',
                action_mask=1 - spice_signals.actions[trial],
                inputs=(model.state['buffer_reward_1'],
                        model.state['buffer_reward_2'],
                        model.state['buffer_reward_3']),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            model.call_module(
                key_module='value_choice_chosen', key_state='value_choice',
                action_mask=spice_signals.actions[trial],
                inputs=(model.state['buffer_action_1'],
                        model.state['buffer_action_2'],
                        model.state['buffer_action_3']),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            model.call_module(
                key_module='value_choice_not_chosen', key_state='value_choice',
                action_mask=1 - spice_signals.actions[trial],
                inputs=(model.state['buffer_action_1'],
                        model.state['buffer_action_2'],
                        model.state['buffer_action_3']),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # buffer updates
            model.state['buffer_reward_3'] = (
                model.state['buffer_reward_2'] * spice_signals.actions[trial]
                + model.state['buffer_reward_3'] * (1 - spice_signals.actions[trial])
            )
            model.state['buffer_reward_2'] = (
                model.state['buffer_reward_1'] * spice_signals.actions[trial]
                + model.state['buffer_reward_2'] * (1 - spice_signals.actions[trial])
            )
            model.state['buffer_reward_1'] = (
                torch.where(spice_signals.actions[trial] == 1, spice_signals.feedback[trial], 0)
                + torch.where(spice_signals.actions[trial] == 0, model.state['buffer_reward_1'], 0)
            )
            model.state['buffer_action_3'] = model.state['buffer_action_2']
            model.state['buffer_action_2'] = model.state['buffer_action_1']
            model.state['buffer_action_1'] = spice_signals.actions[trial]

            spice_signals.logits[trial] = model.state['value_reward'] + model.state['value_choice']

            # Record states: shape (W=1, E, B=1, I) → mean over E → (I,)
            vr_list.append(model.state['value_reward'][0, :, 0, :].mean(dim=0).cpu().numpy())
            vc_list.append(model.state['value_choice'][0, :, 0, :].mean(dim=0).cpu().numpy())

        spice_signals = model.post_forward_pass(spice_signals)

    # Extract logits → probs
    logits_out = spice_signals.logits
    if logits_out.dim() == 5:
        if model.batch_first:
            logits = logits_out[:, 0, :n_valid, 0, :].mean(dim=0)
        else:
            logits = logits_out[:n_valid, 0, :, 0, :].mean(dim=1)
    else:
        logits = logits_out[0, :n_valid, 0, :]

    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    vr = np.stack(vr_list[:n_valid], axis=0)  # (n_valid, n_actions)
    vc = np.stack(vc_list[:n_valid], axis=0)

    return probs, vr, vc


# ---------------------------------------------------------------------------
# Automatic participant selection via structural distance
# ---------------------------------------------------------------------------

def select_distinctive_participants(model, n_select=3):
    """Find n_select participants with maximally different sparsity patterns.

    Uses ensemble-majority voting on presence masks, then greedy max-min
    Hamming distance selection across all modules.

    Returns:
        List of participant IDs.
    """
    modules = model.get_modules()
    P = next(iter(model.sindy_coefficients.values())).shape[1]

    # Build binary signature vector per participant
    sig_vectors = []
    for pid in range(P):
        vec = []
        for m in modules:
            pres = model.sindy_coefficients_presence[m]  # (E, P, X, C)
            if isinstance(pres, torch.Tensor):
                pres = pres.detach().cpu().numpy()
            pres_pid = pres[:, pid, :, :].mean(axis=(0, 1))  # (C,)
            vec.extend((pres_pid > 0.5).astype(float).tolist())
        sig_vectors.append(vec)

    sig_vectors = np.array(sig_vectors)  # (P, total_terms)

    # Find unique signatures and pick one representative per cluster
    sig_tuples = [tuple(v) for v in sig_vectors]
    unique_sigs = {}
    for pid, st in enumerate(sig_tuples):
        if st not in unique_sigs:
            unique_sigs[st] = pid  # first occurrence

    rep_pids = list(unique_sigs.values())
    rep_vecs = sig_vectors[rep_pids]

    # Pairwise Hamming distance
    dist_matrix = squareform(pdist(rep_vecs, metric='hamming'))

    # Greedy max-min selection
    n_reps = len(rep_pids)
    # Start with the two most distant
    max_dist, best_pair = 0, (0, 1)
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            if dist_matrix[i, j] > max_dist:
                max_dist = dist_matrix[i, j]
                best_pair = (i, j)

    selected = [best_pair[0], best_pair[1]]
    for _ in range(n_select - 2):
        best_k, best_min = -1, -1
        for k in range(n_reps):
            if k in selected:
                continue
            min_d = min(dist_matrix[k, s] for s in selected)
            if min_d > best_min:
                best_min = min_d
                best_k = k
        if best_k >= 0:
            selected.append(best_k)

    return [rep_pids[i] for i in selected]


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


def _plot_panel_d_dynamics(dataset, model, participant_ids, session_idx,
                           n_trials_show, cluster_labels=None):
    """Panel d: Actual data + model dynamics.

    5 rows sharing trial axis per participant:
      Row 0 — Reward (actual data, per-action traces in gold shades)
      Row 1 — Action (actual data, step plot)
      Row 2 — P(action 0)
      Row 3 — V_reward (item 0)
      Row 4 — V_choice (item 0)

    Dashed = SPICE-RNN (thicker, brighter), solid = SPICE-EQ.
    """
    n_participants = len(participant_ids)
    brick_red = '#c44e52'
    lw = 2.2
    gold_colors = ['#DAA520', '#FFD700']  # per-action reward colours

    n_rows = 5
    height_ratios = [0.35, 0.35, 1, 1, 1]

    fig, axes = plt.subplots(
        n_rows, n_participants, figsize=(5.5 * n_participants, 7),
        sharex='col', squeeze=False,
        gridspec_kw={'height_ratios': height_ratios},
    )

    row_labels = [
        'Reward', 'Action',
        'P(action 0)', r'$V_{reward}$  (item 0)', r'$V_{choice}$  (item 0)',
    ]

    for col, pid in enumerate(participant_ids):
        xs_single, actions, rewards, n_valid = _extract_session_data(dataset, pid, session_idx)
        n_show = min(n_trials_show, n_valid) if n_trials_show else n_valid
        trials = np.arange(n_show)
        acts_show = actions[:n_show]
        rwds_show = rewards[:n_show]

        # RNN and EQ forward passes
        probs_rnn, vr_rnn, vc_rnn = _run_forward_with_states(model, xs_single, n_valid, False)
        probs_eq, vr_eq, vc_eq = _run_forward_with_states(model, xs_single, n_valid, True)
        probs_rnn, vr_rnn, vc_rnn = probs_rnn[:n_show], vr_rnn[:n_show], vc_rnn[:n_show]
        probs_eq, vr_eq, vc_eq = probs_eq[:n_show], vr_eq[:n_show], vc_eq[:n_show]
        n_actions = probs_rnn.shape[1]

        # ── Row 0: Reward (actual data, single line) ──
        ax = axes[0, col]
        ax.plot(trials, rwds_show, color=gold_colors[0],
                linewidth=3.0, alpha=0.85)
        ax.set_ylim(-0.15, 1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.12)

        # ── Row 1: Action (actual data, step plot) ──
        ax = axes[1, col]
        ax.step(trials, acts_show + 1, where='mid',
                color=brick_red, linewidth=3.0, alpha=0.85)
        ax.set_ylim(0.4, n_actions + 0.6)
        ax.set_yticks(range(1, n_actions + 1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ── Rows 2–4: model dynamics ──
        data_pairs = [
            (probs_rnn[:, 0], probs_eq[:, 0]),
            (vr_rnn[:, 0],    vr_eq[:, 0]),
            (vc_rnn[:, 0],    vc_eq[:, 0]),
        ]
        for row_offset, (rnn_vals, eq_vals) in enumerate(data_pairs):
            row = row_offset + 2
            ax = axes[row, col]
            # RNN: thicker and brighter than EQ
            ax.plot(trials, rnn_vals, color=brick_red, linewidth=lw + 3,
                    linestyle='--', alpha=0.55, zorder=1)
            ax.plot(trials, eq_vals, color=brick_red, linewidth=lw,
                    linestyle='-', alpha=0.9, zorder=2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.12)

            if row == 2:
                ax.set_ylim(-0.05, 1.05)

        # ── Per-column labels ──
        for row in range(n_rows):
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row], fontsize=10)
            if row == 0:
                if cluster_labels is not None:
                    title = f"Cluster {cluster_labels[col]} — P{pid}"
                else:
                    title = f"Participant {pid}"
                axes[row, col].set_title(title, fontsize=11, fontweight='bold')
            if row == n_rows - 1:
                axes[row, col].set_xlabel('Trial', fontsize=10)

    # SPICE-RNN / SPICE-EQ legend (first column, row 2)
    legend_elements = [
        Line2D([0], [0], color=brick_red, linewidth=lw + 3, linestyle='--',
               alpha=0.55, label='SPICE-RNN'),
        Line2D([0], [0], color=brick_red, linewidth=lw, linestyle='-',
               alpha=0.9, label='SPICE-EQ'),
    ]
    axes[2, 0].legend(handles=legend_elements, fontsize=8, loc='upper right',
                      framealpha=0.8, borderpad=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_figure2(
    estimator,
    dataset,
    participant_ids=None,
    cluster_labels=None,
    session_idx=None,
    output_dir='figures/figure2',
    n_trials_show=None,
    n_select=3,
):
    """Create Figure 2 panels: Equation showcase.

    If participant_ids is None, automatically selects n_select structurally
    distinctive participants via max-min Hamming distance on sparsity patterns.

    Each panel saved as detailed + clean versions in output_dir.
    """
    model = estimator.model

    if participant_ids is None:
        participant_ids = select_distinctive_participants(model, n_select=n_select)
        print(f"  Auto-selected participants (structural distance): {participant_ids}")

    # Panel a: Equations
    fig_a = _plot_panel_a_equations(estimator, participant_ids, cluster_labels)
    save_panel(fig_a, output_dir, 'panel_a_equations')

    # Panel b: Choices and rewards
    fig_b = _plot_panel_b_choices(dataset, model, participant_ids, session_idx, n_trials_show, cluster_labels)
    save_panel(fig_b, output_dir, 'panel_b_choices')

    # Panel c: Action probabilities
    fig_c = _plot_panel_c_probs(dataset, model, participant_ids, session_idx, n_trials_show, cluster_labels)
    save_panel(fig_c, output_dir, 'panel_c_action_probs')

    # Panel d: Combined dynamics (P(action), V_reward, V_choice)
    fig_d = _plot_panel_d_dynamics(dataset, model, participant_ids, session_idx, n_trials_show, cluster_labels)
    save_panel(fig_d, output_dir, 'panel_d_dynamics')

    print(f"\nAll Figure 2 panels saved to {output_dir}/")
