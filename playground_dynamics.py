import matplotlib.pyplot as plt
import numpy as np

# Toy example: SPICE dynamics visualization
np.random.seed(42)
n_trials = 30
n_actions = 3
linewidth = 10

# Generate random actions, rewards, and cues
actions_taken = np.random.randint(0, n_actions, n_trials)
rewards = np.random.choice([0, 0.7, 0.9], n_trials)
u_cues = np.random.randn(n_trials, n_actions) * 0.3  # cues for next trial

# One-hot encode actions
actions_onehot = np.zeros((n_trials, n_actions))
for t, a in enumerate(actions_taken):
    actions_onehot[t, a] = 1

# Initialize value states for ground truth model
v1_gt = np.zeros((n_trials + 1, n_actions))
v2_gt = np.zeros((n_trials + 1, n_actions))
v3_gt = np.zeros((n_trials + 1, n_actions))

# Simulate RNN model (baseline) using discovered equations with noise
for t in range(n_trials):
    for a in range(n_actions):
        a_t = actions_onehot[t, a]
        r_t = rewards[t] if a == actions_taken[t] else 0.0
        u_t1 = u_cues[t, a]

        # Apply discovered equations with noise for RNN
        v1_gt[t+1, a] = 0.8 * v1_gt[t, a] + 1.32 * a_t - 0.23 * v1_gt[t, a] * a_t + np.random.randn() * 0.08
        v2_gt[t+1, a] = 0.51 * v2_gt[t, a] + 1.56 * r_t - 0.23 * r_t**2 + np.random.randn() * 0.08
        v3_gt[t+1, a] = 1.43 * u_t1 + np.random.randn() * 0.08

# RNN values are the baseline
v1_rnn = v1_gt.copy()
v2_rnn = v2_gt.copy()
v3_rnn = v3_gt.copy()
total_values_rnn = v1_rnn + v2_rnn + v3_rnn
probs_rnn = np.exp(total_values_rnn) / np.exp(total_values_rnn).sum(axis=1, keepdims=True)

# Ground truth model (participant) diverges from RNN
v1_gt = v1_rnn + np.random.randn(*v1_rnn.shape) * 0.35
v2_gt = v2_rnn + np.random.randn(*v2_rnn.shape) * 0.35
v3_gt = v3_rnn + np.random.randn(*v3_rnn.shape) * 0.35
total_values_gt = v1_gt + v2_gt + v3_gt
probs_gt = np.exp(total_values_gt) / np.exp(total_values_gt).sum(axis=1, keepdims=True)

# Symbolic model (SPICE) also diverges slightly from RNN
v1_sym = v1_rnn + np.random.randn(*v1_rnn.shape) * 0.10
v2_sym = v2_rnn + np.random.randn(*v2_rnn.shape) * 0.10
v3_sym = v3_rnn + np.random.randn(*v3_rnn.shape) * 0.10
total_values_sym = v1_sym + v2_sym + v3_sym
probs_sym = np.exp(total_values_sym) / np.exp(total_values_sym).sum(axis=1, keepdims=True)

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=150, sharex=True,
                                gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.15})

t_range = np.arange(n_trials + 1)

# Top panel: Action probabilities for action 0 (can show all 3 if needed)
action_to_show = 0
# ax1.plot(t_range, probs_gt[:, action_to_show], color='gold', linewidth=linewidth,
#          label='Ground truth (participant p)', alpha=0.9)
ax1.plot(t_range, probs_rnn[:, action_to_show], color='darkorange', linewidth=linewidth,
         linestyle='--', label='RNN', alpha=0.9)
ax1.plot(t_range, probs_sym[:, action_to_show], color='magenta', linewidth=linewidth,
         label='Symbolic (SPICE)', alpha=0.6)

ax1.set_ylabel(f'P(action {action_to_show+1})', fontsize=12)
ax1.set_ylim(-0.05, 1.05)
# ax1.legend(loc='upper right', framealpha=0.95, fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(alpha=0.2, linestyle='--', linewidth=0.5)

# Bottom panel: Value dynamics for action 0 - showing RNN and SPICE only
# Each value (v1, v2, v3) gets its own blue tone, with RNN dashed and SPICE solid

# v1 - darker blue tone
ax2.plot(t_range, v1_rnn[:, action_to_show], color='#1f77b4', linewidth=linewidth,
         linestyle='--', label='$v_1$ (RNN)', alpha=0.9)
ax2.plot(t_range, v1_sym[:, action_to_show], color='#1f77b4', linewidth=linewidth,
         label='$v_1$ (SPICE)', alpha=0.6)

# v2 - medium blue tone
ax2.plot(t_range, v2_rnn[:, action_to_show], color='#5DA5DA', linewidth=linewidth,
         linestyle='--', label='$v_2$ (RNN)', alpha=0.9)
ax2.plot(t_range, v2_sym[:, action_to_show], color='#5DA5DA', linewidth=linewidth,
         label='$v_2$ (SPICE)', alpha=0.6)

# v3 - lighter blue tone
ax2.plot(t_range, v3_rnn[:, action_to_show], color='#9AC9E3', linewidth=linewidth,
         linestyle='--', label='$v_3$ (RNN)', alpha=0.9)
ax2.plot(t_range, v3_sym[:, action_to_show], color='#9AC9E3', linewidth=linewidth,
         label='$v_3$ (SPICE)', alpha=0.6)

ax2.set_xlabel('Trial', fontsize=12)
ax2.set_ylabel('Value', fontsize=12)
ax2.set_xlim(0, n_trials)
# ax2.legend(loc='upper right', framealpha=0.95, fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(alpha=0.2, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('spice_dynamics_example.png', dpi=300, bbox_inches='tight')
plt.show()
