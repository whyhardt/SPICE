import matplotlib.pyplot as plt
import numpy as np


# Toy example: Three armed bandit data visualization
np.random.seed(42)
n_trials = 20
n_actions = 2
linewidth = 10

# Generate cue signal with jumps for different arms (will be set after actions are determined)
u = np.zeros(n_trials)

# Generate drifting reward probabilities for each arm (underlying reward structure)
t = np.arange(n_trials)
reward_drift = np.zeros((n_trials, n_actions))
for arm in range(n_actions):
    # Each arm has a different drifting pattern
    phase = arm * 2 * np.pi / n_actions
    drift = 0.5 + 0.3 * np.sin(t / 15 + phase) + np.cumsum(np.random.randn(n_trials) * 0.02)
    drift = np.clip(drift, 0.2, 0.8)
    reward_drift[:, arm] = drift

# Simulate agent choosing actions with exploration/exploitation
actions = np.zeros(n_trials, dtype=int)
observed_rewards = np.zeros(n_trials)
action_values = np.zeros(n_actions)

for i in range(n_trials):
    # Choose action (epsilon-greedy with learning)
    if i < 5 or np.random.rand() < 0.15:  # exploration
        actions[i] = np.random.randint(0, n_actions)
    else:  # exploitation based on learned values
        actions[i] = np.argmax(action_values + np.random.randn(n_actions) * 0.1)

    # Set cue signal based on chosen action (discrete levels with noise)
    arm_levels = [-0.3, 0.0, 0.3]  # Different level for each arm
    u[i] = arm_levels[actions[i]] + np.random.randn() * 0.05

    # Generate reward for chosen action (probabilistic, can be 0)
    if np.random.rand() < reward_drift[i, actions[i]]:
        # Reward obtained
        observed_rewards[i] = 0.6 + np.random.randn() * 0.15
        observed_rewards[i] = np.clip(observed_rewards[i], 0.3, 1.0)
    else:
        # No reward
        observed_rewards[i] = 0.0

    # Update action value estimates (simple learning)
    action_values[actions[i]] = 0.8 * action_values[actions[i]] + 0.2 * observed_rewards[i]

# Create the plot with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), dpi=150, sharex=True,
                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})

# Top panel: Cue signal and rewards
ax1.plot(t, u, '#ffd428', alpha=0.6, linewidth=linewidth, label='Cue signal (u)')
ax1.plot(t, observed_rewards, '#e8a202', linewidth=linewidth, label='Feedback (r)', alpha=0.8)

# Formatting for top panel
ax1.set_ylabel('Signal value', fontsize=12)
ax1.set_ylim(-0.5, 1.1)
# ax1.legend(loc='upper left', framealpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# ax1.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Bottom panel: Action trajectory
action_display = actions + 1  # Convert 0,1,2 to 1,2,3 for display
ax2.step(t, action_display, where='post', linewidth=linewidth, color="#d62e4e", alpha=0.8)

# Formatting for bottom panel
ax2.set_xlabel('Trial', fontsize=12)
ax2.set_ylabel('Action', fontsize=12)
ax2.set_ylim(0.5, 3.5)
ax2.set_yticks([1, 2, 3])
ax2.set_xlim(0, n_trials-1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('three_armed_bandit_example.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------------------------------------

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
ax1.plot(t_range, probs_rnn[:, action_to_show], color='#d62e4e', linewidth=linewidth,
         linestyle='--', label='RNN', alpha=0.9)
ax1.plot(t_range, probs_sym[:, action_to_show], color='#d62e4e', linewidth=linewidth,
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
ax2.plot(t_range, v1_rnn[:, action_to_show], color='#ffd7d7', linewidth=linewidth,
         linestyle='--', label='$v_1$ (RNN)', alpha=0.9)
ax2.plot(t_range, v1_sym[:, action_to_show], color='#ffd7d7', linewidth=linewidth,
         label='$v_1$ (EQ)', alpha=0.6)

# v2 - medium blue tone
ax2.plot(t_range, v2_rnn[:, action_to_show], color='#ffa6a6', linewidth=linewidth,
         linestyle='--', label='$v_2$ (RNN)', alpha=0.9)
ax2.plot(t_range, v2_sym[:, action_to_show], color='#ffa6a6', linewidth=linewidth,
         label='$v_2$ (EQ)', alpha=0.6)

# v3 - lighter blue tone
ax2.plot(t_range, v3_rnn[:, action_to_show], color='#ff6d6d', linewidth=linewidth,
         linestyle='--', label='$v_3$ (RNN)', alpha=0.9)
ax2.plot(t_range, v3_sym[:, action_to_show], color='#ff6d6d', linewidth=linewidth,
         label='$v_3$ (EQ)', alpha=0.6)

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
