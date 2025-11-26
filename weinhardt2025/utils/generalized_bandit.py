from typing import Iterable, Optional, Tuple
import numpy as np
from scipy.stats import beta


class BanditsGeneral:
    """
    A generalized form of the multi-armed bandit which can enable a drifting paradigm, 
    reversal learning, reward/no-reward and reward/penalty schedules, and arm-correlations.
    
    Features:
    - Drift: Reward probabilities change gradually over time
    - Reversals: Sudden swaps in reward probabilities (controlled by hazard rate)
    - Correlated arms: Reward probabilities can move together
    - Flexible reward schedules: Binary rewards (0/1) or penalty schedules (-1/+1)
    """
    
    def __init__(
        self,
        n_arms: int = 2,
        init_reward_prob: Optional[Iterable[float]] = None,
        drift_rate: float = 0.0,
        hazard_rate: float = 0.0,
        reward_prob_correlation: float = 0.0,
        reward_schedule: str = "binary",  # "binary" (0/1) or "penalty" (-1/+1)
        bounds: Tuple[float, float] = (0.0, 1.0),
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_arms: Number of arms
            init_reward_prob: Initial reward probabilities for each arm
            drift_rate: Rate of Gaussian random walk drift (std dev per step)
            hazard_rate: Probability of reversal on each step
            reward_prob_correlation: Correlation between arm drifts (-1 to 1)
            reward_schedule: "binary" for 0/1 rewards, "penalty" for -1/+1 rewards
            bounds: Min and max values for reward probabilities
            seed: Random seed for reproducibility
        """
        self.n_arms = n_arms
        self.drift_rate = drift_rate
        self.hazard_rate = hazard_rate
        self.reward_prob_correlation = reward_prob_correlation
        self.reward_schedule = reward_schedule
        self.bounds = bounds
        
        # Random number generator
        self.rng = np.random.default_rng(seed)
        
        # Initialize reward probabilities
        if init_reward_prob is None:
            init_reward_prob = self.rng.uniform(bounds[0], bounds[1], n_arms)
        else:
            init_reward_prob = np.array(init_reward_prob)
            if len(init_reward_prob) != n_arms:
                raise ValueError(f"init_reward_prob must have length {n_arms}")
        
        self.init_reward_prob = np.array(init_reward_prob)
        self.reward_prob = self.init_reward_prob.copy()
        
        # Tracking
        self.t = 0
        self.history = {
            'choices': [],
            'rewards': [],
            'reward_probs': [self.reward_prob.copy()],
            'reversals': []
        }
        
        # For correlated drift
        if self.reward_prob_correlation != 0 and self.n_arms == 2:
            # Construct covariance matrix for bivariate normal
            self.drift_cov = np.array([
                [1, self.reward_prob_correlation],
                [self.reward_prob_correlation, 1]
            ]) * (self.drift_rate ** 2)
        else:
            self.drift_cov = None
    
    def step(self, choice: int) -> Tuple[float, dict]:
        """
        Execute one step: apply drift/reversals, then generate reward for chosen arm.
        
        Args:
            choice: Index of chosen arm (0 to n_arms-1)
            
        Returns:
            reward: The reward received
            info: Dictionary with additional information
        """
        if choice < 0 or choice >= self.n_arms:
            raise ValueError(f"Choice must be between 0 and {self.n_arms-1}")
        
        # Apply drift and reversals BEFORE reward is generated
        reversal_occurred = self._apply_dynamics()
        
        # Generate reward based on current probabilities
        reward = self._generate_reward(choice)
        
        # Update history
        self.history['choices'].append(choice)
        self.history['rewards'].append(reward)
        self.history['reward_probs'].append(self.reward_prob.copy())
        self.history['reversals'].append(reversal_occurred)
        
        self.t += 1
        
        info = {
            'reward_prob': self.reward_prob[choice],
            'all_reward_probs': self.reward_prob.copy(),
            'reversal': reversal_occurred,
            'timestep': self.t
        }
        
        return reward, info
    
    def _apply_dynamics(self) -> bool:
        """Apply drift and check for reversals."""
        reversal_occurred = False
        
        # Check for reversal (sudden swap)
        if self.hazard_rate > 0 and self.rng.random() < self.hazard_rate:
            self._apply_reversal()
            reversal_occurred = True
        
        # Apply gradual drift
        if self.drift_rate > 0:
            self._apply_drift()
        
        return reversal_occurred
    
    def _apply_reversal(self):
        """Apply a reversal: swap the reward probabilities."""
        if self.n_arms == 2:
            # Simple swap for 2 arms
            self.reward_prob = self.reward_prob[::-1]
        else:
            # For >2 arms, randomly permute
            self.reward_prob = self.rng.permutation(self.reward_prob)
    
    def _apply_drift(self):
        """Apply Gaussian random walk drift to reward probabilities."""
        if self.drift_cov is not None and self.n_arms == 2:
            # Correlated drift for 2 arms
            drift = self.rng.multivariate_normal(np.zeros(2), self.drift_cov)
        else:
            # Independent drift
            drift = self.rng.normal(0, self.drift_rate, self.n_arms)
        
        # Apply drift and clip to bounds
        self.reward_prob = np.clip(
            self.reward_prob + drift,
            self.bounds[0],
            self.bounds[1]
        )
    
    def _generate_reward(self, choice: int) -> float:
        """Generate reward for chosen arm based on current probability."""
        # Bernoulli trial
        success = self.rng.random() < self.reward_prob[choice]
        
        if self.reward_schedule == "binary":
            return 1.0 if success else 0.0
        elif self.reward_schedule == "penalty":
            return 1.0 if success else -1.0
        else:
            raise ValueError(f"Unknown reward_schedule: {self.reward_schedule}")
    
    def new_sess(self):
        """Reset to initial state for a new session."""
        self.reward_prob = self.init_reward_prob.copy()
        self.t = 0
        self.history = {
            'choices': [],
            'rewards': [],
            'reward_probs': [self.reward_prob.copy()],
            'reversals': []
        }
    
    def get_optimal_arm(self) -> int:
        """Return the index of the arm with highest current reward probability."""
        return int(np.argmax(self.reward_prob))
    
    def get_history_array(self) -> dict:
        """Return history as numpy arrays for analysis."""
        return {
            'choices': np.array(self.history['choices']),
            'rewards': np.array(self.history['rewards']),
            'reward_probs': np.array(self.history['reward_probs']),
            'reversals': np.array(self.history['reversals'])
        }


# Example usage and demonstration
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Set up the figure with subplots for each example
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # Define colors for each example
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Example 1: Static bandit
    print("Running Example 1: Static Bandit")
    bandit = BanditsGeneral(n_arms=2, init_reward_prob=[0.8, 0.2], seed=42)
    probs_history = [bandit.reward_prob.copy()]
    for _ in range(100):
        choice = bandit.rng.integers(0, 2)
        reward, info = bandit.step(choice)
        probs_history.append(info['all_reward_probs'])

    probs_history = np.array(probs_history)
    axes[0].plot(probs_history[:, 0], label='Arm 0', color=colors[0], linewidth=2)
    axes[0].plot(probs_history[:, 1], label='Arm 1', color=colors[0], linewidth=2, linestyle='--')
    axes[0].set_title('Static Bandit', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Reward Probability')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Example 2: Drifting bandit (uncorrelated)
    print("Running Example 2: Drifting Bandit")
    bandit = BanditsGeneral(
        n_arms=2,
        init_reward_prob=[0.8, 0.2],
        drift_rate=0.2,
        seed=42
    )
    probs_history = [bandit.reward_prob.copy()]
    for _ in range(100):
        choice = bandit.rng.integers(0, 2)
        reward, info = bandit.step(choice)
        probs_history.append(info['all_reward_probs'])

    probs_history = np.array(probs_history)
    axes[1].plot(probs_history[:, 0], label='Arm 0', color=colors[1], linewidth=2)
    axes[1].plot(probs_history[:, 1], label='Arm 1', color=colors[1], linewidth=2, linestyle='--')
    axes[1].set_title('Drifting Bandit (Uncorrelated)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('Reward Probability')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1])

    # Example 3: Reversal learning
    print("Running Example 3: Reversal Learning")
    bandit = BanditsGeneral(
        n_arms=2,
        init_reward_prob=[0.8, 0.2],
        hazard_rate=0.05,
        seed=42
    )
    probs_history = [bandit.reward_prob.copy()]
    reversal_points = []
    for i in range(100):
        choice = bandit.rng.integers(0, 2)
        reward, info = bandit.step(choice)
        probs_history.append(info['all_reward_probs'])
        if info['reversal']:
            reversal_points.append(i)

    probs_history = np.array(probs_history)
    axes[2].plot(probs_history[:, 0], label='Arm 0', color=colors[2], linewidth=2)
    axes[2].plot(probs_history[:, 1], label='Arm 1', color=colors[2], linewidth=2, linestyle='--')
    # Mark reversals
    for rp in reversal_points:
        axes[2].axvline(x=rp, color='red', alpha=0.3, linestyle=':')
    axes[2].set_title('Reversal Learning (hazard=0.05)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Trial')
    axes[2].set_ylabel('Reward Probability')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim([0, 1])

    # Example 4: Correlated drift (positive)
    print("Running Example 4: Positively Correlated Drift")
    bandit = BanditsGeneral(
        n_arms=2,
        init_reward_prob=[0.7, 0.3],
        drift_rate=0.2,
        reward_prob_correlation=0.8,
        seed=42
    )
    probs_history = [bandit.reward_prob.copy()]
    for _ in range(100):
        choice = bandit.rng.integers(0, 2)
        reward, info = bandit.step(choice)
        probs_history.append(info['all_reward_probs'])

    probs_history = np.array(probs_history)
    axes[3].plot(probs_history[:, 0], label='Arm 0', color=colors[3], linewidth=2)
    axes[3].plot(probs_history[:, 1], label='Arm 1', color=colors[3], linewidth=2, linestyle='--')
    axes[3].set_title('Positively Correlated Drift (ρ=0.8)', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Trial')
    axes[3].set_ylabel('Reward Probability')
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    axes[3].set_ylim([0, 1])

    # Example 5: Negatively correlated drift
    print("Running Example 5: Negatively Correlated Drift")
    bandit = BanditsGeneral(
        n_arms=2,
        init_reward_prob=[0.7, 0.3],
        drift_rate=0.2,
        reward_prob_correlation=-0.8,
        seed=42
    )
    probs_history = [bandit.reward_prob.copy()]
    for _ in range(100):
        choice = bandit.rng.integers(0, 2)
        reward, info = bandit.step(choice)
        probs_history.append(info['all_reward_probs'])

    probs_history = np.array(probs_history)
    axes[4].plot(probs_history[:, 0], label='Arm 0', color=colors[4], linewidth=2)
    axes[4].plot(probs_history[:, 1], label='Arm 1', color=colors[4], linewidth=2, linestyle='--')
    axes[4].set_title('Negatively Correlated Drift (ρ=-0.8)', fontsize=12, fontweight='bold')
    axes[4].set_xlabel('Trial')
    axes[4].set_ylabel('Reward Probability')
    axes[4].legend()
    axes[4].grid(alpha=0.3)
    axes[4].set_ylim([0, 1])

    # Example 6: Combined drift + reversals
    print("Running Example 6: Drift + Reversals")
    bandit = BanditsGeneral(
        n_arms=2,
        init_reward_prob=[0.7, 0.3],
        drift_rate=0.2,
        hazard_rate=0.03,
        seed=42
    )
    probs_history = [bandit.reward_prob.copy()]
    reversal_points = []
    for i in range(100):
        choice = bandit.rng.integers(0, 2)
        reward, info = bandit.step(choice)
        probs_history.append(info['all_reward_probs'])
        if info['reversal']:
            reversal_points.append(i)

    probs_history = np.array(probs_history)
    axes[5].plot(probs_history[:, 0], label='Arm 0', color=colors[5], linewidth=2)
    axes[5].plot(probs_history[:, 1], label='Arm 1', color=colors[5], linewidth=2, linestyle='--')
    # Mark reversals
    for rp in reversal_points:
        axes[5].axvline(x=rp, color='red', alpha=0.3, linestyle=':')
    axes[5].set_title('Drift + Reversals', fontsize=12, fontweight='bold')
    axes[5].set_xlabel('Trial')
    axes[5].set_ylabel('Reward Probability')
    axes[5].legend()
    axes[5].grid(alpha=0.3)
    axes[5].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('bandit_examples.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved comprehensive visualization to bandit_examples_all.png")
    plt.close()

    # Create a summary figure showing the key differences
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Run all examples with shorter trials for comparison
    n_trials = 150
    examples = [
        ("Static", {"drift_rate": 0, "hazard_rate": 0}),
        ("Drift", {"drift_rate": 0.02, "hazard_rate": 0}),
        ("Reversals", {"drift_rate": 0, "hazard_rate": 0.04}),
        ("Corr. Drift (ρ=+0.8)", {"drift_rate": 0.02, "reward_prob_correlation": 0.8}),
        ("Corr. Drift (ρ=-0.8)", {"drift_rate": 0.02, "reward_prob_correlation": -0.8}),
        ("Drift+Reversals", {"drift_rate": 0.015, "hazard_rate": 0.03}),
    ]

    for idx, (name, params) in enumerate(examples):
        bandit = BanditsGeneral(
            n_arms=2,
            init_reward_prob=[0.75, 0.25],
            seed=42 + idx,
            **params
        )
        
        probs_history = [bandit.reward_prob.copy()]
        for _ in range(n_trials):
            choice = bandit.rng.integers(0, 2)
            reward, info = bandit.step(choice)
            probs_history.append(info['all_reward_probs'])
        
        probs_history = np.array(probs_history)
        # Plot difference between arms to show dynamics
        diff = probs_history[:, 0] - probs_history[:, 1]
        ax.plot(diff, label=name, color=colors[idx], linewidth=2, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel('P(Arm 0) - P(Arm 1)', fontsize=12)
    ax.set_title('Comparison of Bandit Dynamics\n(Difference in Reward Probabilities)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('bandit_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comparison plot to bandit_comparison.png")
    plt.close()

    print("\n=== Summary Statistics ===")
    for idx, (name, params) in enumerate(examples):
        bandit = BanditsGeneral(
            n_arms=2,
            init_reward_prob=[0.75, 0.25],
            seed=42 + idx,
            **params
        )
        
        for _ in range(100):
            choice = bandit.rng.integers(0, 2)
            reward, info = bandit.step(choice)
        
        hist = bandit.get_history_array()
        n_reversals = np.sum(hist['reversals'])
        prob_changes = np.diff(hist['reward_probs'][:, 0])
        mean_change = np.mean(np.abs(prob_changes))
        
        print(f"{name:25s} | Reversals: {n_reversals:2d} | Mean |ΔP|: {mean_change:.4f}")