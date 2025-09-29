"""
Demo script showing the optimization concept for RNN implementations.
Demonstrates the key ideas of scan and vectorization for cognitive RNNs.
"""
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from spice.precoded import RescorlaWagnerRNN
from spice.optimized_rnns import OptimizedRescorlaWagnerRNN


def create_demo_data(n_sessions=50, n_trials=100, n_actions=2):
    """Create demo data for testing."""
    np.random.seed(42)
    torch.manual_seed(42)

    xs = []
    ys = []

    for session in range(n_sessions):
        session_xs = []
        session_ys = []

        for trial in range(n_trials):
            # Random action (one-hot encoded)
            action = np.random.randint(0, n_actions)
            action_onehot = np.zeros(n_actions)
            action_onehot[action] = 1.0

            # Random reward
            reward = np.random.choice([0, 1], p=[0.3, 0.7])
            reward_onehot = np.zeros(n_actions)
            reward_onehot[action] = reward

            # Input vector: [action1, action2, reward1, reward2, block, exp_id, participant_id]
            input_vec = np.concatenate([
                action_onehot,
                reward_onehot,
                [0],  # block
                [0],  # experiment_id
                [0],  # participant_id
            ])

            # Next action
            next_action = np.random.randint(0, n_actions)
            next_action_onehot = np.zeros(n_actions)
            next_action_onehot[next_action] = 1.0

            session_xs.append(input_vec)
            session_ys.append(next_action_onehot)

        xs.append(session_xs)
        ys.append(session_ys)

    xs_tensor = torch.tensor(xs, dtype=torch.float32).transpose(0, 1)
    ys_tensor = torch.tensor(ys, dtype=torch.float32).transpose(0, 1)

    class SimpleDataset:
        def __init__(self, xs, ys):
            self.xs = xs
            self.ys = ys

    return SimpleDataset(xs_tensor, ys_tensor)


def benchmark_implementations(sequence_lengths=[10, 25, 50, 100], n_runs=5):
    """Benchmark original vs optimized implementations."""
    print("Benchmarking RNN Implementations")
    print("=" * 40)

    results = {
        'sequence_lengths': sequence_lengths,
        'original_times': [],
        'optimized_times': [],
        'speedups': []
    }

    for seq_len in sequence_lengths:
        print(f"\nSequence length: {seq_len}")

        # Create test data
        dataset = create_demo_data(n_sessions=20, n_trials=seq_len)
        inputs = dataset.xs[:seq_len, :10]  # Use subset for speed

        # Test original implementation
        original_rnn = RescorlaWagnerRNN(n_actions=2)
        original_rnn.eval()

        # Warmup
        with torch.no_grad():
            _ = original_rnn.forward(inputs)

        start_time = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = original_rnn.forward(inputs)
        original_time = (time.time() - start_time) / n_runs

        # Test optimized implementation
        optimized_rnn = OptimizedRescorlaWagnerRNN(n_actions=2)
        optimized_rnn.eval()
        optimized_rnn.enable_optimization(True)

        # Warmup
        with torch.no_grad():
            _ = optimized_rnn.forward(inputs)

        start_time = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = optimized_rnn.forward(inputs)
        optimized_time = (time.time() - start_time) / n_runs

        speedup = original_time / optimized_time

        results['original_times'].append(original_time)
        results['optimized_times'].append(optimized_time)
        results['speedups'].append(speedup)

        print(f"  Original: {original_time:.4f}s")
        print(f"  Optimized: {optimized_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")

    return results


def demonstrate_scan_vectorization():
    """Demonstrate the core scan vectorization concept."""
    print("\nDemonstrating Scan Vectorization")
    print("=" * 40)

    from spice.optimized_rnns import rescorla_wagner_step, scan_rescorla_wagner

    # Setup
    batch_size, n_actions, seq_len = 5, 2, 10
    initial_state = torch.full((batch_size, n_actions), 0.5)
    actions = torch.randint(0, 2, (seq_len, batch_size, n_actions)).float()
    rewards = torch.randint(0, 2, (seq_len, batch_size, n_actions)).float()
    update_weights = torch.tensor([0.3, 0.0])

    print(f"Processing {seq_len} timesteps for {batch_size} sessions with {n_actions} actions")

    # Sequential processing (original approach)
    start_time = time.time()
    current_state = initial_state.clone()
    sequential_outputs = []
    for t in range(seq_len):
        current_state, output = rescorla_wagner_step(
            current_state, actions[t], rewards[t], update_weights
        )
        sequential_outputs.append(output)
    sequential_time = time.time() - start_time
    sequential_outputs = torch.stack(sequential_outputs, dim=0)

    # Scan processing (optimized approach)
    start_time = time.time()
    final_state, scan_outputs = scan_rescorla_wagner(
        initial_state, actions, rewards, update_weights
    )
    scan_time = time.time() - start_time

    print(f"Sequential processing: {sequential_time:.6f}s")
    print(f"Scan processing: {scan_time:.6f}s")
    print(f"Speedup: {sequential_time / scan_time:.2f}x")

    # Verify outputs are identical
    max_diff = torch.abs(sequential_outputs - scan_outputs).max().item()
    print(f"Maximum output difference: {max_diff:.8f}")


def plot_performance_comparison(results):
    """Plot performance comparison results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot execution times
    ax1.plot(results['sequence_lengths'], results['original_times'], 'b-o', label='Original', linewidth=2)
    ax1.plot(results['sequence_lengths'], results['optimized_times'], 'r-o', label='Optimized', linewidth=2)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot speedup
    ax2.plot(results['sequence_lengths'], results['speedups'], 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Performance Speedup')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('rnn_optimization_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nPlot saved as 'rnn_optimization_results.png'")


def main():
    """Run the optimization demonstration."""
    print("RNN Optimization Demonstration")
    print("=" * 50)

    # Demonstrate scan vectorization concept
    demonstrate_scan_vectorization()

    # Benchmark implementations
    results = benchmark_implementations()

    # Plot results
    plot_performance_comparison(results)

    # Summary
    avg_speedup = np.mean(results['speedups'])
    max_speedup = np.max(results['speedups'])

    print(f"\nSUMMARY:")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Maximum speedup: {max_speedup:.2f}x")

    print("\nOptimization Features Implemented:")
    print("✓ Functional state management")
    print("✓ Vectorized module architecture")
    print("✓ Scan-based forward pass")
    print("✓ Reduced sequential dependencies")
    print("✓ Maintained compatibility with original interface")

    return results


if __name__ == "__main__":
    results = main()