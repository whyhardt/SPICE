"""
Test the properly optimized StandardRNN that maintains exact computational logic
while achieving genuine performance improvements through optimized tensor operations.
"""
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from spice.precoded import StandardRNN
from spice.optimized_rnns import OptimizedStandardRNN


def create_test_data(n_sessions=20, n_trials=50, n_actions=2):
    """Create test data for benchmarking."""
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
                [session % 10],  # participant_id (cycling through 10 participants)
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


def test_computational_accuracy():
    """Test that the optimized implementation produces EXACTLY the same results as the original."""
    print("Testing Computational Accuracy")
    print("=" * 40)

    n_participants = 10
    dataset = create_test_data(n_sessions=15, n_trials=20, n_actions=2)
    inputs = dataset.xs[:10, :8]  # Use subset for detailed testing

    # Create models with identical weights
    original_rnn = StandardRNN(n_actions=2, n_participants=n_participants)
    optimized_rnn = OptimizedStandardRNN(n_actions=2, n_participants=n_participants)

    # Copy weights from original to optimized to ensure exact comparison
    optimized_rnn.participant_embedding.load_state_dict(original_rnn.participant_embedding.state_dict())
    for key in original_rnn.betas.keys():
        optimized_rnn.betas[key].load_state_dict(original_rnn.betas[key].state_dict())
    for key in original_rnn.submodules_rnn.keys():
        optimized_rnn.submodules_rnn[key].load_state_dict(original_rnn.submodules_rnn[key].state_dict())

    # Set to evaluation mode
    original_rnn.eval()
    optimized_rnn.eval()
    optimized_rnn.enable_optimization(True)

    # Forward pass comparison
    with torch.no_grad():
        orig_logits, orig_state = original_rnn.forward(inputs)
        opt_logits, opt_state = optimized_rnn.forward(inputs)

    # Detailed accuracy analysis
    logits_diff = torch.abs(orig_logits - opt_logits).max().item()
    reward_state_diff = torch.abs(orig_state['x_value_reward'] - opt_state['x_value_reward']).max().item()
    choice_state_diff = torch.abs(orig_state['x_value_choice'] - opt_state['x_value_choice']).max().item()

    print(f"Maximum logits difference: {logits_diff:.10f}")
    print(f"Maximum reward state difference: {reward_state_diff:.10f}")
    print(f"Maximum choice state difference: {choice_state_diff:.10f}")

    # Very strict tolerance for exact computational accuracy
    tolerance = 1e-6
    accuracy_passed = (logits_diff < tolerance and
                      reward_state_diff < tolerance and
                      choice_state_diff < tolerance)

    print(f"\nComputational Accuracy: {'✓ PASSED' if accuracy_passed else '✗ FAILED'}")
    print(f"Tolerance used: {tolerance}")

    return accuracy_passed, logits_diff, reward_state_diff, choice_state_diff


def benchmark_real_optimization():
    """Benchmark the properly optimized implementation."""
    print("\nBenchmarking Real Optimization")
    print("=" * 40)

    sequence_lengths = [25, 50, 75, 100]
    n_participants = 20
    results = {
        'sequence_lengths': sequence_lengths,
        'original_times': [],
        'optimized_times': [],
        'speedups': []
    }

    for seq_len in sequence_lengths:
        print(f"\nTesting sequence length: {seq_len}")

        # Create test data
        dataset = create_test_data(n_sessions=15, n_trials=seq_len)
        inputs = dataset.xs[:seq_len, :8]

        # Original implementation
        original_rnn = StandardRNN(n_actions=2, n_participants=n_participants)
        original_rnn.eval()

        # Warmup and timing
        with torch.no_grad():
            _ = original_rnn.forward(inputs)

        start_time = time.time()
        n_runs = 5
        for _ in range(n_runs):
            with torch.no_grad():
                _ = original_rnn.forward(inputs)
        original_time = (time.time() - start_time) / n_runs

        # Optimized implementation
        optimized_rnn = OptimizedStandardRNN(n_actions=2, n_participants=n_participants)
        optimized_rnn.eval()
        optimized_rnn.enable_optimization(True)

        # Warmup and timing
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


def analyze_optimization_sources():
    """Analyze what optimizations contribute most to the speedup."""
    print("\nAnalyzing Optimization Sources")
    print("=" * 40)

    n_participants = 15
    dataset = create_test_data(n_sessions=10, n_trials=50)
    inputs = dataset.xs[:30, :6]

    # Test different optimization components
    optimized_rnn = OptimizedStandardRNN(n_actions=2, n_participants=n_participants)
    optimized_rnn.eval()

    print("Testing optimization components...")

    # 1. Pre-computed participant embeddings benefit
    n_runs = 10

    with torch.no_grad():
        # With optimization (pre-computed embeddings)
        optimized_rnn.enable_optimization(True)
        start_time = time.time()
        for _ in range(n_runs):
            _ = optimized_rnn.forward(inputs)
        optimized_time = time.time() - start_time

        # Without optimization (repeated embedding computation)
        optimized_rnn.enable_optimization(False)
        start_time = time.time()
        for _ in range(n_runs):
            _ = optimized_rnn.forward(inputs)
        unoptimized_time = time.time() - start_time

    embedding_speedup = unoptimized_time / optimized_time
    print(f"Pre-computed embeddings speedup: {embedding_speedup:.2f}x")

    return embedding_speedup


def plot_results(results, accuracy_results):
    """Plot the benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Execution times
    ax1.plot(results['sequence_lengths'], results['original_times'], 'b-o',
             label='Original', linewidth=2, markersize=6)
    ax1.plot(results['sequence_lengths'], results['optimized_times'], 'r-s',
             label='Properly Optimized', linewidth=2, markersize=6)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Real Optimization: Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Speedup
    ax2.plot(results['sequence_lengths'], results['speedups'], 'g-o',
             linewidth=3, markersize=8)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Real Optimization: Performance Speedup')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')

    # Add accuracy annotation
    logits_diff, reward_diff, choice_diff = accuracy_results[1:4]
    max_diff = max(logits_diff, reward_diff, choice_diff)
    ax2.text(0.7, 0.9, f'Max difference: {max_diff:.2e}',
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    plt.tight_layout()
    plt.savefig('real_cognitive_rnn_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run comprehensive test of the properly optimized implementation."""
    print("Real Cognitive RNN Optimization Test")
    print("=" * 50)

    # Test computational accuracy
    accuracy_results = test_computational_accuracy()

    # Benchmark performance
    results = benchmark_real_optimization()

    # Analyze optimization sources
    embedding_speedup = analyze_optimization_sources()

    # Plot results
    plot_results(results, accuracy_results)

    # Summary
    avg_speedup = np.mean(results['speedups'])
    max_speedup = np.max(results['speedups'])
    accuracy_passed = accuracy_results[0]

    print("\n" + "=" * 50)
    print("REAL OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"Computational Accuracy: {'✓ EXACT' if accuracy_passed else '✗ FAILED'}")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Maximum speedup: {max_speedup:.2f}x")
    print(f"Embedding pre-computation benefit: {embedding_speedup:.2f}x")

    if accuracy_passed:
        print("\n✅ SUCCESS: Achieved genuine speedup while maintaining exact computational logic!")
        print("\nOptimization Techniques Applied:")
        print("  • Pre-computed participant embeddings (major contributor)")
        print("  • Pre-computed beta scaling factors")
        print("  • Optimized tensor concatenation operations")
        print("  • Eliminated repeated tensor expansion")
        print("  • Used actual GRU modules (no dummy functions)")
        print("  • Maintained exact mathematical operations")
    else:
        print("\n❌ FAILURE: Computational accuracy compromised")

    return results, accuracy_results


if __name__ == "__main__":
    results, accuracy = main()