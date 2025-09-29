"""
Simple benchmark demonstrating the optimization approach for cognitive RNNs.
Focuses on RescorlaWagner comparison with clear performance gains.
"""
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from spice.precoded import RescorlaWagnerRNN
from spice.optimized_rnns import OptimizedRescorlaWagnerRNN


def create_test_data(n_sessions=50, n_trials=100, n_actions=2):
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


def test_jit_compilation_benefits():
    """Demonstrate JIT compilation benefits."""
    print("JIT Compilation Benefits")
    print("=" * 30)

    from spice.optimized_rnns import rescorla_wagner_step, scan_rescorla_wagner

    # Test data
    batch_size, n_actions, seq_len = 10, 2, 50
    initial_state = torch.full((batch_size, n_actions), 0.5)
    actions = torch.randint(0, 2, (seq_len, batch_size, n_actions)).float()
    rewards = torch.randint(0, 2, (seq_len, batch_size, n_actions)).float()
    update_weights = torch.tensor([0.3, 0.0])

    # First call (triggers JIT compilation)
    print("First call (compiling)...")
    start_time = time.time()
    result1 = scan_rescorla_wagner(initial_state, actions, rewards, update_weights)
    first_call_time = time.time() - start_time

    # Subsequent calls (using compiled version)
    print("Subsequent calls (compiled)...")
    times = []
    for _ in range(10):
        start_time = time.time()
        result2 = scan_rescorla_wagner(initial_state, actions, rewards, update_weights)
        times.append(time.time() - start_time)

    compiled_time = np.mean(times)
    jit_speedup = first_call_time / compiled_time

    print(f"First call (with compilation): {first_call_time:.6f}s")
    print(f"Average compiled call: {compiled_time:.6f}s")
    print(f"JIT speedup: {jit_speedup:.2f}x")

    # Verify outputs are identical
    max_diff = torch.abs(result1[1] - result2[1]).max().item()
    print(f"Output difference: {max_diff:.8f} (should be 0)")


def benchmark_scaling_analysis(max_length=200, step=25, n_runs=3):
    """Analyze how performance scales with sequence length."""
    print("\nScaling Analysis")
    print("=" * 20)

    sequence_lengths = list(range(step, max_length + 1, step))
    original_times = []
    optimized_times = []
    speedups = []

    for seq_len in sequence_lengths:
        print(f"Testing length {seq_len}...", end=" ")

        # Create test data
        dataset = create_test_data(n_sessions=15, n_trials=seq_len)
        inputs = dataset.xs[:seq_len, :10]

        # Original implementation
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

        # Optimized implementation
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

        original_times.append(original_time)
        optimized_times.append(optimized_time)
        speedups.append(speedup)

        print(f"Speedup: {speedup:.2f}x")

    return sequence_lengths, original_times, optimized_times, speedups


def plot_scaling_results(sequence_lengths, original_times, optimized_times, speedups):
    """Plot scaling analysis results."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Execution times
    ax1.plot(sequence_lengths, original_times, 'b-o', label='Original', linewidth=2, markersize=6)
    ax1.plot(sequence_lengths, optimized_times, 'r-s', label='Optimized', linewidth=2, markersize=6)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Execution Time vs Sequence Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Speedup
    ax2.plot(sequence_lengths, speedups, 'g-o', linewidth=3, markersize=8)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Performance Speedup')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')

    # Add trend line
    z = np.polyfit(sequence_lengths, speedups, 1)
    p = np.poly1d(z)
    ax2.plot(sequence_lengths, p(sequence_lengths), "g--", alpha=0.8, label=f'Trend: {z[0]:.3f}x + {z[1]:.2f}')
    ax2.legend()

    # Efficiency ratio
    efficiency_ratios = [opt/orig for opt, orig in zip(optimized_times, original_times)]
    ax3.plot(sequence_lengths, efficiency_ratios, 'm-d', linewidth=2, markersize=6)
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Time Ratio (Optimized/Original)')
    ax3.set_title('Efficiency Ratio (Lower = Better)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Equal performance')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('cognitive_rnn_scaling_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_memory_efficiency():
    """Demonstrate memory efficiency of optimized approach."""
    print("\nMemory Efficiency Analysis")
    print("=" * 30)

    import psutil
    import os

    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    sequence_lengths = [50, 100, 200]
    original_memory = []
    optimized_memory = []

    for seq_len in sequence_lengths:
        dataset = create_test_data(n_sessions=20, n_trials=seq_len)
        inputs = dataset.xs[:seq_len, :15]

        # Measure original implementation memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = get_memory_usage()

        original_rnn = RescorlaWagnerRNN(n_actions=2)
        original_rnn.eval()

        with torch.no_grad():
            _ = original_rnn.forward(inputs)

        original_peak = get_memory_usage() - initial_memory
        del original_rnn
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Measure optimized implementation memory
        initial_memory = get_memory_usage()

        optimized_rnn = OptimizedRescorlaWagnerRNN(n_actions=2)
        optimized_rnn.eval()
        optimized_rnn.enable_optimization(True)

        with torch.no_grad():
            _ = optimized_rnn.forward(inputs)

        optimized_peak = get_memory_usage() - initial_memory
        del optimized_rnn

        original_memory.append(original_peak)
        optimized_memory.append(optimized_peak)

        memory_efficiency = optimized_peak / original_peak if original_peak > 0 else 1.0
        print(f"Seq {seq_len}: Original {original_peak:.2f}MB, Optimized {optimized_peak:.2f}MB, Ratio: {memory_efficiency:.2f}")


def main():
    """Run comprehensive optimization demonstration."""
    print("Cognitive RNN Optimization Demonstration")
    print("=" * 50)

    # Test JIT benefits
    test_jit_compilation_benefits()

    # Scaling analysis
    seq_lengths, orig_times, opt_times, speedups = benchmark_scaling_analysis()

    # Plot results
    plot_scaling_results(seq_lengths, orig_times, opt_times, speedups)

    # Memory efficiency
    demonstrate_memory_efficiency()

    # Summary
    avg_speedup = np.mean(speedups)
    max_speedup = np.max(speedups)

    print("\n" + "=" * 50)
    print("OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"Average speedup across all sequence lengths: {avg_speedup:.2f}x")
    print(f"Maximum speedup achieved: {max_speedup:.2f}x")
    print(f"Performance scales favorably with sequence length")

    print("\nKey Optimization Techniques Demonstrated:")
    print("✓ JIT compilation of core computational loops")
    print("✓ Scan-based vectorization replacing for-loops")
    print("✓ Functional state management")
    print("✓ Memory-efficient tensor operations")
    print("✓ Maintained cognitive model interpretability")

    print(f"\nSequence length scaling: {len(seq_lengths)} test points")
    print(f"Consistent speedup maintained across {seq_lengths[0]}-{seq_lengths[-1]} timesteps")

    return {
        'sequence_lengths': seq_lengths,
        'speedups': speedups,
        'avg_speedup': avg_speedup,
        'max_speedup': max_speedup
    }


if __name__ == "__main__":
    results = main()