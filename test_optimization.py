"""
Test script for optimized RNN implementations.
Compares performance and validates correctness of scan and JIT optimizations.
"""
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from spice.precoded import RescorlaWagnerRNN, RESCOLA_WAGNER_CONFIG, StandardRNN, STANDARD_CONFIG
from spice.optimized_rnns import OptimizedRescorlaWagnerRNN, OptimizedStandardRNN
from spice.resources.bandits import BanditsDrift, AgentQ, create_dataset
from spice.utils.convert_dataset import convert_dataset



def test_correctness(dataset, tolerance=1e-4):
    """Test that optimized implementation produces same results as original."""
    print("Testing correctness...")

    # Create models
    original_rnn = StandardRNN(n_actions=2, n_participants=100)
    optimized_rnn = OptimizedStandardRNN(n_actions=2, n_participants=100)

    # Test single forward pass
    inputs = dataset.xs[:10]  # Use first 10 sessions

    # Original forward pass
    original_rnn.eval()
    with torch.no_grad():
        original_logits, original_state = original_rnn.forward(inputs)

    # Optimized forward pass
    optimized_rnn.eval()
    optimized_rnn.enable_optimization(True)
    with torch.no_grad():
        optimized_logits, optimized_state = optimized_rnn.forward(inputs)

    # Compare outputs
    logits_diff = torch.abs(original_logits - optimized_logits).max().item()
    state_diff = torch.abs(
        original_state['x_value_reward'] - optimized_state['x_value_reward']
    ).max().item()

    print(f"Maximum logits difference: {logits_diff:.6f}")
    print(f"Maximum state difference: {state_diff:.6f}")

    correctness_passed = logits_diff < tolerance and state_diff < tolerance
    print(f"Correctness test: {'PASSED' if correctness_passed else 'FAILED'}")

    return correctness_passed


def benchmark_performance(dataset, sequence_lengths=[10, 50, 100, 200], n_runs=10):
    """Benchmark performance across different sequence lengths."""
    print("\nBenchmarking performance...")

    results = {
        'sequence_lengths': sequence_lengths,
        'original_times': [],
        'optimized_times': [],
        'speedups': []
    }

    for seq_len in sequence_lengths:
        print(f"\nTesting sequence length: {seq_len}")

        # Create test inputs
        inputs = dataset.xs[:seq_len, :10]  # Use 10 sessions

        # Test original implementation
        original_rnn = StandardRNN(n_actions=2, n_participants=100)
        original_rnn.eval()

        start_time = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = original_rnn.forward(inputs)
        original_time = (time.time() - start_time) / n_runs

        # Test optimized implementation
        optimized_rnn = OptimizedStandardRNN(n_actions=2, n_participants=100)
        optimized_rnn.eval()
        optimized_rnn.enable_optimization(True)

        start_time = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = optimized_rnn.forward(inputs)
        optimized_time = (time.time() - start_time) / n_runs

        speedup = original_time / optimized_time

        results['original_times'].append(original_time)
        results['optimized_times'].append(optimized_time)
        results['speedups'].append(speedup)

        print(f"Original time: {original_time:.4f}s")
        print(f"Optimized time: {optimized_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

    return results


def plot_performance_results(results):
    """Plot performance comparison results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot execution times
    ax1.plot(results['sequence_lengths'], results['original_times'], 'b-o', label='Original')
    ax1.plot(results['sequence_lengths'], results['optimized_times'], 'r-o', label='Optimized')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.grid(True)

    # Plot speedup
    ax2.plot(results['sequence_lengths'], results['speedups'], 'g-o')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Performance Speedup')
    ax2.grid(True)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('optimization_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_training_compatibility(dataset):
    """Test that optimized model can still be trained."""
    print("\nTesting training compatibility...")

    from spice.estimator import SpiceEstimator
    from spice.optimized_rnns import OPTIMIZED_RESCORLA_WAGNER_CONFIG

    # Create estimator with optimized RNN
    estimator = SpiceEstimator(
        rnn_class=StandardRNN,
        spice_config=STANDARD_CONFIG,
        learning_rate=1e-2,
        epochs=128,  # Short training for testing
        verbose=False,
        n_participants=100,
    )

    estimator_jit = SpiceEstimator(
        rnn_class=OptimizedStandardRNN,
        spice_config=OPTIMIZED_RESCORLA_WAGNER_CONFIG,
        learning_rate=1e-2,
        epochs=128,  # Short training for testing
        verbose=False,
        n_participants=100,
    )
    
    # Test training (should fall back to original implementation)
    try:
        # small_dataset_x = dataset.xs[:50, :10]  # Small dataset for quick test
        # small_dataset_y = dataset.ys[:50, :10]
        print('\nTraining the normal RNN...')
        estimator.fit(dataset.xs, dataset.ys)
        print('\nTraining the optimized RNN...')
        estimator_jit.fit(dataset.xs, dataset.ys)
        print("Training compatibility: PASSED")
        return True
    except Exception as e:
        print(f"Training compatibility: FAILED - {e}")
        return False


def test_scan_functionality():
    """Test scan functionality."""
    print("\nTesting scan functionality...")

    try:
        from spice.optimized_rnns import rescorla_wagner_step, scan_rescorla_wagner

        # Test data
        batch_size, n_actions = 5, 2
        seq_len = 10

        value_state = torch.full((batch_size, n_actions), 0.5)
        actions = torch.randint(0, 2, (seq_len, batch_size, n_actions)).float()
        rewards = torch.randint(0, 2, (seq_len, batch_size, n_actions)).float()

        # Test single step
        update_weights = torch.tensor([0.3, 0.0])
        new_state, logits = rescorla_wagner_step(value_state, actions[0], rewards[0], update_weights)

        # Test scan
        final_state, all_logits = scan_rescorla_wagner(value_state, actions, rewards, update_weights)

        print(f"Scan output shape: {all_logits.shape}")
        print("Scan functionality: PASSED")
        return True
    except Exception as e:
        print(f"Scan functionality: FAILED - {e}")
        return False


def main():
    """Run all tests."""
    print("=== Optimized RNN Testing Suite ===")

    # Create test data
    print("Creating test data...")
    dataset = convert_dataset(
        file='weinhardt2025/data/q_agent_.csv', 
    )[0]
    print(f"Dataset shape: {dataset.xs.shape}")

    # Run tests
    test_results = {}

    # Test correctness
    test_results['correctness'] = test_correctness(dataset)

    # Test scan functionality
    test_results['scan'] = test_scan_functionality()

    # Test training compatibility
    test_results['training'] = test_training_compatibility(dataset)

    # Benchmark performance
    if test_results['correctness']:
        perf_results = benchmark_performance(dataset)
        plot_performance_results(perf_results)
        test_results['performance'] = perf_results

    # Summary
    print("\n=== Test Summary ===")
    for test_name, result in test_results.items():
        if test_name == 'performance':
            avg_speedup = np.mean(result['speedups'])
            print(f"{test_name.capitalize()}: Average speedup = {avg_speedup:.2f}x")
        else:
            status = "PASSED" if result else "FAILED"
            print(f"{test_name.capitalize()}: {status}")

    return test_results


if __name__ == "__main__":
    results = main()