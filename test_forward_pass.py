"""Test a single forward pass to check for NaN"""
import torch
from spice.utils.convert_dataset import convert_dataset
from spice.precoded import BufferWorkingMemoryRNN, BUFFER_WORKING_MEMORY_CONFIG

# Load dataset
dataset, _, _, _ = convert_dataset(
    file="weinhardt2025/data/dezfouli2019/dezfouli2019.csv",
    remove_failed_trials=True,
)

n_actions = dataset.ys.shape[-1]
n_participants = len(dataset.xs[..., -1].unique())

print(f"Dataset: {n_participants} participants, {n_actions} actions")
print(f"Dataset shape: {dataset.xs.shape}")

# Create model - convert SpiceConfig to dict
sindy_config_dict = {
    'rnn_modules': BUFFER_WORKING_MEMORY_CONFIG.rnn_modules,
    'control_parameters': BUFFER_WORKING_MEMORY_CONFIG.control_parameters,
    'library_setup': BUFFER_WORKING_MEMORY_CONFIG.library_setup,
    'filter_setup': BUFFER_WORKING_MEMORY_CONFIG.filter_setup,
}

model = BufferWorkingMemoryRNN(
    n_actions=n_actions,
    n_participants=n_participants,
    embedding_size=32,
    dropout=0.0,
    spice_config=sindy_config_dict,
    sindy_polynomial_degree=2,
    sindy_ensemble_size=10,
    use_sindy=False,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
)
model = model.to(model.device)

# Test with first batch
batch_size = min(64, dataset.xs.shape[0])
xs_batch = dataset.xs[:batch_size].to(model.device)
ys_batch = dataset.ys[:batch_size].to(model.device)

print(f"\nTesting forward pass with batch size {batch_size}...")
print(f"Input shape: {xs_batch.shape}")
print(f"NaN in input: {torch.isnan(xs_batch).any().item()}")

model.eval()
with torch.no_grad():
    try:
        output, state, sindy_loss = model(xs_batch, None, batch_first=True)

        print(f"\nForward pass completed!")
        print(f"Output shape: {output.shape}")
        print(f"NaN in output: {torch.isnan(output).any().item()}")
        print(f"NaN in sindy_loss: {torch.isnan(sindy_loss).any().item()}")

        # Check state
        print(f"\nChecking model state for NaN:")
        for key, val in state.items():
            has_nan = torch.isnan(val).any().item()
            print(f"  {key}: {'NaN detected!' if has_nan else 'OK'}")

        # Check output statistics
        print(f"\nOutput statistics:")
        print(f"  Min: {output[~torch.isnan(output)].min().item():.4f}")
        print(f"  Max: {output[~torch.isnan(output)].max().item():.4f}")
        print(f"  Mean: {output[~torch.isnan(output)].mean().item():.4f}")

    except Exception as e:
        print(f"ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
