"""
Debug script to check for NaN values in the converted dataset
"""
import torch
from spice.utils.convert_dataset import convert_dataset, split_data_along_sessiondim

print("Loading dataset...")
dataset = convert_dataset(file="weinhardt2025/data/dezfouli2019/dezfouli2019.csv")[0]

# Split data
session_train_test_ratio = [3, 6, 9]
dataset_train, dataset_test = split_data_along_sessiondim(dataset, session_train_test_ratio)

print(f"\nDataset shapes:")
print(f"Train xs: {dataset_train.xs.shape}")
print(f"Train ys: {dataset_train.ys.shape}")

# Check for NaN in xs and ys for each session
print(f"\n{'='*80}")
print("Checking for NaN values in dataset...")
print(f"{'='*80}")

for i in range(min(5, len(dataset_train.xs))):  # Check first 5 sessions
    xs_session = dataset_train.xs[i]
    ys_session = dataset_train.ys[i]

    # Find where data is valid (not padding)
    n_actions = dataset_train.ys.shape[-1]
    valid_mask_xs = ~torch.isnan(xs_session[:, :n_actions].sum(dim=-1))
    valid_mask_ys = ~torch.isnan(ys_session.sum(dim=-1))

    n_valid_xs = valid_mask_xs.sum().item()
    n_valid_ys = valid_mask_ys.sum().item()

    print(f"\nSession {i}:")
    print(f"  Valid timesteps in xs: {n_valid_xs}")
    print(f"  Valid timesteps in ys: {n_valid_ys}")

    if n_valid_xs != n_valid_ys:
        print(f"  WARNING: Mismatch! xs has {n_valid_xs} valid, ys has {n_valid_ys} valid")

    # Check if there are any NaN values within the valid range
    if n_valid_xs > 0:
        xs_valid = xs_session[:n_valid_xs]
        ys_valid = ys_session[:n_valid_ys]

        has_nan_xs = torch.isnan(xs_valid).any().item()
        has_nan_ys = torch.isnan(ys_valid).any().item()

        if has_nan_xs:
            print(f"  ERROR: Found NaN in valid xs data!")
            nan_locations_xs = torch.where(torch.isnan(xs_valid))
            print(f"  NaN locations in xs: {list(zip(nan_locations_xs[0].tolist()[:10], nan_locations_xs[1].tolist()[:10]))}")

        if has_nan_ys:
            print(f"  ERROR: Found NaN in valid ys data!")
            nan_locations_ys = torch.where(torch.isnan(ys_valid))
            print(f"  NaN locations in ys: {list(zip(nan_locations_ys[0].tolist()[:10], nan_locations_ys[1].tolist()[:10]))}")

        if not has_nan_xs and not has_nan_ys:
            print(f"  OK: No NaN in valid data")

    # Show boundary behavior
    if n_valid_xs > 0 and n_valid_xs < len(xs_session):
        print(f"  Last valid xs timestep ({n_valid_xs-1}): {xs_session[n_valid_xs-1, :n_actions]}")
        print(f"  Last valid ys timestep ({n_valid_ys-1}): {ys_session[n_valid_ys-1]}")
        print(f"  First invalid xs timestep ({n_valid_xs}): {xs_session[n_valid_xs, :n_actions]}")

print(f"\n{'='*80}")
print("Dataset check complete!")
print(f"{'='*80}")
