"""
Diagnostic: per-module SINDy approximation error.

Loads a fitted SPICE model, runs a forward pass on the dataset, and reports
the SINDy reconstruction loss broken down by module. This helps identify
which modules have dynamics that are hard to approximate with polynomials.

Usage:
    python weinhardt2026/studies/castro2025/diagnose_sindy_loss.py \
        --module studies.castro2025.spice_castro2025_bias_5 \
        --model weinhardt2026/studies/castro2025/params/spice_castro2025_bias_5.pkl \
        --data weinhardt2026/studies/castro2025/data/eckstein2024.csv
"""

import sys
import os
import argparse
import importlib
from collections import defaultdict

import torch
import numpy as np

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim


def diagnose_sindy_loss(estimator, dataset, n_steps=None):
    """
    Run a forward pass and compute per-module SINDy loss.

    Instead of modifying BaseModel, we hook into compute_sindy_loss_for_module
    to intercept per-module losses before they get summed.

    Args:
        estimator: Fitted SpiceEstimator with loaded model.
        dataset: SpiceDataset to evaluate on.
        n_steps: Number of timesteps to evaluate (None = all).

    Returns:
        Dict mapping module_name -> {
            'loss_total': float (unnormalized MSE),
            'loss_per_item': float (MSE normalized by masked items),
            'n_calls': int,
            'loss_per_trial_per_item': list of floats (per-trial breakdown),
        }
    """
    model = estimator.model
    model.eval()
    model.use_sindy = False  # Force RNN path to get RNN predictions
    model.fit_sindy = True   # Enable SINDy loss computation

    # Storage for per-module losses
    module_losses = defaultdict(lambda: {
        'losses_reg': [],
        'losses_raw': [],    # unnormalized (before /n_modules)
        'n_calls': 0,
    })

    n_modules = len(model.submodules_rnn)

    # Monkey-patch compute_sindy_loss_for_module to record per-module losses
    original_fn = model.compute_sindy_loss_for_module

    def patched_compute_sindy_loss(module_name, h_current, h_next_rnn, controls,
                                    action_mask, participant_ids, experiment_ids,
                                    polynomial_degree=2):
        # Call original
        sindy_loss_reg, sindy_loss_fit = original_fn(
            module_name=module_name,
            h_current=h_current,
            h_next_rnn=h_next_rnn,
            controls=controls,
            action_mask=action_mask,
            participant_ids=participant_ids,
            experiment_ids=experiment_ids,
            polynomial_degree=polynomial_degree,
        )

        # Also compute the raw (unnormalized) MSE for diagnostics
        with torch.no_grad():
            h_next_sindy = model.forward_sindy(
                h_current=h_current,
                key_module=module_name,
                participant_ids=participant_ids,
                experiment_ids=experiment_ids,
                controls=controls,
                polynomial_degree=polynomial_degree,
            )
            diff = (h_next_rnn.detach() - h_next_sindy.detach()) ** 2

            if action_mask is not None:
                masked_diff = torch.where(action_mask == 1, diff, torch.zeros_like(diff))
                n_masked = action_mask.sum(dim=-1).clamp(min=1)
                raw_mse = (masked_diff.sum(dim=-1) / n_masked).mean().item()
            else:
                raw_mse = diff.mean().item()

        module_losses[module_name]['losses_reg'].append(sindy_loss_reg.item())
        module_losses[module_name]['losses_raw'].append(raw_mse)
        module_losses[module_name]['n_calls'] += 1

        return sindy_loss_reg, sindy_loss_fit

    model.compute_sindy_loss_for_module = patched_compute_sindy_loss

    # Run forward pass
    xs = dataset.xs.to(model.device)
    if n_steps is not None:
        xs = xs[:, :n_steps]

    with torch.no_grad():
        # Need gradients disabled for most things but SINDy loss computation
        # needs to happen -- we handle this by calling with training=True temporarily
        was_training = model.training
        model.training = True  # So call_module computes SINDy loss
        model.fit_sindy = True

        logits, _ = model(xs)

        model.training = was_training

    # Restore original function
    model.compute_sindy_loss_for_module = original_fn
    model.fit_sindy = False

    # Aggregate results
    results = {}
    for module_name in model.submodules_rnn:
        if module_name in module_losses:
            data = module_losses[module_name]
            results[module_name] = {
                'loss_normalized': np.mean(data['losses_reg']),   # /n_modules (as used in training)
                'loss_raw': np.mean(data['losses_raw']),          # actual MSE
                'n_calls': data['n_calls'],
                'loss_per_trial': data['losses_raw'],             # per-trial breakdown
            }
        else:
            results[module_name] = {
                'loss_normalized': 0.0,
                'loss_raw': 0.0,
                'n_calls': 0,
                'loss_per_trial': [],
            }

    return results


def print_diagnosis(results, model):
    """Print per-module SINDy loss breakdown."""
    n_modules = len(model.submodules_rnn)

    print("\n" + "=" * 80)
    print("PER-MODULE SINDY APPROXIMATION ERROR")
    print("=" * 80)

    # Sort by raw loss (descending)
    sorted_modules = sorted(results.items(), key=lambda x: x[1]['loss_raw'], reverse=True)

    total_raw = sum(r['loss_raw'] for _, r in sorted_modules)

    print(f"\n{'Module':<35} {'Raw MSE':>10} {'% of Total':>10} {'Calls':>6}")
    print("-" * 65)

    for module_name, data in sorted_modules:
        pct = (data['loss_raw'] / total_raw * 100) if total_raw > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"{module_name:<35} {data['loss_raw']:>10.6f} {pct:>9.1f}% {data['n_calls']:>6}  {bar}")

    print("-" * 65)
    print(f"{'TOTAL':<35} {total_raw:>10.6f} {'100.0%':>10}")

    # Chosen vs not-chosen comparison
    print("\n" + "-" * 80)
    print("CHOSEN vs NOT-CHOSEN COMPARISON")
    print("-" * 80)

    seen = set()
    for module_name, data in sorted_modules:
        if module_name in seen:
            continue

        # Find complementary module
        if '_chosen' in module_name and '_not_chosen' not in module_name:
            complement = module_name.replace('_chosen', '_not_chosen')
        elif '_not_chosen' in module_name:
            complement = module_name.replace('_not_chosen', '_chosen')
        else:
            continue

        if complement in results:
            seen.add(module_name)
            seen.add(complement)
            ratio = data['loss_raw'] / results[complement]['loss_raw'] if results[complement]['loss_raw'] > 0 else float('inf')
            print(f"  {module_name}: {data['loss_raw']:.6f}")
            print(f"  {complement}: {results[complement]['loss_raw']:.6f}")
            print(f"  ratio: {ratio:.2f}x")
            print()

    # Per-module SINDy equations (for context)
    print("-" * 80)
    print("ACTIVE SINDY TERMS PER MODULE")
    print("-" * 80)

    for module_name in model.submodules_rnn:
        if module_name not in model.sindy_coefficients:
            continue
        terms = model.sindy_candidate_terms[module_name]
        presence = model.sindy_coefficients_presence[module_name]
        coeffs = model.sindy_coefficients[module_name]

        # Count active terms across ensemble member 0, averaged over participants
        active_per_participant = presence[0].sum(dim=-1)  # [P, X]
        mean_active = active_per_participant.float().mean().item()

        # Get which terms are active for any participant (union)
        any_active = presence[0].any(dim=0).any(dim=0)  # [n_terms]
        active_terms = [t for t, a in zip(terms, any_active) if a]

        print(f"  {module_name}: {mean_active:.1f} avg active terms")
        if active_terms:
            print(f"    terms: {', '.join(active_terms)}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose per-module SINDy approximation error.')
    parser.add_argument('--module', type=str, required=True, help='SPICE model module path')
    parser.add_argument('--model', type=str, required=True, help='Path to fitted .pkl model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--test_blocks', type=str, default='1,3', help='Test sessions (comma-separated)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'both'],
                        help='Which split to diagnose')
    parser.add_argument('--n_steps', type=int, default=None, help='Max timesteps to evaluate')
    args = parser.parse_args()

    # Load data
    dataset = csv_to_dataset(file=args.data)
    dataset.normalize_rewards()

    test_blocks = [int(s) for s in args.test_blocks.split(',')]
    dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)

     # --------------------------------------------------------------------------------------------
    # RAPID PROTOTYPING
    from spice import SpiceDataset

    # keep only 100 timesteps
    dataset_train = SpiceDataset(dataset_train.xs[:, :100], dataset_train.ys[:, :100])

    # keep only 100 participants for rapid prototyping
    keep_participants = torch.arange(0, 50)

    def keep_subset(dataset, subset):
        participant_ids = dataset.xs[:, 0, 0, -1]
        mask = torch.isin(participant_ids, subset)
        return SpiceDataset(dataset.xs[mask], dataset.ys[mask])

    dataset_train = keep_subset(dataset_train, keep_participants)
    dataset_test = keep_subset(dataset_test, keep_participants)    
    # --------------------------------------------------------------------------------------------
    
    
    # Load model
    spice_module = importlib.import_module(args.module)
    spice_model = spice_module.SpiceModel
    spice_config = spice_module.CONFIG

    n_actions = dataset_train.n_actions
    n_participants = dataset_train.n_participants
    n_experiments = len(dataset_train.xs[..., -2].unique())

    estimator = SpiceEstimator(
        spice_class=spice_model,
        spice_config=spice_config,
        n_participants=n_participants,
        n_experiments=n_experiments,
        n_actions=n_actions,
        sindy_library_polynomial_degree=2,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    estimator.load_spice(args.model)

    # Run diagnosis
    if args.split in ('train', 'both'):
        print("\n>>> TRAIN SET <<<")
        results_train = diagnose_sindy_loss(estimator, dataset_train, n_steps=args.n_steps)
        print_diagnosis(results_train, estimator.model)

    if args.split in ('test', 'both'):
        print("\n>>> TEST SET <<<")
        results_test = diagnose_sindy_loss(estimator, dataset_test, n_steps=args.n_steps)
        print_diagnosis(results_test, estimator.model)

    if args.split == 'both':
        # Compare train vs test per module
        print("\n" + "=" * 80)
        print("TRAIN vs TEST COMPARISON (generalization)")
        print("=" * 80)
        print(f"\n{'Module':<35} {'Train MSE':>10} {'Test MSE':>10} {'Ratio':>8}")
        print("-" * 65)
        for module_name in estimator.model.submodules_rnn:
            train_loss = results_train[module_name]['loss_raw']
            test_loss = results_test[module_name]['loss_raw']
            ratio = test_loss / train_loss if train_loss > 0 else float('inf')
            flag = " <<<" if ratio > 2.0 else ""
            print(f"{module_name:<35} {train_loss:>10.6f} {test_loss:>10.6f} {ratio:>7.2f}x{flag}")
