#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch
import pandas as pd
from spice.estimator import SpiceEstimator
from spice.precoded import Weinhardt2025LibraryRNN, WEINHARDT_2025_CONFIG

def main():
    print("Testing library-based RNN with SpiceEstimator...")

    # Load data
    df = pd.read_csv('weinhardt2025/data/eckstein2022.csv')

    # Use subset for testing
    n_test_participants = 10
    test_sessions = df['session'].unique()[:n_test_participants]
    test_df = df[df['session'].isin(test_sessions)].copy()

    print(f"Using {len(test_df)} trials from {n_test_participants} participants")

    # Create SpiceEstimator with library-based RNN
    device = torch.device('cpu')  # Force CPU for testing

    estimator = SpiceEstimator(
        rnn_class=Weinhardt2025LibraryRNN,
        spice_config=WEINHARDT_2025_CONFIG,
        n_actions=2,
        n_participants=n_test_participants,
        epochs=1,
        learning_rate=0.001,
        batch_size=32,
        device=device,
        embedding_size=16,  # Smaller for testing
        degree=2,           # Polynomial degree
    )

    print(f"SpiceEstimator created with library-based RNN")

    # Fit the model (1 epoch for testing)
    try:
        estimator.fit(
            data=test_df,
            session_col='session',
            choice_col='choice',
            reward_col='reward'
        )

        print("✓ Model fitting completed successfully!")

        # Test prediction
        predictions = estimator.predict(test_df[:100])  # Test on first 100 trials
        print(f"✓ Predictions generated: {predictions.shape}")
        print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    except Exception as e:
        print(f"✗ Error during fitting: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✓ Library-based RNN successfully integrated with SpiceEstimator!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)