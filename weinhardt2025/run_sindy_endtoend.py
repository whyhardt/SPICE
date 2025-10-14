"""
Test script for end-to-end differentiable SPICE training.
This runs the full pipeline with SINDy regularization during RNN training.
"""
import argparse
import torch

from spice.estimator import SpiceEstimator
from spice.utils.convert_dataset import convert_dataset
from spice.resources.rnn_utils import split_data_along_sessiondim, split_data_along_timedim
from spice.resources.bandits import BanditsDrift_eckstein2024, BanditsFlip_eckstein2022
from spice.precoded import Weinhardt2025RNN, WEINHARDT_2025_CONFIG, BufferWorkingMemoryRNN, BUFFER_WORKING_MEMORY_CONFIG


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Trains a SPICE-RNN with end-to-end differentiable SINDy on behavioral data.')

    parser.add_argument('--model', type=str, default=None, help='Model name to load from and/or save to parameters of RNN')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset')

    # data and training parameters
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--l1', type=float, default=0, help='L1 Reg of the RNNs participant embedding')
    parser.add_argument('--l2', type=float, default=0, help='L2 Reg of the RNNs flexible modules (excl. embeddings)')
    parser.add_argument('--time_train_test_ratio', type=float, default=None, help='Ratio of training data; Can also be a comma-separated list of integeres to indicate testing sessions.')
    parser.add_argument('--session_train_test_ratio', type=str, default=None, help='Ratio of training data; Can also be a comma-separated list of integeres to indicate testing sessions.')
    parser.add_argument('--sindy_weight', type=float, default=0.1, help='Weight for SINDy regularization during RNN training')

    args = parser.parse_args()

    # Default parameters for eckstein2022 dataset
    args.model = "weinhardt2025/params/eckstein2022/spice_eckstein2022.pkl"
    args.data = "weinhardt2025/data/eckstein2022/eckstein2022.csv"
    args.time_train_test_ratio = 0.8
    args.epochs = 1024  # Further reduced for initial testing
    args.l2 = 0.
    args.l1 = 0.0001
    args.sindy_weight = 1e-1  # Start with very small weight for stability
    
    print(f"Loading dataset from {args.data}...")
    dataset = convert_dataset(
        file=args.data,
        df_participant_id='session',
        df_block='block',
        df_choice='choice',
        df_reward='reward',
    )[0]

    if args.time_train_test_ratio:
        args.session_train_test_ratio = None
        dataset_train, dataset_test = split_data_along_timedim(dataset, args.time_train_test_ratio)
    elif args.session_train_test_ratio:
        args.session_train_test_ratio = args.session_train_test_ratio.split(',')
        args.session_train_test_ratio = [int(item) for item in args.session_train_test_ratio]
        dataset_train, dataset_test = split_data_along_sessiondim(dataset, [int(item) for item in args.session_train_test_ratio])
    else:
        dataset_train, dataset_test = dataset, dataset

    n_actions = dataset_train.ys.shape[-1]
    n_participants = len(dataset_train.xs[..., -1].unique())

    print(f"Dataset: {n_participants} participants, {n_actions} actions")
    print(f"Train/Test split: {args.time_train_test_ratio}")

    print(f"\nInitializing SpiceEstimator with end-to-end SINDy training (weight={args.sindy_weight})...")
    estimator = SpiceEstimator(
        rnn_class=BufferWorkingMemoryRNN,
        spice_config=BUFFER_WORKING_MEMORY_CONFIG,
        n_actions=n_actions,
        n_participants=n_participants,
        epochs=args.epochs,
        bagging=True,
        scheduler=False,  # Enable scheduler for better convergence
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        train_test_ratio=args.time_train_test_ratio if args.time_train_test_ratio else args.session_train_test_ratio,
        l1_weight_decay=args.l1,
        l2_weight_decay=args.l2,
        dropout=0.,
        learning_rate=1e-2,
        sindy_weight=args.sindy_weight,  # Enable end-to-end SINDy regularization
        spice_library_polynomial_degree=2,
        save_path_spice=args.model,
        sindy_threshold_frequency = 32,
        spice_optim_threshold=0.01,
    )
    
    estimator.load_spice(args.model)
    
    print(f"\nStarting training on {estimator.device}...")
    print("=" * 80)
    estimator.fit(dataset_train.xs, dataset_train.ys, data_test=dataset_train.xs, target_test=dataset_train.ys)
    # estimator.load_spice(args.model)
    print("=" * 80)
    print("\nTraining complete!")
    
    print(f"\nModel saved to: {args.model}")
    
    # Print example SPICE model for first participant
    print("\nExample SPICE model (participant 0):")
    print("-" * 80)
    estimator.print_spice_model(participant_id=0)
    print("-" * 80)

    from spice.utils.plotting import plot_session
    import matplotlib.pyplot as plt
    agents = {'rnn': estimator.rnn_agent, 'spice': estimator.spice_agent}
    fig, axs = plot_session(agents, dataset_train.xs[0])
    plt.show()
    
    print("\nNext step: Run analysis_model_evaluation.py to evaluate performance!")
