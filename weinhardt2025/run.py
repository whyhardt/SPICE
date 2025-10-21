"""
Test script for end-to-end differentiable SPICE training.
This runs the full pipeline with SINDy regularization during RNN training.
"""
import argparse
import torch

from spice.estimator import SpiceEstimator
from spice.utils.convert_dataset import convert_dataset, split_data_along_sessiondim, split_data_along_timedim
from spice.precoded import BufferWorkingMemoryRNN, BUFFER_WORKING_MEMORY_CONFIG


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

    # args.model = "weinhardt2025/params/eckstein2022/spice_eckstein2022.pkl"
    # args.data = "weinhardt2025/data/eckstein2022/eckstein2022.csv"
    # args.time_train_test_ratio = 0.8
    
    # args.model = "weinhardt2025/params/eckstein2024/spice_eckstein2024.pkl"
    # args.data = "weinhardt2025/data/eckstein2024/eckstein2024.csv"
    # args.session_train_test_ratio = "1,3"
    
    args.model = "weinhardt2025/params/dezfouli2019/spice_dezfouli2019_sindy_1.pkl"
    args.data = "weinhardt2025/data/dezfouli2019/dezfouli2019.csv"
    args.session_train_test_ratio = "3,6,9"
    
    args.epochs = 1000 # Further reduced for initial testing
    args.l2 = 0.01
    args.l1 = 0.
    dropout = 0.
    args.sindy_weight = 1  # Start with very small weight for stability
    sindy_threshold = 0.1
    sindy_thresholding_frequency = 100
    
    print(f"Loading dataset from {args.data}...")
    dataset = convert_dataset(
        file=args.data,
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
        dropout=dropout,
        learning_rate=1e-2,
        sindy_weight=args.sindy_weight,  # Enable end-to-end SINDy regularization
        spice_library_polynomial_degree=2,
        save_path_spice=args.model,
        sindy_threshold_frequency=sindy_thresholding_frequency,
        spice_optim_threshold=sindy_threshold,
    )
    
    if args.epochs == 0:
        estimator.load_spice(args.model)
    
    print(f"\nStarting training on {estimator.device}...")
    print("=" * 80)
    estimator.fit(dataset_train.xs, dataset_train.ys)#, data_test=dataset_train.xs, target_test=dataset_train.ys)
    print("=" * 80)
    print("\nTraining complete!")
    
    print(f"\nModel saved to: {args.model}")
    
    # Print example SPICE model for first participant
    example_participant = 0
    print(f"\nExample SPICE model (participant {example_participant}):")
    print("-" * 80)
    estimator.print_spice_model(participant_id=example_participant)
    print("-" * 80)

    # from spice.utils.plotting import plot_session
    # import matplotlib.pyplot as plt
    # agents = {'rnn': estimator.rnn_agent, 'spice': estimator.spice_agent}
    # fig, axs = plot_session(agents, dataset_train.xs[example_participant])
    # plt.show()
    
    print("\nNext step: Run analysis_model_evaluation.py to evaluate performance!")
