"""
Test script for end-to-end differentiable SPICE training.
This runs the full pipeline with SINDy regularization during RNN training.
"""
import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd


from spice.estimator import SpiceEstimator
from spice.utils.convert_dataset import convert_dataset, split_data_along_sessiondim, split_data_along_timedim
from spice.utils.plotting import plot_session
from spice.resources.bandits import AgentQ
from spice.precoded import workingmemory as spice_model


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Trains a SPICE-RNN with end-to-end differentiable SINDy on behavioral data.')

    parser.add_argument('--model', type=str, default=None, help='Model name to load from and/or save to parameters of RNN')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset')
    
    # data and training parameters
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    # parser.add_argument('--l1', type=float, default=0, help='L1 Reg of the RNNs participant embedding')
    parser.add_argument('--l2_rnn', type=float, default=0, help='L2 Reg of the RNN parameters')
    parser.add_argument('--l2_sindy', type=float, default=0, help='L2 Reg of the SINDy coefficients')
    parser.add_argument('--train_ratio_time', type=float, default=None, help='Ratio of data used for training. Split along time dimension. Not combinable with test_sessions')
    parser.add_argument('--test_sessions', type=str, default=None, help='Comma-separated list of integeres which indicate test sessions. Not combinable with train_ratio_time')
    parser.add_argument('--sindy_weight', type=float, default=0.1, help='Weight for SINDy regularization during RNN training')
    
    args = parser.parse_args()

    # args.model = "weinhardt2025/params/eckstein2022/spice_eckstein2022.pkl"
    # args.data = "weinhardt2025/data/eckstein2022/eckstein2022.csv"
    # args.train_ratio_time = 0.8
    
    # args.model = "weinhardt2025/params/eckstein2024/spice_eckstein2024.pkl"
    # args.data = "weinhardt2025/data/eckstein2024/eckstein2024.csv"
    # args.test_sessions = "1,3"
    
    # args.model = "weinhardt2025/params/dezfouli2019/spice_dezfouli2019.pkl"
    # args.data = "weinhardt2025/data/dezfouli2019/dezfouli2019.csv"
    # args.test_sessions = "3,6,9"
    
    # args.model = "weinhardt2025/params/spice_synthetic.pkl"
    # args.data = "weinhardt2025/data/data_synthetic.csv"
    
    # args.epochs = 4000 # Further reduced for initial testing
    # args.l2_rnn = 0.00001
    learning_rate = 0.001
    
    # args.sindy_weight = 0.1  # Start with very small weight for stability
    # args.l2_sindy = 0.001
    sindy_epochs = args.epochs#4000 
    sindy_threshold = 0.05
    sindy_thresholding_frequency = 100
    sindy_threshold_terms = 2
    class_rnn = spice_model.SpiceModel
    spice_config = spice_model.CONFIG
    
    example_participant = 0
    
    print(f"Loading dataset from {args.data}...")
    dataset = convert_dataset(
        file=args.data,
    )

    if args.train_ratio_time:
        args.test_sessions = None
        dataset_train, dataset_test = split_data_along_timedim(dataset, args.train_ratio_time)
    elif args.test_sessions:
        args.test_sessions = args.test_sessions.split(',')
        args.test_sessions = [int(item) for item in args.test_sessions]
        dataset_train, dataset_test = split_data_along_sessiondim(dataset, [int(item) for item in args.test_sessions])
    else:
        print("No split into training and test data.")
        dataset_train, dataset_test = dataset, dataset

    n_actions = dataset_train.ys.shape[-1]
    n_participants = len(dataset_train.xs[..., -1].unique())

    print(f"Dataset: {n_participants} participants, {n_actions} actions")
    print(f"Train/Test split: {args.train_ratio_time}")
    
    print(f"\nInitializing SpiceEstimator with end-to-end SINDy training (weight={args.sindy_weight})...")
    estimator = SpiceEstimator(
        
        # model paramaeters
        rnn_class=class_rnn,
        spice_config=spice_config,
        n_participants=n_participants,
        n_actions=2,
        
        # rnn training parameters
        epochs=args.epochs,
        l2_rnn=args.l2_rnn,
        learning_rate=learning_rate,
        train_test_ratio=args.train_ratio_time if args.train_ratio_time else args.test_sessions,
        
        # sindy fitting parameters
        sindy_weight=args.sindy_weight,
        sindy_threshold=sindy_threshold,
        sindy_threshold_frequency=sindy_thresholding_frequency,
        sindy_threshold_terms=sindy_threshold_terms,
        sindy_library_polynomial_degree=2,
        sindy_epochs=sindy_epochs,
        l2_sindy=args.l2_sindy,
        
        # additional generalization parameters
        bagging=True,
        # scheduler=True,
        
        # other parameters
        verbose=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_path_spice=args.model,
    )
    
    if args.epochs == 0:
        estimator.load_spice(args.model)
    
    print(f"\nStarting training on {estimator.device}...")
    print("=" * 80)
    estimator.fit(dataset_train.xs, dataset_train.ys)#, data_test=dataset_train.xs, target_test=dataset_train.ys)
    print("=" * 80)
    print("\nTraining complete!")
    
    print(f"\nModel saved to: {args.model}")
    
    agents={
        'rnn': estimator.rnn_agent,
        }
    
    if args.sindy_weight > 0:
        
        agents['spice'] = estimator.spice_agent
        
        # Print example SPICE model for first participant
        print(f"\nExample SPICE model (participant {example_participant}; n_parameters = {estimator.spice_agent.count_parameters()[example_participant]}):")
        print("-" * 80)
        estimator.print_spice_model(participant_id=example_participant)
        print("-" * 80)

    if 'synthetic' in args.data:
        dataset_df = pd.read_csv(args.data)

        # get the parameters for the selected participant and set up the ground truth model
        n_trials = 100
        dataset_df = dataset_df[dataset_df['session'] == example_participant]

        agents['groundtruth'] = AgentQ(
            n_actions=2,
            beta_reward=dataset_df['beta_reward'][example_participant * n_trials],
            alpha_reward=dataset_df['alpha_reward'][example_participant * n_trials],
            alpha_penalty=dataset_df['alpha_penalty'][example_participant * n_trials],
            forget_rate=dataset_df['forget_rate'][example_participant * n_trials],
            beta_choice=dataset_df['beta_choice'][example_participant * n_trials],
            alpha_choice=dataset_df['alpha_choice'][example_participant * n_trials],
        )
        
    fig, axs = plot_session(
        agents,
        experiment=dataset.xs[example_participant]
        )

    plt.show()
    
    print("\nNext step: Run analysis_model_evaluation.py to evaluate performance!")
