import argparse

import torch

from spice.estimator import SpiceEstimator
from spice.utils.convert_dataset import convert_dataset
from spice.resources.rnn_utils import split_data_along_sessiondim, split_data_along_timedim
from spice.resources.bandits import BanditsDrift_eckstein2024
from spice.precoded import Weinhardt2025RNN, WEINHARDT_2025_CONFIG, BufferWorkingMemoryRNN, BUFFER_WORKING_MEMORY_CONFIG


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Trains a SPICE-RNN on behavioral data to uncover the underlying Q-Values via different cognitive modules.')

    parser.add_argument('--model', type=str, default=None, help='Model name to load from and/or save to parameters of RNN')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset')

    # data and training parameters
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--l1', type=float, default=0, help='L1 Reg of the RNNs participant embedding')
    parser.add_argument('--l2', type=float, default=0, help='L2 Reg of the RNNs flexible modules (excl. embeddings)')
    parser.add_argument('--time_train_test_ratio', type=float, default=None, help='Ratio of training data; Can also be a comma-separated list of integeres to indicate testing sessions.')
    parser.add_argument('--session_train_test_ratio', type=str, default=None, help='Ratio of training data; Can also be a comma-separated list of integeres to indicate testing sessions.')
    
    args = parser.parse_args()
    
    args.model = "weinhardt2025/params/eckstein2024/rnn_eckstein2024_test24.pkl"
    args.data = "weinhardt2025/data/eckstein2024/eckstein2024.csv"
    # args.time_train_test_ratio = 0.8
    args.session_train_test_ratio = "2,4"
    # args.l1 = 0.0001
    args.l2 = 0.0005
    args.epochs = 8192
    
    dataset = convert_dataset(
        file=args.data,
        df_participant_id='s_id',
        df_block='block',
        df_choice='action',
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
    
    simulation_environment = BanditsDrift_eckstein2024(sigma=0.2, n_actions=4)
    
    estimator = SpiceEstimator(
        fit_spice=False,
        rnn_class=BufferWorkingMemoryRNN,
        spice_config=BUFFER_WORKING_MEMORY_CONFIG,
        n_actions=n_actions,
        n_participants=n_participants,
        epochs=args.epochs,
        bagging=True,
        scheduler=False,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        train_test_ratio=args.time_train_test_ratio if args.time_train_test_ratio else args.session_train_test_ratio,
        l1_weight_decay=args.l1,
        l2_weight_decay=args.l2,
        dropout=0.5,
        learning_rate=1e-3,
        use_optuna=True,
        spice_library_polynomial_degree=2,
        simulation_environment=simulation_environment,
        n_sessions_off_policy=1,
        # spice_optim_regularization=0.01,
        # spice_optim_threshold=0.05,
        optuna_threshold=0.,
        save_path_rnn=args.model,
        save_path_spice=args.model.replace('rnn', 'spice'),
    )
    
    # estimator.fit(dataset_train.xs, dataset_train.ys, data_test=dataset_test.xs, target_test=dataset_test.ys)
    
    # # estimator.load_rnn_model(args.model)
    # # estimator.load_spice_model(args.model.replace('rnn', 'spice'))
    
    participant_id = None
    estimator.load_rnn_model(args.model)
    estimator.fit_spice(dataset_train.xs, dataset_train.ys, participant_id=participant_id)
    
    estimator.print_spice_model(participant_id=participant_id)
    # # estimator.print_spice_model(participant_id=130)
    # # estimator.print_spice_model(participant_id=250)
    
    # import matplotlib.pyplot as plt
    # from spice.utils.plotting import plot_session
    # agents = {'rnn': estimator.rnn_agent, 'spice': estimator.spice_agent}
    # fig, axs = plot_session(agents, dataset_train.xs[participant_id])
    # plt.show()
