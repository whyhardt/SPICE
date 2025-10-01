import argparse

import torch

from spice.estimator import SpiceEstimator
from spice.utils.convert_dataset import convert_dataset
from spice.resources.rnn_utils import split_data_along_sessiondim
from spice.precoded import Weinhardt2025RNN, WEINHARDT_2025_CONFIG


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Trains a SPICE-RNN on behavioral data to uncover the underlying Q-Values via different cognitive modules.')

    parser.add_argument('--model', type=str, default=None, help='Model name to load from and/or save to parameters of RNN')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset')

    # data and training parameters
    parser.add_argument('--l2', type=float, default=0, help='Learning rate of the RNN')
    parser.add_argument('--train_test_ratio', type=str, default="2", help='Ratio of training data; Can also be a comma-separated list of integeres to indicate testing sessions.')

    args = parser.parse_args()
    
    dataset_train = convert_dataset(
        file=args.data,
        df_participant_id='session',
        df_block='block',
        df_choice='choice',
        df_reward='reward',
    )[0]
    
    dataset_train.xs[..., -1] = 0
    
    # dataset_train, dataset_test = split_data_along_sessiondim(dataset, [int(args.train_test_ratio)])
    
    n_actions = dataset_train.ys.shape[-1]
    n_participants = len(dataset_train.xs[..., -1].unique())
    
    estimator = SpiceEstimator(
        rnn_class=Weinhardt2025RNN,
        spice_config=WEINHARDT_2025_CONFIG,
        n_actions=n_actions,
        n_participants=n_participants,
        epochs=1,
        bagging=True,
        scheduler=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        train_test_ratio=args.train_test_ratio,
        l2_weight_decay=args.l2,
        dropout=0.5,
        use_optuna=True,
        fit_spice=True,
        save_path_rnn=args.model,
        save_path_spice=args.model.replace('rnn', 'spice'),
    )
    
    estimator.fit(dataset_train.xs, dataset_train.ys)
