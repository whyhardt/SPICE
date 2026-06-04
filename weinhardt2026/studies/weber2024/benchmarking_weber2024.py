import torch

from spice import SpiceDataset, csv_to_dataset, split_data_along_sessiondim


def get_dataset(path_data: str = None, test_sessions: tuple[int] = None):
    
    dataset = csv_to_dataset(
        file = path_data,
        df_participant_id='participant',
        df_experiment_id='experiment',
        df_choice='choice',
        df_feedback='reward',
        df_block='block',
        additional_inputs=['laserRotation', 'shieldRotation', 'totalReward'],
        timeshift_additional_inputs=True,
    )
    
    # restructure data to have only two actions (stay, move) instead of three (stay, move_clockwise, move_counter_clockwise)
    move = dataset.xs[..., 1:3].sum(dim=-1, keepdim=True)
    rewards_move = dataset.xs[..., 4:6].nan_to_num(0).sum(dim=-1, keepdim=True)
    move_ys = dataset.ys[..., 1:3].sum(dim=-1, keepdim=True)
    # create restructured dataset
    xs = torch.concat((dataset.xs[..., :1], move, dataset.xs[..., 3:4], rewards_move, dataset.xs[..., 6:]), dim=-1)
    ys = torch.concat((dataset.ys[..., :1], move_ys), dim=-1)
    dataset = SpiceDataset(xs, ys)

    if test_sessions is not None:
        dataset_train, dataset_test = split_data_along_sessiondim(dataset, test_sessions)
    else:
        dataset_train, dataset_test = dataset, None
        
    return dataset_train, dataset_test