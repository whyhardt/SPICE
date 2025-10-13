import torch
from torch.utils.data import Dataset
from typing import Union, List


class DatasetRNN(Dataset):
    def __init__(
        self, 
        xs: torch.Tensor, 
        ys: torch.Tensor,
        normalize_features: tuple[int] = None, 
        sequence_length: int = None,
        stride: int = 1,
        device=None,
        ):
        """Initializes the dataset for training the RNN. Holds information about the previous actions and rewards as well as the next action.
        Actions can be either one-hot encoded or indexes.

        Args:
            xs (torch.Tensor): Actions and rewards in the shape (n_sessions, n_timesteps, n_features)
            ys (torch.Tensor): Next action
            batch_size (Optional[int], optional): Sets batch size if desired else uses n_samples as batch size.
            device (torch.Device, optional): Torch device. If None, uses cuda if available else cpu.
        """
        
        if device is None:
            device = torch.device('cpu')
        
        # check for type of xs and ys
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(xs, dtype=torch.float32)
        if not isinstance(ys, torch.Tensor):
            ys = torch.tensor(ys, dtype=torch.float32)
        
        # check dimensions of xs and ys
        if len(xs.shape) == 2:
            xs = xs.unsqueeze(0)
        if len(ys.shape) == 2:
            ys = ys.unsqueeze(0)
            
        if normalize_features is not None:
            if isinstance(normalize_features, int):
                normalize_features = tuple(normalize_features)
            for feature in normalize_features:
                xs[:, :, feature] = self.normalize(xs[:, :, feature])
        
        # normalize data
        # x_std = xs.std(dim=(0, 1))
        # x_mean = xs.mean(dim=(0, 1))
        # xs = (xs - x_mean) / x_std
        # ys = (ys - x_mean) / x_std
        
        sequence_length = None if sequence_length == -1 else sequence_length
        self.sequence_length = sequence_length if sequence_length is not None else xs.shape[1]
        self.stride = stride
        
        if sequence_length is not None:
            xs, ys = self.set_sequences(xs, ys)
        self.device = device
        self.xs = xs.to(device)
        self.ys = ys.to(device)
        
    def normalize(self, data):
        x_min = torch.min(data)
        x_max = torch.max(data)
        return (data - x_min) / (x_max - x_min)
        
    def set_sequences(self, xs, ys):
        # sets sequences of length sequence_length with specified stride from the dataset
        xs_sequences = []
        ys_sequences = []
        for i in range(0, max(1, xs.shape[1]-self.sequence_length), self.stride):
            xs_sequences.append(xs[:, i:i+self.sequence_length, :])
            ys_sequences.append(ys[:, i:i+self.sequence_length, :])
        xs = torch.cat(xs_sequences, dim=0)
        ys = torch.cat(ys_sequences, dim=0)
        
        if len(xs.shape) == 2:
            xs = xs.unsqueeze(1)
            ys = ys.unsqueeze(1)
            
        return xs, ys
    
    def __len__(self):
        return self.xs.shape[0]
    
    def __getitem__(self, idx):
        return self.xs[idx, :], self.ys[idx, :]


def load_checkpoint(params_path, model, optimizer):
    # load trained parameters
    state_dict = torch.load(params_path, map_location=torch.device('cpu'))
    
    state_dict_model = state_dict['model']
    state_dict_optimizer = state_dict['optimizer']
    
    model.load_state_dict(state_dict_model)
    optimizer.load_state_dict(state_dict_optimizer)
    
    return model, optimizer


def parameter_file_naming(params_path, alpha_reward, alpha_penalty, alpha_counterfactual, confirmation_bias, forget_rate, beta_reward, alpha_choice, beta_choice, variance, verbose=False):
    # create name for corresponding rnn
  
    params_path += '_rnn'
    
    params_path += '_br' + str(beta_reward).replace('.', '')
    
    if alpha_reward > 0:
        params_path += '_a' + str(alpha_reward).replace('.', '')
    
    if alpha_penalty >= 0:
        params_path += '_ap' + str(alpha_penalty).replace('.', '')
    
    if alpha_counterfactual > 0:
        params_path += '_ac' + str(alpha_counterfactual).replace('.', '')
        
    if beta_choice > 0 and alpha_choice > 0:
        params_path += '_bch' + str(beta_choice).replace('.', '')
        params_path += '_ach' + str(alpha_choice).replace('.', '')
    
    if forget_rate > 0:
        params_path += '_f' + str(forget_rate).replace('.', '')
        
    if confirmation_bias > 0:
        params_path += '_cb' + str(confirmation_bias).replace('.', '')
    
    if not isinstance(variance, dict):
        if variance != 0:
            params_path += '_var' + str(variance).replace('.', '').replace('-1','Mean')
    else:
        params_path += '_varDict'
        
    # if non_binary_reward:
    #     params_path += '_nonbinary'
        
    params_path += '.pkl'
    
    if verbose:
        print(f'Automatically generated name for model parameter file: {params_path}.')
        
    return params_path


def split_data_along_timedim(dataset: DatasetRNN, split_ratio: float, device: torch.device = torch.device('cpu')):
    """Split the data along the time dimension (dim=1). 
    Each session (dim=0) can be of individual length and is therefore post-padded with -1.
    To split the data into training and testing samples according to the split_ratio each session's individual length has to be considered.
    E.g.: 
    1. session_length = len(session 0) -> 120
    2. split_index = int(split_ratio * session length) -> 96
    3. samples for training data -> 96
    4. samples for testing data -> 24 = 120 - 96    

    Args:
        data (torch.Tensor): Data containing all sessions in the shape (session, time, features)
        split_ratio (float): Float number indicating the ratio to be used as training data

    Returns:
        tuple(DatasetRNN, DatasetRNN): Training data and testing data splitted along time dimension (dim=1)
    """
    
    dim = 1
    xs, ys = dataset.xs, dataset.ys
    
    # Create a mask of non-zero elements
    non_zero_mask = (xs[..., 0] != -1).int()
    
    # Find cumulative sum along the specified dimension in reverse order
    cumsum = torch.cumsum(non_zero_mask, dim)
    
    # Find the index where the cumulative sum becomes 1 in the reversed array
    last_nonzero_indices = torch.argmax(cumsum, dim=dim)
    
    # compute the indeces at which the data is going to be splitted into training and testing data
    split_indices = (last_nonzero_indices * split_ratio).int()
    
    # initialize training data and testing data storages
    train_xs = torch.zeros((xs.shape[0], max(split_indices), xs.shape[2]), device=dataset.device) - 1
    test_xs = torch.zeros((xs.shape[0], max(last_nonzero_indices - split_indices), xs.shape[2]), device=dataset.device) - 1
    train_ys = torch.zeros((xs.shape[0], max(split_indices), ys.shape[2]), device=dataset.device) - 1
    test_ys = torch.zeros((xs.shape[0], max(last_nonzero_indices - split_indices), ys.shape[2]), device=dataset.device) - 1
    
    # get columns which had no -1 values in the first place to fill them up entirely in the training and testing data
    # necessary for e.g. participant-IDs because otherwise -1 will be passed to embedding layer -> Error
    example_session_id = torch.argmax((last_nonzero_indices < xs.shape[1]-1).int()).item()
    full_columns = xs[example_session_id, -1] != -1
    
    # fill up training and testing data
    for index_session in range(xs.shape[0]):
        train_xs[index_session, :split_indices[index_session]] = xs[index_session, :split_indices[index_session]]
        test_xs[index_session, :last_nonzero_indices[index_session]-split_indices[index_session]] = xs[index_session, split_indices[index_session]:last_nonzero_indices[index_session]]
        train_ys[index_session, :split_indices[index_session]] = ys[index_session, :split_indices[index_session]]
        test_ys[index_session, :last_nonzero_indices[index_session]-split_indices[index_session]] = ys[index_session, split_indices[index_session]:last_nonzero_indices[index_session]]

        # fill up non-"-1"-columns (only applicable for xs)
        train_xs[index_session, :, full_columns] = xs[index_session, :train_xs.shape[1], full_columns]
        test_xs[index_session, :, full_columns] = xs[index_session, :test_xs.shape[1], full_columns]
    
    return DatasetRNN(train_xs, train_ys, device=device), DatasetRNN(test_xs, test_ys, device=device)


def split_data_along_sessiondim(dataset: DatasetRNN, list_test_sessions: List[int] = None, device: torch.device = torch.device('cpu')):
    """Split the data along the time dimension (dim=1). 
    Each session (dim=0) can be of individual length and is therefore post-padded with -1.
    To split the data into training and testing samples according to the split_ratio each session's individual length has to be considered.
    E.g.: 
    1. session_length = len(session 0) -> 120
    2. split_index = int(split_ratio * session length) -> 96
    3. samples for training data -> 96
    4. samples for testing data -> 24 = 120 - 96    

    Args:
        data (torch.Tensor): Data containing all sessions in the shape (session, time, features)
        split_ratio (float): Float number indicating the ratio to be used as training data

    Returns:
        tuple(DatasetRNN, DatasetRNN): Training data and testing data splitted along time dimension (dim=1)
    """
    
    if list_test_sessions is not None:
        
        dim = 1
        xs, ys = dataset.xs.cpu(), dataset.ys.cpu()
        
        # get participant ids
        participants_ids = xs[:, 0, -1].unique()
        
        # get sessions ids
        session_ids = xs[:, 0, -3].unique()
        
        # set training sessions
        if list_test_sessions:
            session_ids_test = torch.tensor(list_test_sessions, dtype=torch.float32)
        else:
            session_ids_test = torch.tensor([])

        if all(session_ids[:-1] < 1):
            session_ids_test /= len(session_ids) - 1

        # build list of training sessions (only includes sessions that exist in data)
        session_ids_train = []
        for sid in session_ids:
            if not sid in session_ids_test:
                session_ids_train.append(sid)
        session_ids_train = torch.tensor(session_ids_train, dtype=torch.float32)     
        
        # setup new variables - use lists to collect only existing participant-session combinations
        train_xs_list, test_xs_list, train_ys_list, test_ys_list = [], [], [], []

        for pid in participants_ids:
            for sid in session_ids:
                mask_ids = torch.logical_and(xs[:, 0, -3] == sid, xs[:, 0, -1] == pid)
                if mask_ids.max():
                    if sid in session_ids_train:
                        train_xs_list.append(xs[mask_ids])
                        train_ys_list.append(ys[mask_ids])
                    elif sid in session_ids_test:
                        test_xs_list.append(xs[mask_ids])
                        test_ys_list.append(ys[mask_ids])
                    else:
                        raise ValueError("session id was not found in training nor test sessions.")

        # stack into tensors - only includes actual data, no empty rows
        train_xs = torch.cat(train_xs_list, dim=0) if train_xs_list else torch.zeros((0, *xs.shape[1:]))
        train_ys = torch.cat(train_ys_list, dim=0) if train_ys_list else torch.zeros((0, *ys.shape[1:]))
        test_xs = torch.cat(test_xs_list, dim=0) if test_xs_list else torch.zeros((0, *xs.shape[1:]))
        test_ys = torch.cat(test_ys_list, dim=0) if test_ys_list else torch.zeros((0, *ys.shape[1:]))

        return DatasetRNN(train_xs, train_ys, device=device), DatasetRNN(test_xs, test_ys, device=device)

    else:
        return dataset, dataset

def reshape_data_along_participantdim(dataset: DatasetRNN, device: torch.device = torch.device('cpu')):
    """Reshape the data along the participant dim.

    Args:
        dataset (DatasetRNN): current dataset of shape (participants*sessions*, trial, features)
        device (torch.device, optional): Defaults to torch.device('cpu').

    Returns:
        DatasetRNN: restructured dataset with shape (participant, session, trial, features)
    """

    xs, ys = dataset.xs.cpu(), dataset.ys.cpu()

    # get participant ids
    participants_ids = xs[:, 0, -1].unique()

    # setup new variables
    xs_new, ys_new = [], []
    n_sessions_max = 0

    # First pass: collect participant-level sessions and find max number of sessions
    for pid in participants_ids:
        mask_ids = xs[:, 0, -1] == pid
        xs_new.append(xs[mask_ids])
        ys_new.append(ys[mask_ids])
        n_sessions_max = max(n_sessions_max, len(xs[mask_ids]))

    # Second pass: pad all participants to have n_sessions_max sessions
    for i in range(len(xs_new)):
        if xs_new[i].shape[0] < n_sessions_max:
            # Create padding for missing sessions
            n_missing = n_sessions_max - xs_new[i].shape[0]
            pad_sessions_xs = torch.zeros((n_missing, *xs_new[i].shape[1:]), device=xs_new[i].device)
            # Set feature dimensions to -1 (except last 3 which are IDs)
            pad_sessions_xs[..., :-3] = -1
            # Set participant ID to match this participant
            pad_sessions_xs[..., -1] = xs_new[i][0, 0, -1]
            # Concatenate padding
            xs_new[i] = torch.cat((xs_new[i], pad_sessions_xs), dim=0)

            # Pad ys with -1
            pad_sessions_ys = torch.full((n_missing, *ys_new[i].shape[1:]), -1.0, device=ys_new[i].device)
            ys_new[i] = torch.cat((ys_new[i], pad_sessions_ys), dim=0)

    # setup new dataset with shape (participant, session, trial, features)
    return DatasetRNN(torch.stack(xs_new), torch.stack(ys_new), device=device)