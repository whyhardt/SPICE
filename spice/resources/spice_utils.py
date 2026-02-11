from typing import Union, Dict, Iterable, List, Optional
import numpy as np
import torch


class SpiceDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        xs: torch.Tensor, 
        ys: torch.Tensor,
        normalize_features: tuple[int] = None, 
        sequence_length: int = None,
        stride: int = 1,
        device=None,
        ):
        """Initializes the dataset for training the RNN.

        Args:
            xs (torch.Tensor): Input features. Accepts 3D (sessions, outer_ts, features) which is auto-promoted
                to 4D (sessions, outer_ts, within_trial_ts, features) with within_trial_ts=1.
                Also accepts 4D directly for within-trial sequence data (e.g. DDM).
            ys (torch.Tensor): Targets. Same shape conventions as xs.
            device (torch.Device, optional): Torch device. If None, uses cpu.
        """
        
        if device is None:
            device = torch.device('cpu')
        
        # check for type of xs and ys
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(xs, dtype=torch.float32)
        if not isinstance(ys, torch.Tensor):
            ys = torch.tensor(ys, dtype=torch.float32)
        
        # check dimensions of xs and ys — target shape: (sessions, outer_ts, within_ts, features)
        if len(xs.shape) == 2:
            xs = xs.unsqueeze(0)
        if len(ys.shape) == 2:
            ys = ys.unsqueeze(0)
        if len(xs.shape) == 3:
            xs = xs.unsqueeze(2)  # add within_trial_timesteps=1
        if len(ys.shape) == 3:
            ys = ys.unsqueeze(2)
            
        if normalize_features is not None:
            if isinstance(normalize_features, int):
                normalize_features = tuple(normalize_features)
            for feature in normalize_features:
                xs[:, :, :, feature] = self.normalize(xs[:, :, :, feature])
        
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
        # slices along dim=1 (outer_ts / block_timesteps)
        xs_sequences = []
        ys_sequences = []
        for i in range(0, max(1, xs.shape[1]-self.sequence_length), self.stride):
            xs_sequences.append(xs[:, i:i+self.sequence_length])
            ys_sequences.append(ys[:, i:i+self.sequence_length])
        xs = torch.cat(xs_sequences, dim=0)
        ys = torch.cat(ys_sequences, dim=0)
        return xs, ys
    
    def __len__(self):
        return self.xs.shape[0]
    
    def __getitem__(self, idx):
        return self.xs[idx, :], self.ys[idx, :]
    

class SpiceConfig():
    def __init__(self,
                 library_setup: Dict[str, Iterable[str]],
                 memory_state: Union[List[str], Dict[str, float]],
                 states_in_logit: List[str] = None, 
                 ):
        """
        Config class for SPICE model.

        Args:
            library_setup: Dictionary of rnn modules and their input signals (without self-references)
            memory_state: Dictionary of memory state variables and their initial values
            states_in_logit: List of memory states which are included directly in the logit computation. If None: All states are used. 
        """
        
        self.library_setup = library_setup
        
        self.control_signals = []
        for key in library_setup.keys():
            self.control_signals += library_setup[key]
            self.library_setup[key] = tuple(self.library_setup[key])
        self.control_signals = tuple(np.unique(np.array(self.control_signals)))
        self.modules = tuple(library_setup.keys())                
        self.all_features = self.modules + self.control_signals
        
        if isinstance(memory_state, list):
            memory_state_dict = {}
            for state in memory_state:
                memory_state_dict[state] = 0.
            self.memory_state = memory_state_dict
        else:
            self.memory_state = memory_state
        
        if states_in_logit:
            # check that all states_in_logit actually appear in the memory state
            invalid_states = []
            for state in states_in_logit:
                if not state in memory_state:
                    invalid_states.append(state)
            if len(invalid_states) > 0:
                raise ValueError(f"SpiceConfigError: states_in_logit contains states ({invalid_states}) which are not present in the memory state ({list(memory_state.keys())}).")
            self.states_in_logit = states_in_logit
        else:
            self.states_in_logit = [state for state in memory_state]
            
        if not self.check_library_setup(self.library_setup, self.all_features):
            raise ValueError('\nLibrary setup does not match feature list.')
        
    @staticmethod
    def check_library_setup(library_setup: Dict[str, List[str]], feature_names: List[str], verbose=False) -> bool:
        msg = '\n'
        for key in library_setup.keys():
            if key not in feature_names:
                msg += f'Key {key} not in feature_names.\n'
            else:
                for feature in library_setup[key]:
                    if feature not in feature_names:
                        msg += f'Key {key}: Feature {feature} not in feature_names.\n'
        if msg != '\n':
            msg += f'Valid feature names are {feature_names}.\n'
            print(msg)
            return False
        else:
            if verbose:
                print('Library setup is valid. All keys and features appear in the provided list of features.')
            return True
        
        
class SpiceSignals:
    
    def __init__(self):
        
        self.participant_ids = None
        self.experiment_ids = None
        self.blocks = None
        self.actions = None
        self.rewards = None
        self.trials = None
        self.time_trial = None
        self.additional_inputs = None
        self.logits = None
        self.timesteps = None
        self.sindy_loss_timesteps = None
        self.mask_valid_trials = None