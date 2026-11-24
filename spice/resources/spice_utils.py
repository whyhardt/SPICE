import inspect
from typing import Callable, Union, Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch


class SpiceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        n_reward_features: int = None,
        sequence_length: int = None,
        stride: int = 1,
        device=None,
        continuous_action: bool = False,
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
        
        sequence_length = None if sequence_length == -1 else sequence_length
        self.sequence_length = sequence_length if sequence_length is not None else xs.shape[1]
        self.stride = stride
        
        if sequence_length is not None:
            xs, ys = self.set_sequences(xs, ys)
        self.device = device
        self.xs = xs.to(device)
        self.ys = ys.to(device)
        
        self.n_actions = self.ys.shape[-1]
        self.n_participants = self.xs[..., -1].max().int().item() + 1
        self.n_experiments = self.xs[..., -2].max().int().item() + 1
        self.n_blocks = self.xs[..., -3].max().int().item() + 1
        self.n_trials = self.xs.shape[1]
        self.n_timesteps = self.xs.shape[2]
        self.n_reward_features = n_reward_features if n_reward_features is not None else self.n_actions
        self.n_additional_inputs = self.xs.shape[-1] - (self.n_actions + self.n_reward_features + 5)
        self.continuous_action = continuous_action
        
    def normalize_column(self, index, range=None):
        """Normalize feature column(s) in xs to a target range.

        Args:
            index: Feature column index (int, slice, or list of ints).
            range: Target (min, max) tuple. If None, auto-detected from data:
                - (0, 1) if all values are non-negative
                - (-1, 0) if all values are non-positive
                - (-1, 1) if values span both signs

        Returns:
            self (for chaining).
        """
        data = self.xs[..., index]
        valid = data[~torch.isnan(data)]
        if valid.numel() == 0:
            return self
        d_min = valid.min().item()
        d_max = valid.max().item()
        if d_max == d_min:
            return self

        if range is not None:
            target_min, target_max = range
        else:
            target_min = -1 if d_min < 0 else 0
            target_max = 1 if d_max > 0 else 0
            if target_min == target_max:
                return self

        normalized = (data - d_min) / (d_max - d_min)
        self.xs[..., index] = normalized * (target_max - target_min) + target_min
        return self

    def normalize_rewards(self, range=None):
        """Normalize reward feature columns in xs.

        Convenience wrapper around normalize_column for the reward slice.
        """
        reward_start = self.n_actions
        reward_end = self.n_actions + self.n_reward_features
        return self.normalize_column(slice(reward_start, reward_end), range=range)
        
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
                 discovered_modules: Dict[str, Iterable[str]],
                 memory_state: Union[List[str], Dict[str, float]],
                 fixed_modules: Dict[str, Tuple[Callable, Iterable[str]]] = None,
                 constants: Dict[str, Optional[Callable]] = None,
                 additional_inputs: List[str] = None,
                 ):
        """
        Config class for SPICE model.

        Args:
            discovered_modules: Dictionary of modules whose equations are discovered (RNN + SINDy) and
                their input signals (without self-references)
            memory_state: Dictionary of memory state variables and their initial values
            fixed_modules: Dictionary of hand-coded modules as `name: (function, input_labels)`. The
                function receives the inputs given to `call_module` in the order of `input_labels`
                (its own state included explicitly, if needed) and returns the new state value.
                Use module-level functions (not lambdas) so the config stays picklable.
            constants: Dictionary of per-participant constants as `name: activation`; each is computed
                once per forward pass from the participant embedding (`activation(FFN(embedding))`,
                activation None = identity) and can be given as input to any module.
            additional_inputs: List of additional control signals in dataset. If None: No additional control signals available
        """

        self.discovered_modules = discovered_modules
        self.fixed_modules = {} if fixed_modules is None else dict(fixed_modules)
        self.constants = {} if constants is None else dict(constants)
        self.additional_inputs = additional_inputs if additional_inputs is not None else tuple([])

        self.control_signals = []
        for key in discovered_modules.keys():
            self.control_signals += discovered_modules[key]
            self.discovered_modules[key] = tuple(self.discovered_modules[key])
        self.control_signals = tuple(np.unique(np.array(self.control_signals)))
        self.modules = tuple(discovered_modules.keys())
        self.all_features = self.modules + self.control_signals

        if isinstance(memory_state, list):
            memory_state_dict = {}
            for state in memory_state:
                memory_state_dict[state] = 0.
            self.memory_state = memory_state_dict
        else:
            self.memory_state = memory_state

        if not self.check_discovered_modules(self.discovered_modules, self.all_features):
            raise ValueError('\nLibrary setup does not match feature list.')
        self.check_fixed_modules(self.fixed_modules)

        names = list(self.modules) + list(self.fixed_modules) + list(self.constants)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"SpiceConfigError: module names {duplicates} appear in more than one of discovered_modules, fixed_modules and constants.")

    @staticmethod
    def check_fixed_modules(fixed_modules: Dict[str, Tuple[Callable, Iterable[str]]]):
        for key, spec in fixed_modules.items():
            if not (isinstance(spec, tuple) and len(spec) == 2 and callable(spec[0])):
                raise ValueError(f"SpiceConfigError: fixed module {key} must be given as (function, input_labels).")
            function, input_labels = spec
            parameters = inspect.signature(function).parameters.values()
            if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters):
                continue
            n_args = len([p for p in parameters if p.default is p.empty])
            if n_args != len(input_labels):
                raise ValueError(f"SpiceConfigError: fixed module {key}: {function.__name__} takes {n_args} arguments but {len(input_labels)} input labels are given ({list(input_labels)}).")
            fixed_modules[key] = (function, tuple(input_labels))

    @staticmethod
    def check_discovered_modules(discovered_modules: Dict[str, List[str]], feature_names: List[str], verbose=False) -> bool:
        msg = '\n'
        for key in discovered_modules.keys():
            if key not in feature_names:
                msg += f'Key {key} not in feature_names.\n'
            else:
                for feature in discovered_modules[key]:
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
        self.feedback = None
        self.trials = None
        self.time_trial = None
        self.additional_inputs = None
        self.logits = None
        self.trials = None
        self.sindy_loss_timesteps = None
        self.mask_valid_trials = None