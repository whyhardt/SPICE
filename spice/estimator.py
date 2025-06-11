"""
SPICE training pipeline as a scikit-learn estimator
"""
import warnings
import time
import torch
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from typing import Dict, Optional, Tuple, Union, Iterable

from .resources import (
    AgentNetwork,
    AgentSpice,
    check_library_setup,
    fit_spice,
    DatasetRNN,
    BaseRNN,
)
from .resources.rnn_training import fit_model
from .resources.rnn_utils import load_checkpoint
from .resources.sindy_utils import load_spice, save_spice


warnings.filterwarnings("ignore")

class SpiceConfig():
    def __init__(self,
                 library_setup: Dict[str, Iterable[str]],
                 filter_setup: Dict[str, Iterable[Union[str, float, int, bool]]],
                 control_parameters: Iterable[str],
                 rnn_modules: Iterable[str]):
        self.library_setup = library_setup
        self.filter_setup = filter_setup
        self.control_parameters = control_parameters
        self.rnn_modules = rnn_modules

        self.spice_feature_list = rnn_modules + control_parameters

        if not check_library_setup(self.library_setup, self.spice_feature_list, verbose=True):
            raise ValueError('\nLibrary setup does not match feature list.')


class SpiceEstimator(BaseEstimator):
    """
    Scikit-learn estimator for fitting the SPICE model and making predictions.
    
    Combines an RNN for predicting behavioral choices with SPICE for discovering
    the underlying dynamical equations of cognitive mechanisms.
    """
    
    def __init__(
        self,
        
        # RNN class. Can be one of the precoded models in rnn.py or a custom implementation.
        rnn_class: BaseRNN,
        
        spice_config: SpiceConfig,
        list_signals: Optional[Iterable[str]] = ['x_V', 'c_a', 'c_r'],
        
        # RNN parameters
        hidden_size: int = 8,
        dropout: float = 0.25,

        # Data/Environment parameters
        n_actions: int = 2,
        n_participants: int = 0,
        n_experiments: int = 0,
        
        # RNN training parameters
        epochs: int = 128,
        bagging: bool = False,
        sequence_length: Optional[int] = -1,  # -1 for keeping the sequence length in the data to its original length, otherwise strided windows of length sequence_length,
        n_steps_per_call: Optional[int] = 16,  # number of timesteps in one backward-call; -1 for full sequence
        batch_size: Optional[int] = -1,  # -1 for a batch-size equal to the number of participants in the data
        learning_rate: Optional[float] = 5e-3,
        convergence_threshold: Optional[float] = 1e-7,
        device: Optional[torch.device] = torch.device('cpu'),
        scheduler: Optional[bool] = False, 
        train_test_ratio: Optional[float] = 1.,
        l1_weight_decay: Optional[float] = 1e-4,
        l2_weight_decay: Optional[float] = 1e-4,
        
        # SPICE training parameters
        spice_optim_threshold: Optional[float] = 0.03,
        spice_optim_regularization: Optional[float] = 1e-2,
        spice_library_polynomial_degree: Optional[int] = 1,
        spice_participant_id: Optional[int] = None,  # Set to participant id to fit to a single participant
        
        verbose: Optional[bool] = False,

        save_path_rnn: Optional[str] = None,
        save_path_spice: Optional[str] = None
    ):
        
        super(BaseEstimator, self).__init__()
        
        # Training parameters
        self.epochs = epochs
        self.bagging = bagging
        self.sequence_length = sequence_length
        self.n_steps_per_call = n_steps_per_call
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.scheduler = scheduler
        self.train_test_ratio = train_test_ratio
        self.device = device
        self.verbose = verbose
        self.l1_weight_decay = l1_weight_decay
        self.l2_weight_decay = l2_weight_decay

        # Save parameters
        self.save_path_rnn = save_path_rnn
        self.save_path_spice = save_path_spice

        # SPICE training parameters
        self.spice_optim_threshold = spice_optim_threshold
        self.spice_library_polynomial_degree = spice_library_polynomial_degree
        self.spice_optim_regularization = spice_optim_regularization
        self.spice_participant_id = spice_participant_id

        # Data parameters
        self.n_actions = n_actions
        self.n_participants = n_participants
        self.n_experiments = n_experiments
        
        # RNN parameters
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.rnn_agent = None
        self.spice_agent = None
        self.spice_features = None
        
        self.rnn_model = rnn_class(
            n_actions=n_actions,
            list_signals=spice_config.spice_feature_list,
            hidden_size=hidden_size, 
            n_participants=n_participants,
            n_experiments=n_experiments
        ).to(device)

        self.spice_library_config = spice_config.library_setup
        self.spice_filter_config = spice_config.filter_setup
        self.control_parameters = spice_config.control_parameters
        self.rnn_modules = spice_config.rnn_modules

        self.spice_feature_list = spice_config.spice_feature_list

        self.optimizer_rnn = torch.optim.Adam(self.rnn_model.parameters(), lr=learning_rate)
    
    def fit(self, conditions: np.ndarray, targets: np.ndarray):
        """
        Fit the RNN and SPICE models to given data.
        
        Args:
            conditions: Array of shape (n_participants, n_trials, n_features)
            targets: Array of shape (n_participants, n_trials, n_actions)
        """
        
        dataset = DatasetRNN(conditions, targets)
        start_time = time.time()
        
        # ------------------------------------------------------------------------
        # Fit RNN
        # ------------------------------------------------------------------------
        
        if self.verbose:
            print('\nTraining the RNN...')
        
        batch_size = conditions.shape[0] if self.batch_size == -1 else self.batch_size
        
        rnn_model, rnn_optimizer, _ = fit_model(
            model=self.rnn_model,
            dataset_train=dataset,
            optimizer=self.optimizer_rnn,
            convergence_threshold=self.convergence_threshold,
            epochs=self.epochs,
            batch_size=batch_size,
            bagging=self.bagging,
            scheduler=self.scheduler,
            n_steps=self.n_steps_per_call,
            l1_weight_decay=self.l1_weight_decay,
            l2_weight_decay=self.l2_weight_decay,
            verbose=self.verbose,
        )

        rnn_model.eval()
        self.rnn_model = rnn_model
        self.rnn_optimizer = rnn_optimizer
        self.rnn_agent = AgentNetwork(rnn_model, self.n_actions, deterministic=True, device=self.device)
        
        if self.verbose:
            print('\nRNN training finished.')
            print(f'Training took {time.time() - start_time:.2f} seconds.')

        if self.save_path_rnn is not None:
            print(f'Saving RNN model to {self.save_path_rnn}...')
            self.save_spice(self.save_path_rnn, None)
            print(f'RNN model saved to {self.save_path_rnn}')            
        
        # ------------------------------------------------------------------------
        # Fit SPICE
        # ------------------------------------------------------------------------
        
        self.spice_agent = {}
        self.spice_features = {}
        spice_modules = {rnn_module: {} for rnn_module in self.rnn_modules}

        self.spice_agent, self.spice_features = fit_spice(
            rnn_modules=self.rnn_modules,
            control_signals=self.control_parameters,
            agent_rnn=self.rnn_agent,
            data=dataset,
            polynomial_degree=self.spice_library_polynomial_degree,
            library_setup=self.spice_library_config,
            filter_setup=self.spice_filter_config,
            optimizer_threshold=self.spice_optim_threshold,
            optimizer_alpha=self.spice_optim_regularization,
            participant_id=self.spice_participant_id,
            verbose=self.verbose,
        )

        if self.verbose:
            print('SPICE training finished.')
            print(f'Training took {time.time() - start_time:.2f} seconds.')

        if self.save_path_spice is not None:
            print(f'Saving SPICE model to {self.save_path_spice}...')
            self.save_spice(None, self.save_path_spice)
            print(f'SPICE model saved to {self.save_path_spice}')


    
    def predict(self, conditions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using both RNN and SPICE models.
        
        Args:
            conditions: Array of shape (n_participants, n_trials, n_features)
            
        Returns:
            Tuple containing:
            - RNN predictions
            - SPICE predictions
        """
        
        # get rnn prediction about action probability
        conditions = torch.tensor(conditions, dtype=torch.float32, device=self.device)
        prediction_rnn = np.full((*conditions.shape[:-1], self.n_actions), np.nan).reshape(-1, self.n_actions)
        prediction_spice = np.full((*conditions.shape[:-1], self.n_actions), np.nan).reshape(-1, self.n_actions)
        mask = torch.sum(conditions[..., :self.n_actions].reshape(-1, self.n_actions), dim=-1, keepdim=False) != -2
        
        # rnn predictions
        prediction = self.rnn_agent._model(conditions, batch_first=True)[0].reshape(-1, self.n_actions)
        prediction = torch.nn.functional.softmax(prediction, dim=-1).detach().cpu().numpy()
        prediction_rnn[mask.detach().cpu().numpy()] = prediction[mask]
        prediction_rnn = prediction_rnn.reshape(*conditions.shape[:-1], self.n_actions)
        
        # SPICE predictions
        prediction = self.spice_agent._model(conditions, batch_first=True)[0].reshape(-1, self.n_actions)
        prediction = torch.nn.functional.softmax(prediction, dim=-1).detach().cpu().numpy()
        prediction_spice[mask.detach().cpu().numpy()] = prediction[mask]
        prediction_spice = prediction_spice.reshape(*conditions.shape[:-1], self.n_actions)
        
        return prediction_rnn, prediction_spice

    def get_spice_features(self) -> Dict:
        """
        Get the learned SPICE features and equations.
        
        Returns:
            Dictionary containing features and equations for each agent/model
        """
        if self.spice_features is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        return self.spice_features
    
    def get_spice_agents(self) -> Dict:
        """
        Get the trained SPICE agents.
        
        Returns:
            Dictionary containing SPICE agents
        """
        if self.spice_agent is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        return self.spice_agent

    def get_participant_embeddings(self) -> Dict:
        if hasattr(self.rnn_model, 'participant_embedding'):
            participant_ids = torch.arange(self.n_participants, device=self.device, dtype=torch.int32).view(-1, 1)
            embeddings = self.rnn_model.participant_embedding(participant_ids)
            return {participant_id.item(): embeddings[participant_id, 0] for participant_id in participant_ids}
        else:
            print(f'RNN model has no participant_embedding module.')
            return None
        
    def load_rnn_model(self, path_model: str, deterministic: bool = True):
        self.rnn_model, self.rnn_optimizer = load_checkpoint(path_model, self.rnn_model, self.optimizer_rnn)
        self.rnn_agent = AgentNetwork(self.rnn_model, self.n_actions, deterministic=deterministic, device=self.device)
        
    def load_spice_model(self, path_spice: str, deterministic: bool = True):
        spice_modules = load_spice(path_spice)
        self.spice_agent = AgentSpice(model_rnn=self.rnn_agent._model, sindy_modules=spice_modules, n_actions=self.rnn_agent._n_actions, deterministic=deterministic)

    def load_spice(self, path_rnn: str, path_spice: str, deterministic: bool = True):
        """
        Load the RNN and SPICE models from the given paths.
        
        Args:
            path_rnn: Path to the RNN model
            path_spice: Path to the SPICE model
            deterministic: Whether to use a deterministic model (default: True)
        """
        if path_rnn is not None:
            self.load_rnn_model(path_rnn, deterministic=deterministic)
        if path_spice is not None:
            self.load_spice_model(path_spice, deterministic=deterministic)

    def save_spice(self, path_rnn: str = None, path_spice: str = None):
        """
        Save the RNN and SPICE models to the given paths.
        If path_rnn is None, only the SPICE model will be saved (requires a fitted SPICE model including the RNN).
        If path_spice is None, only the RNN model will be saved (requires a fitted RNN model).
        
        Args:
            path_rnn: Path to the RNN model
            path_spice: Path to the SPICE model
        """
        if path_rnn is not None:
            # Save RNN model
            state_dict = {'model': self.rnn_model.state_dict(), 'optimizer': self.rnn_optimizer.state_dict()}
            torch.save(state_dict, path_rnn)
        if path_spice is not None:
            # Save SPICE model
            save_spice(self.spice_agent, path_spice)