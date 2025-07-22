"""
SPICE training pipeline as a scikit-learn estimator
"""
import warnings
import time
import torch
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from typing import Dict, Optional, Tuple, Union, Iterable, List

from .resources.rnn_training import fit_model
from .resources.rnn_utils import load_checkpoint, DatasetRNN
from .resources.sindy_utils import load_spice, save_spice, check_library_setup
from .resources.sindy_training import fit_spice
from .resources.bandits import AgentNetwork, AgentSpice
from .resources.rnn import BaseRNN, ParameterModule


warnings.filterwarnings("ignore")

class SpiceConfig():
    def __init__(self,
                 library_setup: Dict[str, Iterable[str]],
                 filter_setup: Dict[str, Iterable[Union[str, float, int, bool]]],
                 control_parameters: Iterable[str],
                 rnn_modules: Iterable[str]):
        """
        Config class for SPICE model.

        Args:
            library_setup: Dictionary of library setup
            filter_setup: Dictionary of filter setup
            control_parameters: List of control parameters
            rnn_modules: List of RNN modules
        """
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
        
        # RNN class and SPICE configuration. Can be one of the precoded models in rnn.py or a custom implementation.
        rnn_class: BaseRNN,
        spice_config: SpiceConfig,
        
        # Data/Environment parameters
        n_actions: int = 2,
        n_participants: int = 0,
        n_experiments: int = 0,
        
        # RNN training parameters
        epochs: int = 128,
        bagging: bool = False,
        sequence_length: Optional[int] = -1,  # -1 for keeping the sequence length in the data to its original length, otherwise strided windows of length sequence_length,
        n_steps_per_call: Optional[int] = -1,  # number of timesteps in one backward-call; -1 for full sequence
        batch_size: Optional[int] = -1,  # -1 for a batch-size equal to the number of participants in the data
        learning_rate: Optional[float] = 1e-2,
        convergence_threshold: Optional[float] = 0,
        device: Optional[torch.device] = torch.device('cpu'),
        scheduler: Optional[bool] = False, 
        train_test_ratio: Optional[Union[float, List[int]]] = 1.,
        l2_weight_decay: Optional[float] = 1e-4,
        dropout: Optional[float] = 0.5,
        
        # SPICE training parameters
        spice_optimizer_type: Optional[str] = 'SR3_weighted_l1',
        spice_optim_threshold: Optional[float] = 0.05,
        spice_optim_regularization: Optional[float] = 1e-2,
        spice_library_polynomial_degree: Optional[int] = 1,
        n_trials_off_policy: Optional[int] = 1000,
        n_sessions_off_policy: Optional[int] = 1,
        n_trials_same_action_off_policy: Optional[int] = 5,
        use_optuna: Optional[bool] = False,
        optuna_threshold: Optional[float] = 0.1,
        optuna_n_trials: Optional[int] = 50,
        
        verbose: Optional[bool] = False,

        save_path_rnn: Optional[str] = None,
        save_path_spice: Optional[str] = None
    ):
        """
        Args:
            rnn_class: RNN class. Can be one of the precoded models in rnn.py or a custom implementation.
            spice_config: SPICE config
            list_signals: List of signals to use for SPICE
            hidden_size: Hidden size of the RNN
            dropout: Dropout rate of the RNN
            n_actions: Number of actions
            n_participants: Number of participants
            n_experiments: Number of experiments
            epochs: Number of epochs to train the RNN
            bagging: Whether to use bagging for the RNN
            sequence_length: Sequence length for the RNN
            n_steps_per_call: Number of steps per call for the RNN
            batch_size: Batch size for the RNN
            learning_rate: Learning rate for the RNN
            convergence_threshold: Convergence threshold for the RNN
            device: Device to use for the RNN (default: 'cpu')
            scheduler: Whether to use a scheduler for the RNN (default: False)
            train_test_ratio: Ratio of training to test data (default: 1.)
            l2_weight_decay: L2 weight decay for the RNN
            verbose: Whether to print verbose output (default: False)
            save_path_rnn: File path (.pkl) to save RNN model after training (default: None)
            save_path_spice: File path (.pkl) to save SPICE model after training (default: None)
        """
        
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
        self.deterministic = False
        self.l2_weight_decay = l2_weight_decay

        # Save parameters
        self.save_path_rnn = save_path_rnn
        self.save_path_spice = save_path_spice

        # SPICE training parameters
        self.spice_optimizer_type = spice_optimizer_type
        self.spice_optim_threshold = spice_optim_threshold
        self.spice_library_polynomial_degree = spice_library_polynomial_degree
        self.spice_optim_regularization = spice_optim_regularization
        self.n_trials_off_policy = n_trials_off_policy
        self.n_sessions_off_policy = n_sessions_off_policy
        self.n_trials_same_action_off_policy = n_trials_same_action_off_policy
        self.use_optuna = use_optuna
        self.optuna_threshold = optuna_threshold
        self.optuna_n_trials = optuna_n_trials

        # Data parameters
        self.n_actions = n_actions
        self.n_participants = n_participants
        self.n_experiments = n_experiments
        
        # RNN parameters
        self.dropout = dropout
        
        self.rnn_agent = None
        self.spice_agent = None
        self.spice_features = None
        
        self.rnn_model = rnn_class(
            n_actions=n_actions,
            n_participants=n_participants,
            n_experiments=n_experiments,
        ).to(device)

        self.spice_library_config = spice_config.library_setup
        self.spice_filter_config = spice_config.filter_setup
        self.control_parameters = spice_config.control_parameters
        self.rnn_modules = spice_config.rnn_modules

        self.spice_feature_list = spice_config.spice_feature_list

        self.optimizer_rnn = torch.optim.Adam(self.rnn_model.parameters(), lr=learning_rate)
    
    def fit(self, data: np.ndarray, targets: np.ndarray, data_test: np.ndarray = None, target_test: np.ndarray = None):
        """
        Fit the RNN and SPICE models to given data.
        
        Args:
            conditions: Array of shape (n_participants, n_trials, n_features)
            targets: Array of shape (n_participants, n_trials, n_actions)
        """
        
        dataset = DatasetRNN(data, targets)
        dataset_test = DatasetRNN(data_test, target_test) if data_test is not None and target_test is not None else None
        
        start_time = time.time()
        
        # ------------------------------------------------------------------------
        # Fit RNN
        # ------------------------------------------------------------------------
        
        # if self.verbose:
        print('\nTraining the RNN...')
        
        batch_size = data.shape[0] if self.batch_size == -1 else self.batch_size
        
        rnn_model, rnn_optimizer, _ = fit_model(
            model=self.rnn_model,
            dataset_train=dataset,
            dataset_test=dataset_test,
            optimizer=self.optimizer_rnn,
            convergence_threshold=self.convergence_threshold,
            epochs=self.epochs,
            batch_size=batch_size,
            bagging=self.bagging,
            scheduler=self.scheduler,
            n_steps=self.n_steps_per_call,
            verbose=self.verbose,
            path_save_checkpoints=None,
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
            
        # ------------------------------------------------------------------------
        # Fit SPICE
        # ------------------------------------------------------------------------
        
        self.spice_agent = {}
        self.spice_features = {}
        spice_modules = {rnn_module: {} for rnn_module in self.rnn_modules}

        self.spice_agent = fit_spice(
            rnn_modules=self.rnn_modules,
            control_signals=self.control_parameters,
            agent_rnn=self.rnn_agent,
            data=dataset,
            polynomial_degree=self.spice_library_polynomial_degree,
            library_setup=self.spice_library_config,
            filter_setup=self.spice_filter_config,
            optimizer_type=self.spice_optimizer_type,
            optimizer_threshold=self.spice_optim_threshold,
            optimizer_alpha=self.spice_optim_regularization,
            shuffle = False,
            n_trials_off_policy = self.n_trials_off_policy,
            n_sessions_off_policy = self.n_sessions_off_policy,
            n_trials_same_action_off_policy = self.n_trials_same_action_off_policy,
            train_test_ratio = self.train_test_ratio,
            deterministic = self.deterministic,
            verbose = self.verbose,
            use_optuna = self.use_optuna,
            optuna_threshold = self.optuna_threshold,
            optuna_n_trials = self.optuna_n_trials,
        )[0]

        # self.spice_features = self.spice_agent.get_spice_features()
        
        if self.verbose:
            print('SPICE training finished.')
            print(f'Training took {time.time() - start_time:.2f} seconds.')

        if self.save_path_spice is not None:
            print(f'Saving SPICE model to {self.save_path_spice}...')
            self.save_spice(None, self.save_path_spice)

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
        
        if isinstance(conditions, np.ndarray):
            conditions = torch.tensor(conditions, dtype=torch.float32, device=self.device)
        elif isinstance(conditions, torch.Tensor):
            pass
        else:
            raise TypeError(f"conditions must be either of type numpy.ndarray or torch.Tensor.")
        
        # get predictions about action probability by RNN and SPICE separately
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

    def print_spice_model(self, participant_id: int = 0) -> None:
        """
        Get the learned SPICE features and equations.
        
        Returns:
            Dictionary containing features and equations for each agent/model
        """
        print("SPICE modules:")
        self.spice_agent.print_model(participant_id)
        
        if hasattr(self.rnn_model, 'betas') and len(self.rnn_model.betas) > 0:
            for key in self.rnn_model.betas:
                if not isinstance(self.rnn_model.betas[key], ParameterModule):
                    participant_embedding = self.rnn_model.participant_embedding(torch.tensor(participant_id, device=self.device).int())
                else:
                    participant_embedding = None
                print(f"beta({key}) = {self.rnn_model.betas[key](participant_embedding).item():.4f}")
    
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