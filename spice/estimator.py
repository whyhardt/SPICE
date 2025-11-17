"""
SPICE training pipeline as a scikit-learn estimator
"""
import warnings
import time
import torch
import numpy as np
import pysindy as ps
from sklearn.base import BaseEstimator
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union, List

from .resources.spice_training import fit_model
from .resources.bandits import AgentNetwork, Bandits
from .resources.rnn import BaseRNN
from .resources.spice_utils import SpiceConfig, SpiceDataset


warnings.filterwarnings("ignore")

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
        n_participants: int = 1,
        n_experiments: int = 1,
        n_items: int = None,
        
        # RNN training parameters
        epochs: int = 1,
        bagging: bool = False,
        sequence_length: Optional[int] = -1,  # -1 for keeping the sequence length in the data to its original length, otherwise strided windows of length sequence_length,
        n_steps_per_call: Optional[int] = -1,  # number of timesteps in one backward-call; -1 for full sequence
        batch_size: Optional[int] = -1,  # -1 for a batch-size equal to the number of participants in the data
        learning_rate: Optional[float] = 1e-2,
        convergence_threshold: Optional[float] = 0,
        device: Optional[torch.device] = torch.device('cpu'),
        scheduler: Optional[bool] = False, 
        train_test_ratio: Optional[Union[float, List[int]]] = 1.,
        l2_weight_decay: Optional[float] = 0,
        dropout: Optional[float] = 0.,
        
        # SPICE training parameters
        sindy_weight: Optional[float] = 0.1,  # Weight for SINDy regularization loss
        sindy_epochs: Optional[int] = 1000,
        sindy_threshold_frequency = 50,
        sindy_threshold: Optional[float] = 0.05,
        sindy_regularization: Optional[float] = 1e-2,
        sindy_library_polynomial_degree: Optional[int] = 1,
        use_sindy: bool = False,

        verbose: Optional[bool] = False,
        save_path_spice: Optional[str] = None
    ):
        """
        Args:
            rnn_class: RNN class. Can be one of the precoded models in rnn.py or a custom implementation.
            spice_config: SPICE config
            dropout: Dropout rate of the RNN
            n_actions: Number of actions
            n_items: Number of total action items (including the ones not selectable at the current trial)
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
            save_path_spice: File path (.pkl) to save SPICE model after training (default: None)
        """
        
        super(BaseEstimator, self).__init__()
        
        self.use_sindy = use_sindy
        
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
        self.save_path_model = save_path_spice

        # SINDy training parameters
        self.sindy_weight = sindy_weight
        self.sindy_threshold_frequency = sindy_threshold_frequency
        self.sindy_threshold = sindy_threshold
        self.sindy_library_polynomial_degree = sindy_library_polynomial_degree
        self.sindy_regularization = sindy_regularization
        self.sindy_epochs = sindy_epochs
        
        # Data parameters
        self.n_actions = n_actions
        self.n_items = n_items
        self.n_participants = n_participants
        self.n_experiments = n_experiments
        
        # RNN parameters
        self.dropout = dropout
        
        self.spice_config = spice_config
        self.rnn_class = rnn_class
        self.rnn_agent = None
        self.spice_agent = None
        self.spice_features = None
        
        self.rnn_model = rnn_class(
            n_actions=n_actions,
            n_participants=n_participants,
            n_experiments=n_experiments,
            dropout=dropout,
            enable_sindy_reg=(sindy_weight > 0),
            spice_config=spice_config,
            sindy_polynomial_degree=sindy_library_polynomial_degree,
            use_sindy=use_sindy,
            n_items=n_items,
        ).to(device)
        
        self.rnn_optimizer = torch.optim.AdamW(self.rnn_model.parameters(), lr=learning_rate, weight_decay=l2_weight_decay)
            
    def fit(self, data: np.ndarray, targets: np.ndarray, data_test: np.ndarray = None, target_test: np.ndarray = None):
        """
        Fit the RNN and SPICE models to given data.
        
        Args:
            conditions: Array of shape (n_participants, n_trials, n_features)
            targets: Array of shape (n_participants, n_trials, n_actions)
        """
        
        dataset = SpiceDataset(data, targets)
        dataset_test = SpiceDataset(data_test, target_test) if data_test is not None and target_test is not None else None
        
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
            optimizer=self.rnn_optimizer,
            convergence_threshold=self.convergence_threshold,
            sindy_weight=self.sindy_weight,
            sindy_threshold=self.sindy_threshold,
            sindy_threshold_frequency=self.sindy_threshold_frequency,
            sindy_epochs=self.sindy_epochs,
            epochs=self.epochs,
            batch_size=batch_size,
            bagging=self.bagging,
            scheduler=self.scheduler,
            n_steps=self.n_steps_per_call,
            verbose=self.verbose,
            path_save_checkpoints=None,
        )

        self.rnn_model = rnn_model
        self.rnn_optimizer = rnn_optimizer

        # Get trained state dict
        state_dict = rnn_model.state_dict()

        # Create RNN agent and load trained weights
        # IMPORTANT: Match ensemble_size from trained model (may be 1 after stage 2)
        rnn_agent_model = self.rnn_class(
                n_actions=rnn_model.n_actions,
                n_participants=rnn_model.n_participants,
                n_experiments=len(dataset.xs[..., -2].unique()),
                dropout=0,
                spice_config=rnn_model.spice_config,
                sindy_polynomial_degree=rnn_model.sindy_polynomial_degree,
                n_items=rnn_model.n_items,
            )
        rnn_agent_model.sindy_ensemble_size = rnn_model.sindy_ensemble_size  # Match trained ensemble size
        rnn_agent_model.setup_sindy_coefficients(rnn_model.sindy_polynomial_degree)
        rnn_agent_model.load_state_dict(state_dict)
        self.rnn_agent = AgentNetwork(rnn_agent_model, self.n_actions, device=self.device, use_sindy=False)

        # Create SPICE agent and load trained weights
        spice_agent_model = self.rnn_class(
                n_actions=rnn_model.n_actions,
                n_participants=rnn_model.n_participants,
                n_experiments=rnn_model.n_experiments,
                dropout=0,
                spice_config=rnn_model.spice_config,
                sindy_polynomial_degree=rnn_model.sindy_polynomial_degree,
                n_items=rnn_model.n_items,
            )
        spice_agent_model.sindy_ensemble_size = rnn_model.sindy_ensemble_size  # Match trained ensemble size
        spice_agent_model.setup_sindy_coefficients(rnn_model.sindy_polynomial_degree)
        spice_agent_model.load_state_dict(state_dict)
        self.spice_agent = AgentNetwork(spice_agent_model, self.n_actions, device=self.device, use_sindy=True)
        
        if self.verbose:
            print('\nRNN training finished.')
            print(f'Training took {time.time() - start_time:.2f} seconds.')

        if self.save_path_model is not None:
            print(f'Saving SPICE model to {self.save_path_model}...')
            self.save_spice(self.save_path_model)
   
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
        prediction = self.rnn_agent.model(conditions, batch_first=True)[0].reshape(-1, self.n_actions)
        prediction = torch.nn.functional.softmax(prediction, dim=-1).detach().cpu().numpy()
        prediction_rnn[mask.detach().cpu().numpy()] = prediction[mask]
        prediction_rnn = prediction_rnn.reshape(*conditions.shape[:-1], self.n_actions)
        
        # SPICE predictions
        prediction = self.spice_agent.model(conditions, batch_first=True)[0].reshape(-1, self.n_actions)
        prediction = torch.nn.functional.softmax(prediction, dim=-1).detach().cpu().numpy()
        prediction_spice[mask.detach().cpu().numpy()] = prediction[mask]
        prediction_spice = prediction_spice.reshape(*conditions.shape[:-1], self.n_actions)
        
        return prediction_rnn, prediction_spice

    def print_spice_model(self, participant_id: int = 0) -> None:
        """
        Get the learned SPICE features and equations.
        """
        
        self.rnn_model.print_spice_model(participant_id)
    
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
        
    def load_spice(self, path_model: str, deterministic: bool = True):
        
        # LOAD RNN MODEL AND OPTIMIZER
                
        # load trained parameters
        state_dict = torch.load(path_model, map_location=torch.device('cpu'))
        
        self.rnn_model.sindy_ensemble_size = state_dict['model']['sindy_coefficients.'+next(iter(self.rnn_model.submodules_rnn))].shape[1]
        self.rnn_model.setup_sindy_coefficients()
        
        self.rnn_model.load_state_dict(state_dict['model'])
        self.rnn_model.set_initial_state(batch_size=self.rnn_model.n_participants)
        # optimizer.load_state_dict(state_dict['optimizer'])

        self.rnn_model = self.rnn_model.to(self.rnn_model.device)
        self.rnn_optimizer = self.rnn_optimizer
        
        # SETUP RNN AND SYMBOLIC AGENT
        state_dict = self.rnn_model.state_dict() 
        for agent_type in [('rnn_agent', False), ('spice_agent', True)]:
            model = self.rnn_class(
                    spice_config=self.spice_config,
                    n_actions=self.rnn_model.n_actions,
                    n_participants=self.rnn_model.n_participants,
                    n_experiments=self.rnn_model.n_experiments,
                    sindy_config=self.rnn_model.spice_config,
                    sindy_polynomial_degree=self.rnn_model.sindy_polynomial_degree,
                    sindy_ensemble_size=self.rnn_model.sindy_ensemble_size,
                )
            model.load_state_dict(state_dict)
            agent = AgentNetwork(model, self.n_actions, device=self.device, use_sindy=agent_type[1], deterministic=deterministic)
            setattr(self, agent_type[0], agent)
        
    def save_spice(self, path_rnn: str):
        """
        Save the RNN and SPICE models to the given paths.
        If path_rnn is None, only the SPICE model will be saved (requires a fitted SPICE model including the RNN).
        If path_spice is None, only the RNN model will be saved (requires a fitted RNN model).
        
        Args:
            path_rnn: Path to the RNN model
            path_spice: Path to the SPICE model
        """
        
        # Save RNN model
        state_dict = {'model': self.rnn_model.state_dict(), 'optimizer': self.rnn_optimizer.state_dict()}
        torch.save(state_dict, path_rnn)