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
from .resources.bandits import AgentNetwork, AgentSpice, Bandits
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
        use_sindy: bool = False,
        
        # Data/Environment parameters
        n_actions: int = 2,
        n_participants: int = 0,
        n_experiments: int = 0,
        
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
        l1_weight_decay: Optional[float] = 0,
        dropout: Optional[float] = 0.,
        
        # SPICE training parameters
        sindy_threshold_frequency = 50,
        spice_optimizer_type: Optional[str] = 'SR3_weighted_l1',
        spice_optim_threshold: Optional[float] = 0.05,
        spice_optim_regularization: Optional[float] = 1e-2,
        spice_library_polynomial_degree: Optional[int] = 1,
        simulation_environment: Bandits = None,
        n_trials_off_policy: Optional[int] = 1000,
        n_sessions_off_policy: Optional[int] = 1,
        n_trials_same_action_off_policy: Optional[int] = 5,
        use_optuna: Optional[bool] = False,
        optuna_threshold: Optional[float] = 0.1,
        optuna_n_trials: Optional[int] = 50,

        # Differentiable SINDy parameters (end-to-end training)
        sindy_weight: Optional[float] = 0.1,  # Weight for SINDy regularization loss
        
        verbose: Optional[bool] = False,
        
        save_path_spice: Optional[str] = None
    ):
        """
        Args:
            rnn_class: RNN class. Can be one of the precoded models in rnn.py or a custom implementation.
            spice_config: SPICE config
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
        self.l1_weight_decay = l1_weight_decay

        # Save parameters
        self.save_path_model = save_path_spice

        # SPICE training parameters
        self.spice_optimizer_type = spice_optimizer_type
        self.spice_optim_threshold = spice_optim_threshold
        self.spice_library_polynomial_degree = spice_library_polynomial_degree
        self.spice_optim_regularization = spice_optim_regularization
        self.simulation_environment = simulation_environment
        self.n_trials_off_policy = n_trials_off_policy
        self.n_sessions_off_policy = n_sessions_off_policy
        self.n_trials_same_action_off_policy = n_trials_same_action_off_policy
        self.use_optuna = use_optuna
        self.optuna_threshold = optuna_threshold
        self.optuna_n_trials = optuna_n_trials
        
        # SINDy regularization parameters (for differentiable end-to-end training)
        self.sindy_weight = sindy_weight
        self.sindy_threshold_value = spice_optim_threshold
        self.sindy_threshold_frequency = sindy_threshold_frequency

        # Data parameters
        self.n_actions = n_actions
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
            sindy_polynomial_degree=spice_library_polynomial_degree,
            use_sindy=use_sindy,
        ).to(device)
        
        # Separate parameters
        # individual_params = list(self.rnn_model.participant_embedding.parameters()) + list(self.rnn_model.betas['x_value_reward'].parameters()) + list(self.rnn_model.betas['x_value_choice'].parameters())
        # sindy_coefficients = list(self.rnn_model.sindy_coefficients.parameters())
        # rnn_params = list(self.rnn_model.submodules_rnn.parameters())#[p for p in self.rnn_model.parameters() if not any(p is ip for ip in individual_params)]
        
        # if l1_weight_decay != 0:
        # self.optimizer_rnn = torch.optim.AdamW([
        #     {'params': individual_params, 'weight_decay': 0.0},
        #     {'params': sindy_coefficients, 'weight_decay': l2_weight_decay},
        #     {'params': rnn_params, 'weight_decay': l2_weight_decay}
        # ], lr=learning_rate)
        # else:
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
            l1_weight_decay=self.l1_weight_decay,
            sindy_weight=self.sindy_weight,
            sindy_threshold_value=self.sindy_threshold_value,
            sindy_threshold_frequency=self.sindy_threshold_frequency,
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
                
    def _extract_sindy_from_rnn(self):
        """
        Extract learned SINDy coefficients from RNN and create AgentSpice for compatibility.
        This is used when end-to-end differentiable training is enabled (sindy_weight > 0).
        """
        if not hasattr(self.rnn_model, 'sindy_coefficients') or len(self.rnn_model.sindy_coefficients) == 0:
            print("Warning: No SINDy coefficients found in RNN model. Skipping SPICE extraction.")
            return None

        print("\nExtracting learned SPICE coefficients from RNN...")

        sindy_modules = {}

        for module_name in self.rnn_modules:
            if module_name not in self.rnn_model.sindy_coefficients:
                continue

            sindy_modules[module_name] = {}

            # Get coefficients and masks
            coeffs = self.rnn_model.sindy_coefficients[module_name].detach().cpu().numpy()
            masks = self.rnn_model.sindy_masks[module_name].cpu().numpy()
            feature_names = self.rnn_model.sindy_library_names[module_name]

            # Create pysindy model for each participant
            for pid in range(self.n_participants):
                coef_sparse = coeffs[pid] * masks[pid]

                # Create a minimal pysindy model for compatibility
                # We don't need the full optimizer, just the structure
                sindy_model = ps.SINDy(
                    optimizer=ps.STLSQ(threshold=0.0),  # Dummy optimizer
                    feature_library=ps.PolynomialLibrary(
                        degree=self.spice_library_polynomial_degree,
                        interaction_only=False
                    ),
                    discrete_time=True,
                    feature_names=feature_names,
                )

                # Manually set the learned coefficients
                sindy_model.model = sindy_model._make_model()
                sindy_model.model.steps[-1][1].coef_ = coef_sparse.reshape(1, -1)
                sindy_model.n_input_features_ = len(feature_names)
                sindy_model.n_output_features_ = 1
                
                sindy_modules[module_name][pid] = sindy_model

        # Create AgentSpice
        agent_spice = AgentSpice(
            model_rnn=deepcopy(self.rnn_model),
            sindy_modules=sindy_modules,
            n_actions=self.n_actions,
            deterministic=True
        )

        print(f"Extracted SPICE models for {len(sindy_modules)} modules across {self.n_participants} participants.")

        return agent_spice

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