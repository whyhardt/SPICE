"""
SPICE training pipeline as a scikit-learn estimator
"""
import warnings
import time
import torch
import numpy as np
from sklearn.base import BaseEstimator
from typing import Dict, Optional, Tuple, List, Union
from copy import copy

from .spice_training import fit_spice, cross_entropy_loss
from .model import BaseModel
from .spice_utils import SpiceConfig, SpiceDataset


warnings.filterwarnings("ignore")

class SpiceEstimator(BaseEstimator):
    """
    Scikit-learn estimator for fitting the SPICE model and making predictions.
    
    Combines an RNN for predicting behavioral choices with SPICE for discovering
    the underlying dynamical equations of cognitive mechanisms.
    """
    
    def __init__(
        self,

        # RNN class and SPICE configuration
        spice_class: BaseModel,
        spice_config: SpiceConfig,

        # Data/Environment parameters
        n_actions: int = 2,
        n_participants: int = 1,
        n_experiments: int = 1,
        n_items: int = None,
        n_reward_features: int = None,

        # RNN training parameters
        epochs: Optional[int] = 1,
        warmup_steps: Optional[int] = 0,
        bagging: Optional[bool] = False,
        n_steps_per_call: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = 1e-2,
        convergence_threshold: Optional[float] = 0,
        device: Optional[torch.device] = torch.device('cpu'),
        scheduler: Optional[bool] = False,
        ensemble_size: Optional[int] = 1,
        embedding_size: Optional[int] = 32,
        l2_rnn: Optional[float] = 0,
        dropout: Optional[float] = 0.,
        loss_fn: Optional[callable] = cross_entropy_loss,

        # Polynomial pruning parameters
        sindy_library_polynomial_degree: Optional[int] = 1,
        sindy_pruning_frequency: Optional[int] = 1,
        sindy_threshold_pruning: Optional[float] = None,
        sindy_ensemble_pruning: Optional[float] = None,
        sindy_population_pruning: Optional[float] = None,

        verbose: Optional[bool] = False,
        keep_log: Optional[bool] = False,
        save_path_spice: Optional[str] = None,
        compiled_forward: Optional[bool] = True,

        kwargs_spice_class: Optional[dict] = {},
    ):
        """
        Args:
            spice_class: RNN class. Can be one of the precoded models or a custom BaseModel subclass.
            spice_config: SpiceConfig defining submodules, memory states, and logit mapping.
            n_actions: Number of observable actions.
            n_items: Number of internal item representations (defaults to n_actions if None).
            n_participants: Number of participants in the dataset.
            n_experiments: Number of experiments.
            n_reward_features: Number of reward feature columns in the dataset.
            epochs: Number of training epochs.
            warmup_steps: Epochs before pruning begins (no pruning during warmup).
            bagging: Whether to use bagging.
            n_steps_per_call: BPTT truncation length (None = full sequence).
            batch_size: Training batch size (None = auto-detect max via GPU probing).
            learning_rate: Learning rate for RNN parameters.
            convergence_threshold: Early stopping threshold (0 = disabled).
            device: Compute device (default: 'cpu').
            scheduler: Enable ReduceOnPlateauWithRestarts LR scheduler.
            ensemble_size: Number of independent RNN ensemble members.
            l2_rnn: L2 weight decay (applied via AdamW — implicitly penalizes higher-degree polynomial terms more).
            dropout: Dropout rate in RNN modules.
            loss_fn: Behavioral loss function (prediction, target) -> scalar.
            sindy_library_polynomial_degree: Max polynomial degree.
            sindy_pruning_frequency: Epochs between pruning events.
            sindy_threshold_pruning: Minimum effect size delta for CI test (None = disabled).
            sindy_ensemble_pruning: Confidence level for ensemble CI test (primary pruning mechanism).
            sindy_population_pruning: Cross-participant presence threshold 0-1 (None = disabled).
            verbose: Print training progress.
            keep_log: Keep full training log (vs. live terminal update).
            save_path_spice: File path (.pkl) to auto-save SPICE model after training.
            compiled_forward: Use @torch.compile for forward loops.
            kwargs_spice_class: Extra keyword arguments forwarded to spice_class.__init__().
        """

        super(BaseEstimator, self).__init__()

        # Training parameters
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.n_steps_per_call = n_steps_per_call
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.scheduler = scheduler
        self.device = device
        self.verbose = verbose
        self.keep_log = keep_log
        self.deterministic = False
        self.ensemble_size = ensemble_size
        self.embedding_size = embedding_size
        self.l2_rnn = l2_rnn
        self.loss_fn = loss_fn
        self.compiled_forward = compiled_forward

        # Save parameters
        self.save_path_model = save_path_spice

        # Pruning parameters
        self.sindy_library_polynomial_degree = sindy_library_polynomial_degree
        self.sindy_pruning_frequency = sindy_pruning_frequency
        self.sindy_threshold_pruning = sindy_threshold_pruning
        self.sindy_population_pruning = sindy_population_pruning
        self.sindy_ensemble_pruning = sindy_ensemble_pruning

        # Data parameters
        self.n_actions = n_actions
        self.n_items = n_items
        self.n_reward_features = n_reward_features
        self.n_participants = n_participants
        self.n_experiments = n_experiments

        # RNN parameters
        self.dropout = dropout

        # SPICE attributes
        self.spice_config = spice_config
        self.spice_class = spice_class
        self.spice_features = None
        self.model = spice_class(
            n_actions=n_actions,
            n_participants=n_participants,
            n_experiments=n_experiments,
            dropout=dropout,
            spice_config=spice_config,
            sindy_polynomial_degree=sindy_library_polynomial_degree,
            ensemble_size=ensemble_size,
            embedding_size=embedding_size,
            n_items=n_items,
            n_reward_features=n_reward_features,
            device=device,
            compiled_forward=compiled_forward,
            **kwargs_spice_class,
        ).to(device)

        # Single optimizer: L2 weight decay via AdamW handles regularization
        self.rnn_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=l2_rnn,
        )
        
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
        
        rnn_model, rnn_optimizer = fit_spice(
            model=self.model,
            optimizer=self.rnn_optimizer,
            dataset_train=dataset,
            dataset_test=dataset_test,

            epochs=self.epochs,
            n_warmup_steps=self.warmup_steps,
            batch_size=self.batch_size,
            scheduler=self.scheduler,
            n_steps=self.n_steps_per_call,
            convergence_threshold=self.convergence_threshold,
            loss_fn=self.loss_fn,

            sindy_pruning_frequency=self.sindy_pruning_frequency,
            sindy_threshold_pruning=self.sindy_threshold_pruning,
            sindy_ensemble_pruning=self.sindy_ensemble_pruning,
            sindy_population_pruning=self.sindy_population_pruning,

            verbose=self.verbose,
            keep_log=self.keep_log,
            path_save_checkpoints=None,
        )

        self.model = rnn_model
        self.rnn_optimizer = rnn_optimizer
        
        if self.verbose:
            print('\nRNN training finished.')
            print(f'Training took {time.time() - start_time:.2f} seconds.')

        if self.save_path_model is not None:
            print(f'Saving SPICE model to {self.save_path_model}...')
            self.save_spice(self.save_path_model)
   
    def predict(self, conditions: np.ndarray) -> np.ndarray:
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
        
        # average across ensemble dim (E, B, T, W, A) -> (B, T, W, A)
        prediction = self.model(conditions)[0].mean(dim=0).detach().cpu().numpy()
        return prediction
    
    def print_spice_model(self, participant_id: int = 0, experiment_id: int = 0) -> None:
        """
        Get the learned SPICE features and equations.
        """
        
        self.model.print(participant_id=participant_id, experiment_id=experiment_id)

    def get_participant_embeddings(self) -> Dict:
        if hasattr(self.model, 'participant_embedding'):
            participant_ids = torch.arange(self.n_participants, device=self.device, dtype=torch.int32).view(-1, 1)
            embeddings = self.model.participant_embedding(participant_ids)
            return {participant_id.item(): embeddings[participant_id, 0] for participant_id in participant_ids}
        else:
            print(f'RNN model has no participant_embedding module.')
            return None

    def get_sindy_coefficients(self, key_module: Optional[str] = None, aggregate: bool = False) -> Dict[str, np.ndarray]:
        """Returns a dict of modules holding a numpy array with the sindy coefficients of shape (ensemble, participant, experiment, coefficient)."""
        
        return self.model.get_sindy_coefficients(key_module=key_module, aggregate=aggregate)
    
    def count_sindy_coefficients(self):
        return self.model.count_sindy_coefficients()
       
    def get_modules(self):
        return self.model.get_modules()
    
    def get_candidate_terms(self, key_module: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        return self.model.get_candidate_terms(key_module=key_module)
        
    def load_spice(self, path_model: str, deterministic: bool = True):
        """Load a saved SPICE model (RNN weights + polynomial presence masks)."""

        loaded_parameters = torch.load(path_model, map_location=torch.device('cpu'))

        self.model = self.spice_class(
            spice_config=self.spice_config,
            n_actions=self.model.n_actions,
            n_items=self.model.n_items,
            n_reward_features=self.model.n_reward_features,
            n_participants=self.model.n_participants,
            n_experiments=self.model.n_experiments,
            sindy_polynomial_degree=self.model.sindy_polynomial_degree,
            ensemble_size=self.model.ensemble_size,
            device=self.model.device,
        )

        self.model.load_state_dict(loaded_parameters['model'])
        self.model.sindy_coefficients_presence = loaded_parameters['sindy_coefficients_presence']
        self.model.init_state(batch_size=self.model.n_participants)

        self.model = self.model.to(self.model.device)
        self.model.eval()

    def save_spice(self, path_rnn: str):
        """Save the SPICE model (RNN weights, optimizer state, and polynomial presence masks)."""

        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.rnn_optimizer.state_dict(),
            'sindy_coefficients_presence': self.model.sindy_coefficients_presence,
        }
        torch.save(state_dict, path_rnn)
        
    def set_device(self, device: torch.device):
        self.model.to(device)
        self.device = device
    
    def aggregate(self, mode: bool = True):
        self.model.aggregate = mode

    def eval(self, aggregate: bool = True):
        self.model.eval(aggregate=aggregate)

    def train(self, mode: bool = True):
        self.model.train(mode=mode)
    
    def __call__(self, conditions: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        return self.model(conditions, state)