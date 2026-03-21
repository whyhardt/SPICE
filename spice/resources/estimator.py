"""
SPICE training pipeline as a scikit-learn estimator
"""
import warnings
import time
import torch
import numpy as np
from sklearn.base import BaseEstimator
from typing import Dict, Optional, Tuple, List, Union

from .spice_training import fit_spice, cross_entropy_loss
from .rnn import BaseRNN
from .spice_utils import SpiceConfig, SpiceDataset
from ..utils.agent import Agent


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
        spice_class: BaseRNN,
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
        n_steps_per_call: Optional[int] = None,  # number of timesteps in one backward-call; -1 for full sequence
        batch_size: Optional[int] = None,  # -1 for a batch-size equal to the number of participants in the data
        learning_rate: Optional[float] = 1e-2,
        convergence_threshold: Optional[float] = 0,
        device: Optional[torch.device] = torch.device('cpu'),
        scheduler: Optional[bool] = False,
        ensemble_size: Optional[int] = 1,
        l2_rnn: Optional[float] = 0,
        dropout: Optional[float] = 0.,
        loss_fn: Optional[callable] = cross_entropy_loss,

        # SPICE training parameters
        use_sindy: Optional[bool] = False,
        sindy_weight: Optional[float] = 0.1,  # Weight for SINDy regularization loss
        sindy_alpha: Optional[float] = 1e-4,  # Degree-weighted coefficient penalty strength (ridge alpha)
        sindy_library_polynomial_degree: Optional[int] = 1,
        sindy_pruning_frequency: Optional[int] = 1,  # Epochs between pruning events
        sindy_threshold_pruning: Optional[float] = None,  # Optional per-member threshold pruning (None to disable)
        sindy_ensemble_pruning: Optional[float] = None,  # Ensemble t-test significance level (primary pruning mechanism)
        sindy_population_pruning: Optional[float] = None,  # Optional cross-participant filter (0-1)
        sindy_reconditioning_epochs: Optional[int] = 3,  # Pure SINDy SGD epochs after ridge recalibration
        sindy_refit: Optional[bool] = True,  # Enable Stage 2 Training (SINDy refit on frozen RNN parameters) 

        verbose: Optional[bool] = False,
        keep_log: Optional[bool] = False,
        save_path_spice: Optional[str] = None,
        compiled_forward: Optional[bool] = True,
        
        kwargs_rnn_class: Optional[dict] = {},
    ):
        """
        Args:
            spice_class: RNN class. Can be one of the precoded models or a custom BaseRNN subclass.
            spice_config: SpiceConfig defining submodules, memory states, and logit mapping.
            n_actions: Number of observable actions.
            n_items: Number of internal item representations (defaults to n_actions if None).
            n_participants: Number of participants in the dataset.
            n_experiments: Number of experiments.
            n_reward_features: Number of reward feature columns in the dataset.
            epochs: Number of training epochs.
            warmup_steps: Epochs of exponential SINDy weight warmup (no pruning during warmup).
            bagging: Whether to use bagging.
            n_steps_per_call: BPTT truncation length (None = full sequence).
            batch_size: Training batch size (None = auto-detect max via GPU probing).
            learning_rate: Learning rate for RNN parameters.
            convergence_threshold: Early stopping threshold (0 = disabled).
            device: Compute device (default: 'cpu').
            scheduler: Enable ReduceOnPlateauWithRestarts LR scheduler.
            ensemble_size: Number of independent RNN ensemble members.
            l2_rnn: L2 weight decay for RNN parameters.
            dropout: Dropout rate in GRU modules.
            loss_fn: Behavioral loss function (prediction, target) -> scalar.
            use_sindy: Enable SINDy integration.
            sindy_weight: Lambda for SINDy regularization loss.
            sindy_alpha: Degree-weighted L1 penalty strength.
            sindy_library_polynomial_degree: Max polynomial degree for SINDy candidate library.
            sindy_pruning_frequency: Epochs between pruning events.
            sindy_threshold_pruning: Minimum effect size delta for CI test (None = disabled).
            sindy_ensemble_pruning: Confidence level for ensemble CI test (primary pruning mechanism).
            sindy_population_pruning: Cross-participant presence threshold 0-1 (None = disabled).
            sindy_reconditioning_epochs: Pure SINDy SGD epochs after ridge recalibration to warm-start the optimizer (0 = disable).
            verbose: Print training progress.
            keep_log: Keep full training log (vs. live terminal update).
            save_path_spice: File path (.pkl) to auto-save SPICE model after training.
            compiled_forward: Use @torch.compile for forward loops.
            kwargs_rnn_class: Extra keyword arguments forwarded to spice_class.__init__().
        """
        
        super(BaseEstimator, self).__init__()
        
        self.use_sindy = use_sindy
        
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
        self.l2_rnn = l2_rnn
        self.loss_fn = loss_fn
        self.compiled_forward = compiled_forward

        # Save parameters
        self.save_path_model = save_path_spice

        # SINDy training parameters
        self.sindy_weight = sindy_weight
        self.sindy_alpha = sindy_alpha
        self.sindy_library_polynomial_degree = sindy_library_polynomial_degree
        self.sindy_pruning_frequency = sindy_pruning_frequency
        self.sindy_threshold_pruning = sindy_threshold_pruning
        self.sindy_population_pruning = sindy_population_pruning
        self.sindy_ensemble_pruning = sindy_ensemble_pruning
        self.sindy_reconditioning_epochs = sindy_reconditioning_epochs
        self.sindy_refit = sindy_refit
        
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
        self.rnn_class = spice_class
        self.spice_features = None
        self.model = spice_class(
            n_actions=n_actions,
            n_participants=n_participants,
            n_experiments=n_experiments,
            dropout=dropout,
            spice_config=spice_config,
            sindy_polynomial_degree=sindy_library_polynomial_degree,
            sindy_alpha=sindy_alpha,
            ensemble_size=ensemble_size,
            use_sindy=use_sindy,
            n_items=n_items,
            n_reward_features=n_reward_features,
            device=device,
            compiled_forward=compiled_forward,
            fit_sindy=sindy_weight > 0,
            **kwargs_rnn_class,
        ).to(device)

        sindy_params = []
        rnn_params = []
        for name, param in self.model.named_parameters():
            if 'sindy' in name:
                sindy_params.append(param)
            else:
                rnn_params.append(param)
        # Separate optimizer param groups: SINDy coefficients get fixed lr, RNN params get configurable lr + weight decay
        self.rnn_optimizer = torch.optim.AdamW(
            [
            {'params': sindy_params, 'weight_decay': 0, 'lr': 0.01},
            {'params': rnn_params, 'weight_decay': l2_rnn, 'lr': learning_rate},
            ],
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

            sindy_weight=self.sindy_weight,
            sindy_alpha=self.sindy_alpha,
            sindy_pruning_frequency=self.sindy_pruning_frequency,
            sindy_threshold_pruning=self.sindy_threshold_pruning,
            sindy_ensemble_pruning=self.sindy_ensemble_pruning,
            sindy_population_pruning=self.sindy_population_pruning,
            sindy_reconditioning_epochs=self.sindy_reconditioning_epochs,
            sindy_refit=self.sindy_refit,
            
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
        
        # rnn predictions — average across ensemble dim (E, B, T, W, A) -> (B, T, W, A)
        self.model.use_sindy = False
        self.model.aggregate = True
        prediction_rnn = self.model(conditions, batch_first=True)[0].mean(dim=0)
        
        # SPICE predictions — average across ensemble dim
        self.model.use_sindy = True
        prediction_spice = self.model(conditions, batch_first=True)[0].mean(dim=0)
        
        return prediction_rnn, prediction_spice

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
        
        # LOAD RNN MODEL AND OPTIMIZER
                
        # load trained parameters
        loaded_parameters = torch.load(path_model, map_location=torch.device('cpu'))
        
        # Infer ensemble_size from saved coefficient shape: (E, P, X, terms)
        self.model.ensemble_size = loaded_parameters['model']['sindy_coefficients.'+next(iter(self.model.submodules_rnn))].shape[0]
        
        self.model = self.rnn_class(
            spice_config=self.spice_config,
            n_actions=self.model.n_actions,
            n_items=self.model.n_items,
            n_reward_features=self.model.n_reward_features,
            n_participants=self.model.n_participants,
            n_experiments=self.model.n_experiments,
            sindy_polynomial_degree=self.model.sindy_polynomial_degree,
            ensemble_size=self.model.ensemble_size,
            use_sindy=True,
            device=self.model.device,
            )
        
        for module in self.get_modules():
            self.model.setup_sindy_coefficients(key_module=module, polynomial_degree=self.model.sindy_specs[module]['polynomial_degree'])
        self.model.sindy_coefficients_presence = loaded_parameters['sindy_coefficients_presence']

        self.model.load_state_dict(loaded_parameters['model'])
        self.model.init_state(batch_size=self.model.n_participants)

        self.model = self.model.to(self.model.device)
            
    def save_spice(self, path_rnn: str):
        """
        Save the SPICE model (RNN weights, optimizer state, and SINDy coefficient masks) to a .pkl file.

        Args:
            path_rnn: File path to save the model.
        """
        
        # Save RNN model
        state_dict = {
            'model': self.model.state_dict(), 
            'optimizer': self.rnn_optimizer.state_dict(), 
            'sindy_coefficients_presence': self.model.sindy_coefficients_presence,
            }
        torch.save(state_dict, path_rnn)
        
    def set_device(self, device: torch.device):
        self.model.to(device)