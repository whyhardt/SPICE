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

from .training import fit_spice, cross_entropy_loss
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
        
        # RNN class and SPICE configuration. Can be one of the precoded models in rnn.py or a custom implementation.
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
        n_steps_per_call: Optional[int] = None,  # number of timesteps in one backward-call; -1 for full sequence
        batch_size: Optional[int] = None,  # -1 for a batch-size equal to the number of participants in the data
        learning_rate: Optional[float] = 1e-2,
        convergence_threshold: Optional[float] = 0,
        device: Optional[torch.device] = torch.device('cpu'),
        ensemble_size: Optional[int] = 10,
        l2_rnn: Optional[float] = 0,
        dropout: Optional[float] = 0.1,
        loss_fn: Optional[callable] = cross_entropy_loss,
        loss_fn_kwargs: Optional[dict] = {'label_smoothing': 0.01},
        embedding_size: Optional[int] = 8,

        # SPICE training parameters
        use_sindy: Optional[bool] = False,
        sindy_weight: Optional[float] = 0.01,  # Weight for SINDy regularization loss
        sindy_lambda_loading: Optional[float] = 1e-4,  # L1 strength for the proximal step on concept loadings, and the ridge alpha
        sindy_lambda_concept: Optional[float] = 1e-4,  # L1 strength for the proximal step on concept directions (support density)
        sindy_library_polynomial_degree: Optional[int] = 2,
        sindy_pruning_frequency: Optional[int] = 100,  # Epochs between pruning events
        sindy_threshold_pruning: Optional[float] = 0.01,  # Optional per-member threshold pruning (None to disable)
        sindy_ensemble_pruning: Optional[float] = 0.5,  # Minimum ensemble ratio for a term to survive (primary pruning mechanism)
        sindy_pruning_terms: Optional[int] = None, # Overrides both per-event pruning budgets (concept gates and concept support). Defaults to None: each is sized so it can reach 0 within 'epochs-epochs_warmup' epochs
        sindy_refit: Optional[bool] = True,  # Enable Stage 2 Training (SINDy refit on frozen RNN parameters)
        sindy_ridge: Optional[bool] = True,  # Use ridge regression initialization in Stage 2.2 (falls back to SGD on failure)
        sindy_shooting_steps: Optional[int] = 100,  # Multi-step shooting horizon for Stage 2 (1 = one-step-ahead)

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
            warmup_steps: Epochs of exponential SINDy weight warmup (no pruning during warmup).
            bagging: Whether to use bagging.
            n_steps_per_call: BPTT truncation length (None = full sequence).
            batch_size: Training batch size (None = auto-detect max via GPU probing).
            learning_rate: Learning rate for RNN parameters.
            convergence_threshold: Early stopping threshold (0 = disabled).
            device: Compute device (default: 'cpu').
            ensemble_size: Number of independent RNN ensemble members.
            l2_rnn: L2 weight decay for RNN parameters.
            dropout: Dropout rate in GRU modules.
            loss_fn: Behavioral loss function (prediction, target) -> scalar.
            use_sindy: Enable SINDy integration.
            sindy_weight: Lambda for SINDy regularization loss.
            sindy_lambda_loading: L1 strength for the proximal step on the concept loadings (Z).
            sindy_lambda_concept: L1 strength for the proximal step on the concept directions (V).
                Sets how dense a concept's support may be. With sindy_lambda_loading alone the
                objective is minimised by maximally dense concepts, since a k-dense
                unit-norm direction carries sqrt(k) of coefficient mass per unit loading.
            sindy_library_polynomial_degree: Max polynomial degree for SINDy candidate library.
            sindy_pruning_frequency: Epochs between pruning events.
            sindy_threshold_pruning: Minimum |coefficient| for a member to count as
                supporting a term in the ensemble ratio test (None = disabled).
            sindy_ensemble_pruning: Minimum fraction of ensemble members that must
                exceed sindy_threshold_pruning for a term to survive. Primary pruning mechanism.
            sindy_ridge: Use closed-form ridge regression to initialize SINDy coefficients in Stage 2.2.
                Falls back to SGD on failure. Set False to use pure SGD. (default: True)
            sindy_shooting_steps: Multi-step shooting horizon for Stage 2 SINDy refit.
                1 = one-step-ahead. Values > 1 roll out K steps to penalize compounding error. (default: 20)
            verbose: Print training progress.
            keep_log: Keep full training log (vs. live terminal update).
            save_path_spice: File path (.pkl) to auto-save SPICE model after training.
            compiled_forward: Use @torch.compile for forward loops.
            kwargs_rnn_class: Extra keyword arguments forwarded to spice_class.__init__().
        """
        
        super(BaseEstimator, self).__init__()
        
        # Training parameters
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.n_steps_per_call = n_steps_per_call
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.device = device
        self.verbose = verbose
        self.keep_log = keep_log
        self.deterministic = False
        self.loss_fn = loss_fn
        self.loss_fn_kwargs = loss_fn_kwargs
        self.compiled_forward = compiled_forward

        # Save parameters
        self.save_path_model = save_path_spice

        # SINDy training parameters
        self.sindy_weight = sindy_weight
        self.sindy_lambda_loading = sindy_lambda_loading
        self.sindy_lambda_concept = sindy_lambda_concept
        self.sindy_library_polynomial_degree = sindy_library_polynomial_degree
        self.sindy_pruning_frequency = sindy_pruning_frequency
        self.sindy_threshold_pruning = sindy_threshold_pruning
        self.sindy_ensemble_pruning = sindy_ensemble_pruning
        self.sindy_pruning_terms = sindy_pruning_terms
        self.sindy_refit = sindy_refit
        self.sindy_ridge = sindy_ridge
        self.sindy_shooting_steps = sindy_shooting_steps
        
        # Data parameters
        self.n_actions = n_actions
        self.n_items = n_items
        self.n_reward_features = n_reward_features
        self.n_participants = n_participants
        self.n_experiments = n_experiments
        
        # RNN parameters
        self.l2_rnn = l2_rnn
        self.dropout = dropout
        self.ensemble_size = ensemble_size
        self.embedding_size = embedding_size
        
        # SPICE attributes
        self.spice_config = spice_config
        self.spice_class = spice_class
        self.spice_features = None
        self.kwargs_spice_class = kwargs_spice_class
        self.model = spice_class(
            n_actions=n_actions,
            n_participants=n_participants,
            n_experiments=n_experiments,
            dropout=dropout,
            spice_config=spice_config,
            sindy_polynomial_degree=sindy_library_polynomial_degree,
            sindy_alpha=sindy_lambda_loading,  # doubles as the ridge alpha inside BaseModel
            ensemble_size=ensemble_size,
            embedding_size=embedding_size,
            n_items=n_items,
            n_reward_features=n_reward_features,
            device=device,
            compiled_forward=compiled_forward,
            fit_sindy=sindy_weight > 0,
            **kwargs_spice_class,
        ).to(device)

        self.use_sindy(use_sindy)

        self._build_optimizer()

    def _build_optimizer(self):
        """(Re)build the optimizer over the *current* self.model's parameters.

        Must be called again whenever self.model is replaced (e.g. load_spice), since
        an optimizer holds parameter tensors by identity: left pointing at the previous
        model's tensors it still steps without error, but nothing the forward pass reads
        ever changes.

        Three param groups, tagged by role rather than identified by position: the
        concept directions are shared across the whole population while the loadings are
        per unit, so they are separable knobs. Downstream schedulers look these up by
        'role', never by index.
        """
        direction_params = []
        loading_params = []
        rnn_params = []
        for name, param in self.model.named_parameters():
            if 'sindy_concept_directions' in name:
                direction_params.append(param)
            elif 'sindy_concept_loadings' in name:
                loading_params.append(param)
            else:
                rnn_params.append(param)
        self.rnn_optimizer = torch.optim.AdamW(
            [
            {'params': direction_params, 'weight_decay': 0, 'lr': 0.01, 'role': 'directions'},
            {'params': loading_params, 'weight_decay': 0, 'lr': 0.01, 'role': 'loadings'},
            {'params': rnn_params, 'weight_decay': self.l2_rnn, 'lr': self.learning_rate, 'role': 'rnn'},
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
            n_steps=self.n_steps_per_call,

            convergence_threshold=self.convergence_threshold,
            loss_fn=self.loss_fn,
            loss_fn_kwargs = self.loss_fn_kwargs,

            sindy_weight=self.sindy_weight,
            sindy_lambda_loading=self.sindy_lambda_loading,
            sindy_lambda_concept=self.sindy_lambda_concept,
            sindy_pruning_frequency=self.sindy_pruning_frequency,
            sindy_threshold_pruning=self.sindy_threshold_pruning,
            sindy_ensemble_pruning=self.sindy_ensemble_pruning,
            sindy_pruning_terms=self.sindy_pruning_terms,
            sindy_refit=self.sindy_refit,
            sindy_ridge=self.sindy_ridge,
            sindy_shooting_steps=self.sindy_shooting_steps,

            verbose=self.verbose,
            keep_log=self.keep_log,
            path_save_checkpoints=self.save_path_model,
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
        
        logits = self.model(conditions)[0]
        # SINDy mode: use member 0 (all members fitted to consensus targets)
        # RNN mode: use ensemble mean (no single member saw all data)
        if logits.dim() == 5:
            prediction = logits[0] if self.model.use_sindy else logits.mean(dim=0)
        else:
            prediction = logits
        prediction = prediction.detach().cpu().numpy()
        return prediction
    
    def print_spice_model(self, participant_id: int = 0, experiment_id: int = 0) -> None:
        """
        Get the learned SPICE features and equations.
        """
        
        self.model.print(participant_id=participant_id, experiment_id=experiment_id)

    def get_participant_embeddings(self, ensemble_id: int = 0) -> Dict:
        if hasattr(self.model, 'participant_embedding'):
            participant_ids = torch.arange(self.n_participants, device=self.device, dtype=torch.int32)
            embeddings = self.model.participant_embedding(participant_ids)  # (E, P, D)
            embeddings = embeddings[ensemble_id]  # (P, D)
            return {pid: embeddings[pid] for pid in range(self.n_participants)}
        else:
            print(f'RNN model has no participant_embedding module.')
            return None

    def get_sindy_coefficients(self, key_module: Optional[str] = None, aggregate: bool = False) -> Dict[str, np.ndarray]:
        """Per module, the coefficients implied by the factorization, (E, P, X, T).

        Derived from Z @ V rather than stored. Use get_concept_loadings() for anything
        reporting individual differences -- loadings on a shared dictionary are
        comparable across participants in a way raw per-term coefficients are not.
        """
        
        return self.model.get_sindy_coefficients(key_module=key_module, aggregate=aggregate)
    
    def count_spice_parameters(self):
        """Degrees of freedom, split into per-participant loadings and shared directions.

        Returned as two separate numbers on purpose -- see BaseModel.count_spice_parameters.
        """
        return self.model.count_spice_parameters()

    def get_concepts(self, key_module: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Per module, the (n_concepts, n_terms) population-level concept dictionary."""
        return self.model.get_concepts(key_module=key_module)

    def get_concept_loadings(self, key_module: Optional[str] = None, aggregate: bool = False) -> Dict[str, np.ndarray]:
        """Per module, each participant's non-negative loading on every concept.

        This is the quantity to report individual differences on. Raw per-term
        coefficients are a derived view of it and are not comparable across
        participants in the way loadings on a shared dictionary are.
        """
        return self.model.get_concept_loadings(key_module=key_module, aggregate=aggregate)

    def get_modules(self):
        return self.model.get_modules()
    
    def get_candidate_terms(self, key_module: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        return self.model.get_candidate_terms(key_module=key_module)
        
    def load_spice(self, path_model: str, deterministic: bool = True):
        
        # LOAD RNN MODEL AND OPTIMIZER
                
        # load trained parameters
        loaded_parameters = torch.load(path_model, map_location=torch.device('cpu'))
        
        # Infer ensemble_size from saved loading shape: (E, P, X, n_concepts)
        self.model.ensemble_size = loaded_parameters['model']['sindy_concept_loadings.'+next(iter(self.model.submodules_rnn))].shape[0]
        self.ensemble_size = self.model.ensemble_size
        
        self.model = self.spice_class(
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
            embedding_size=self.model.embedding_size,
            compiled_forward=self.model.compiled_forward,
            **self.kwargs_spice_class,
            )
        
        for module in self.get_modules():
            self.model.setup_sindy_concepts(key_module=module, polynomial_degree=self.model.sindy_specs[module]['polynomial_degree'])
        self.model.sindy_concept_support = loaded_parameters['sindy_concept_support']
        self.model.sindy_concept_gates = loaded_parameters['sindy_concept_gates']
        if 'sindy_term_prior_mask' in loaded_parameters:
            self.model.sindy_term_prior_mask = loaded_parameters['sindy_term_prior_mask']

        self.model.load_state_dict(loaded_parameters['model'])
        self.model.init_state(batch_size=self.model.n_participants)

        self.model = self.model.to(self.model.device)
        self.model.eval()

        # The optimizer built in __init__ holds the discarded model's tensors; without
        # this a subsequent .fit() runs but updates nothing.
        self._build_optimizer()

        # Restore the checkpoint's Adam moments so a resumed run continues on the
        # curvature estimate it stopped with instead of re-warming from zero. Only the
        # per-parameter state is taken: group hyperparameters stay the ones this
        # estimator was constructed with, so a resume never silently inherits an
        # lr the previous run's ReduceLROnPlateau had already annealed.
        if 'optimizer' in loaded_parameters:
            hyperparameters = [
                {k: v for k, v in group.items() if k != 'params'}
                for group in self.rnn_optimizer.param_groups
            ]
            try:
                self.rnn_optimizer.load_state_dict(loaded_parameters['optimizer'])
            except ValueError as error:
                warnings.warn(
                    f"Could not restore optimizer state from {path_model} ({error}); "
                    "continuing with freshly initialized moments."
                )
            else:
                for group, saved in zip(self.rnn_optimizer.param_groups, hyperparameters):
                    group.update(saved)
            
    def save_spice(self, path_rnn: str):
        """
        Save the SPICE model (RNN weights, optimizer state, and concept structure) to a .pkl file.

        Args:
            path_rnn: File path to save the model.
        """
        
        # Save RNN model
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.rnn_optimizer.state_dict(),
            'sindy_concept_support': self.model.sindy_concept_support,
            'sindy_concept_gates': self.model.sindy_concept_gates,
            'sindy_term_prior_mask': self.model.sindy_term_prior_mask,
            }
        torch.save(state_dict, path_rnn)
        
    def set_device(self, device: torch.device):
        self.model.to(device)
        self.device = device
    
    def use_sindy(self, mode: bool = True):
        self._use_sindy = mode
        self.model.use_sindy = mode
    
    def eval(self, use_sindy: bool = True):
        self.model.eval(use_sindy=use_sindy)
        self.use_sindy(mode=use_sindy)
        
    def train(self, mode: bool = True, use_sindy: bool = False):
        self.model.train(mode=mode, use_sindy=use_sindy)
        self.use_sindy(mode=self.use_sindy)
    
    def __call__(self, conditions: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        return self.model(conditions, state)