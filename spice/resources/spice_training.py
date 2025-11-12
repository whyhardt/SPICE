import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from typing import Tuple

from .rnn import BaseRNN
from .spice_utils import SpiceDataset
from .sindy_differentiable import threshold_coefficients


class ReduceOnPlateauWithRestarts:
    def __init__(self, optimizer, min_lr, factor, patience):
        """
        Plateau-based LR scheduler with restarts to the base learning rate when min_lr is hit.

        Parameters:
        - optimizer: Optimizer instance.
        - min_lr: The minimum learning rate after reductions.
        - factor: Multiplicative factor to reduce the LR on plateau.
        - patience: Number of epochs with no improvement before reducing the LR.
        """
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]  # Extract base LRs
        self.factor = factor
        self.patience = patience
        self.base_patience = patience
        
        self.best = float('inf')  # Initialize the best validation loss as infinity
        self.num_bad_epochs = 0  # Initialize the count of bad epochs
        self.num_cycles_completed = 0
        
        # Store the current learning rate for each parameter group
        self.current_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, metrics):
        """
        Update the learning rate based on the validation loss.
        """
        if metrics < self.best:
            self.best = metrics
            self.num_bad_epochs = 0  # Reset bad epochs counter
        else:
            self.num_bad_epochs += 1
        
        # Check if patience is exceeded
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()  # Reduce learning rates
            self._adjust_patience()  # Adjust the patience according to the learning rate
            self.num_bad_epochs = 0  # Reset bad epochs counter

    def _reduce_lr(self):
        """
        Reduce the learning rate for all parameter groups by the given factor.
        """
        for i, param_group in enumerate(self.optimizer.param_groups):            
            
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            # Check if the new learning rate has hit min_lr and reset if so
            if new_lr <= self.min_lr:
                param_group['lr'] = self.base_lrs[i]
                self.num_cycles_completed += 1
            else:
                param_group['lr'] = new_lr
    
    def _adjust_patience(self):
        """
        Adjust the patience according to the learning rate.
        """
        # self.patience = max([self.patience * (1+self.num_cycles_completed) if self.get_lr()[-1] < self.base_lrs[-1] else self.base_patience, 200])
        self.patience = self.patience * 2 if self.get_lr()[-1] < self.base_lrs[-1] else self.base_patience

    def get_lr(self):
        """
        Retrieve the current learning rates for all parameter groups.
        """
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_last_lr(self):
        """
        Retrieve the last computed learning rates for all parameter groups.
        """
        return [group['lr'] for group in self.optimizer.param_groups]
    

def batch_train(
    model: BaseRNN,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    l1_weight_decay: float = 0.,
    sindy_weight: float = 0.,
    n_steps: int = -1,
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(label_smoothing=0.),
    ):

    """
    Trains a model with the given batch.
    """

    if n_steps == -1:
        n_steps = xs.shape[1]

    model.set_initial_state(batch_size=len(xs))
    state = model.get_state(detach=True)

    loss_batch = 0
    iterations = 0
    for t in range(0, xs.shape[1], n_steps):
        n_steps = min(xs.shape[1]-t, n_steps)
        xs_step = xs[:, t:t+n_steps]
        ys_step = ys[:, t:t+n_steps]
        
        # Mask out padding (NaN values) - valid trials have non-NaN actions
        mask = ~torch.isnan(xs_step[..., :model.n_actions].sum(dim=-1, keepdim=True).repeat(1, 1, model.n_actions))
        
        state = model.get_state(detach=True)
        ys_pred, _ = model(xs_step, state, batch_first=True)
        
        ys_pred = torch.where(mask, ys_pred, torch.zeros_like(ys_pred))
        ys_step = torch.where(mask, ys_step, torch.zeros_like(ys_step))
        
        loss_step = loss_fn(
            ys_pred.reshape(-1, model.n_actions),
            torch.argmax(ys_step.reshape(-1, model.n_actions), dim=1),
            )

        # small l2-regularization on logits to keep the absolute values in the smalles possible range (only diff between values is necessary)
        loss_step += 0.001 * ys_pred.abs().sum(dim=-1).mean()
        
        # Add SINDy regularization loss
        if sindy_weight > 0 and model.sindy_loss != 0:
            loss_step = loss_step + sindy_weight * model.sindy_loss
            
        loss_batch += loss_step
        iterations += 1
        
        if torch.is_grad_enabled():

            # backpropagation
            optimizer.zero_grad()
            loss_step.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model, optimizer, loss_batch.item()/iterations
    

def fit_sindy_second_stage(
    model: BaseRNN,
    dataset_train: SpiceDataset,
    dataset_test: SpiceDataset = None,
    learning_rate: float = 1e-3,
    epochs: int = 1,
    threshold: float = 0.05,
    batch_size: int = -1,
    verbose: bool = True,
    ):
    """
    Second stage SINDy fitting: freeze RNN weights and refit SINDy coefficients
    on the trained RNN hidden states with no thresholding (threshold=0).

    This follows the approach from sindy-shred where SINDy coefficients are
    discarded after initial training and refitted on the learned hidden states.

    IMPORTANT: Collapses ensemble to a single SINDy model (ensemble_size=1).

    Args:
        model (BaseRNN): Trained RNN model with SINDy coefficients
        dataset_train (DatasetRNN): Training dataset
        dataset_test (DatasetRNN, optional): Validation dataset. Defaults to None.
        learning_rate (float): Learning rate for SINDy coefficient optimization
        epochs (int): Number of epochs for second stage training
        batch_size (int): Batch size for training
        verbose (bool): Print progress

    Returns:
        BaseRNN: Model with refitted SINDy coefficients (single model, no ensemble)
    """

    # Always print header for second stage (important step)
    print("\n" + "="*80)
    print("Starting second stage SINDy fitting (threshold=0, single model)")
    print("="*80)

    criterion = nn.MSELoss()

    # Freeze all RNN parameters, only train SINDy coefficients
    for param in model.parameters():
        param.requires_grad = False

    # COLLAPSE ENSEMBLE TO SINGLE MODEL
    # Set ensemble size to 1 for stage 2 (no ensemble in second stage)
    model.sindy_ensemble_size = 1

    # Re-initialize SINDy coefficients with single model (ensemble_size=1)
    model.setup_sindy_coefficients(model.sindy_polynomial_degree, ensemble_size=1)
    model.to(model.device)

    # Create optimizer for only SINDy coefficients
    sindy_params = list(model.sindy_coefficients.values())
    optimizer_sindy = torch.optim.AdamW(sindy_params, lr=learning_rate, weight_decay=0.01)
    
    # Set model to training mode for SINDy loss computation
    model.train()
    model.use_sindy = False
    
    # --------------------------------------------------------
    # VECTORIZATION OF SINDY TRAINING
    # --------------------------------------------------------
    # rnn-weights are frozen -> we can vectorize the whole SINDy training by computing all rnn states for each timestep ad-hoc
    
    def vectorize_training_data(dataset):
        # Initialize model state for correct batch size based on actual dataset
        # dataset.xs.shape[0] gives the number of sessions/participants in first dimension
        len_dataset = dataset.xs.shape[0]
        n_timesteps = dataset.xs.shape[1]
        model.set_initial_state(batch_size=len_dataset)

        # initialize state buffer with time-flattened batch-dim
        state_buffer_current = {state: torch.zeros((n_timesteps*len_dataset, model.n_items), dtype=torch.float32, device=model.device) for state in model.get_state()}
        state_buffer_next = {state: torch.zeros((n_timesteps*len_dataset, model.n_items), dtype=torch.float32, device=model.device) for state in model.get_state()}

        with torch.no_grad():
            for t in range(n_timesteps):
                # save current state (input to forward-pass) in state buffer
                for state in state_buffer_next:
                    state_buffer_current[state][t*len_dataset:(t+1)*len_dataset] = model.get_state()[state]

                # compute updated state
                updated_state = model(dataset.xs[:, t][:, None].to(model.device), model.get_state(), batch_first=True)[1]
                
                # save updated state (training target) in state buffer
                for state in state_buffer_next:
                    state_buffer_next[state][t*len_dataset:(t+1)*len_dataset] = updated_state[state]
        
        # reshape the dataset to be aligned with state buffer
        xs = dataset.xs.transpose(0, 1).reshape(n_timesteps*len_dataset, 1, -1)
        ys = dataset.ys.transpose(0, 1).reshape(n_timesteps*len_dataset, 1, -1)
        dataset = SpiceDataset(xs, ys)
        
        return state_buffer_current, state_buffer_next, dataset
    
    input_state_buffer_train, target_state_buffer_train, dataset_train = vectorize_training_data(dataset_train)
    if dataset_test:
        input_state_buffer_test, target_state_buffer_test, dataset_test = vectorize_training_data(dataset_test)
    
    
    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------
    
    model.use_sindy = True
    xs = dataset_train.xs
    nan_mask = ~torch.isnan(xs[:, 0, :model.n_actions].sum(dim=-1))
    batch_size = dataset_train.xs.shape[0] if batch_size == -1 else batch_size
    len_dataset = dataset_train.xs.shape[0]
    
    for epoch in range(epochs):
        t_start = time.time()
        loss_train = 0
        iterations = 0
        
        for idx in range(0, len_dataset, batch_size):
            
            optimizer_sindy.zero_grad()
            
            batched_nan_mask = nan_mask[idx:idx+batch_size].to(model.device)
            batched_input_state_buffer, batched_target_state_buffer = {}, {}
            for state in input_state_buffer_train:
                batched_input_state_buffer[state] = input_state_buffer_train[state][idx:idx+batch_size][batched_nan_mask]
                batched_target_state_buffer[state] = target_state_buffer_train[state][idx:idx+batch_size][batched_nan_mask]
            
            # get sindy-based state updates from original rnn states
            state_pred = model(xs[idx:idx+batch_size, :1].to(model.device)[batched_nan_mask], batched_input_state_buffer, batch_first=True)[1]
            
            loss_batch = 0
            for state in model.spice_config.states_in_logit:
                loss_batch += criterion(batched_target_state_buffer[state], state_pred[state])

            # Backward pass - only update SINDy coefficients
            loss_batch.backward()
            optimizer_sindy.step()
            
            loss_train += loss_batch.item()
            iterations += 1
            
        loss_train /= iterations

        loss_test = 0
        
        # THRESHOLDING STEP
        if epoch % 100 == 0 and epoch != 0:
                model.thresholding(threshold=0, base_threshold=threshold, n_terms_cutoff=1)
        
        # Print progress
        msg = f'SINDy Stage 2 - Epoch {epoch+1}/{epochs} --- L(Train): {loss_train:.7f}'
        if dataset_test is not None:
            msg += f'; L(Val): {loss_test:.7f}'
        msg += f'; Time: {time.time()-t_start:.2f}s'
        print(msg, end='\r' if epoch < epochs-1 else '\n')

    # Restore requires_grad for all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Always print completion message
    print("="*80)
    print("Second stage SINDy fitting complete!")
    print("="*80)

    if verbose:
        print("\nRefitted SPICE model (participant 0):")
        print("-"*80)
        model.print_spice_model(participant_id=0)
        print("-"*80)
    
    return model.train()


def fit_model(
    model: BaseRNN,
    dataset_train: SpiceDataset,
    dataset_test: SpiceDataset = None,
    optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-7,
    l1_weight_decay: float = 0.,
    sindy_weight: float = 0.,
    sindy_epochs: int = 1000,
    sindy_threshold: float = 0.01,
    sindy_threshold_frequency: int = 100,
    epochs: int = 1,
    batch_size: int = -1,
    bagging: bool = False,
    scheduler: bool = False,
    n_steps: int = -1,
    verbose: bool = True,
    path_save_checkpoints: str = None,
    ) -> Tuple[BaseRNN, torch.optim.Optimizer, float]:
    """_summary_

    Args:
        model (BaseRNN): A child class of the BaseRNN, which implements the forward method
        dataset_train (DatasetRNN): training data for the RNN of shape (Batch, Timesteps, Features) with Features being (Actions, Rewards, Participant ID) -> (n_actions, n_actions, 1)
        dataset_test (DatasetRNN, optional): Validation dataset during training. Defaults to None.
        optimizer (torch.optim.Optimizer, optional): Torch-optimizer. Defaults to None.
        convergence_threshold (float, optional): Threshold of convergence value, which determines early stopping of training. Defaults to 1e-5.
        epochs (int, optional): Total number of training epochs. Defaults to 1.
        batch_size (int, optional): Batch size. Defaults to -1.
        bagging (bool, optional): Enables bootstrap aggregation. Defaults to False.
        n_steps (int, optional): Number of steps passed at once through the RNN to compute a gradient over steps. Defaults to -1.
        verbose (bool, optional): Verbosity. Defaults to True.

    Returns:
        (BaseRNN, Optimizer, float): (Trained RNN, Optimizer with last state, Training loss)
    """
    
    
    # initialize dataloader
    if batch_size == -1:
        batch_size = len(dataset_train)
    
    # use random sampling with replacement
    if bagging:
        batch_size = max(batch_size, 64)
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=batch_size) if bagging else None
    else:
        sampler = None
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, shuffle=True if sampler is None else False)
    # if dataset_test is None:
    #     dataset_test = dataset_train
    if dataset_test is not None:
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))
    
    # set up learning rate scheduler
    warmup_steps = 0
    warmup_steps = warmup_steps if epochs > warmup_steps else 1 #int(epochs * 0.125/16)
    if scheduler and optimizer is not None:
        # Define the LambdaLR scheduler for warm-up
        # def warmup_lr_lambda(current_step):
        #     if current_step < warmup_steps / 0.8:
        #         return min(1e-1, float(current_step) / float(max(1, warmup_steps))) / (optimizer.param_groups[0]['lr']) / 100
        #     else:
        #         return 1 - float(current_step) / float(max(1, warmup_steps)) / (optimizer.param_groups[0]['lr']) / 100
        #     return 1.0  # No change after warm-up phase
        default_lr = optimizer.param_groups[0]['lr'] + 0
        def warmup_lr_lambda(current_step):
            scale = 1e-1 / default_lr
            # if current_step < warmup_steps * 0.8:
            #     return 0.1 * scale  # Scaling factor during the first 80% of warmup
            # elif current_step < warmup_steps:
            #     # Linearly anneal towards 1.0 in the last 20% of warmup steps
            #     progress = (current_step - warmup_steps * 0.8) / (warmup_steps * 0.2)
            #     return (0.1 + progress * (1.0 - 0.1)) * scale
            if current_step < warmup_steps:
                # Linearly anneal towards 1.0 in the last 20% of warmup steps
                # progress = (current_step - warmup_steps) / (warmup_steps)
                # return (0.1 + progress * (1.0 - 0.1)) * scale
                return current_step / warmup_steps
            else:
                return 1.0  # Default learning rate scaling after warmup

        # Create the scheduler with the Lambda function
        scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
                
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=warmup_steps if warmup_steps > 0 else 64, T_mult=2)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=10, threshold=0, cooldown=64, min_lr=1e-6)
        # scheduler = ReduceOnPlateauWithRestarts(optimizer=optimizer, min_lr=1e-6, factor=0.1, patience=8)
    else:
        scheduler_warmup, scheduler = None, None
        
    if epochs == 0:
        continue_training = False
        msg = 'No training epochs specified. Model will not be trained.'
        if verbose:
            print(msg)
    else:
        continue_training = True
        converged = False
        n_calls_to_train_model = 0
        convergence_value = 1
        last_loss = 1
        recency_factor = 0.5
    
    loss_train = 0
    loss_test = 0
    iterations_per_epoch = max(len(dataset_train), 64) // batch_size
    save_at_epoch = warmup_steps
    
    # start training
    while continue_training:
        try:
            loss_train = 0
            loss_test = 0
            t_start = time.time()
            for _ in range(iterations_per_epoch):
                # get next batch
                xs, ys = next(iter(dataloader_train))
                if xs.device != model.device:
                    xs = xs.to(model.device)
                    ys = ys.to(model.device)
                # train model
                model, optimizer, loss_i = batch_train(
                    model=model,
                    xs=xs,
                    ys=ys,
                    optimizer=optimizer,
                    n_steps=n_steps,
                    l1_weight_decay=l1_weight_decay,
                    sindy_weight=sindy_weight,
                )
                loss_train += loss_i
            
            n_calls_to_train_model += 1
            loss_train /= iterations_per_epoch
            
            if dataset_test is not None:
                model = model.eval(use_sindy=True)
                with torch.no_grad():
                    xs, ys = next(iter(dataloader_test))
                    if xs.device != model.device:
                        xs = xs.to(model.device)
                        ys = ys.to(model.device)
                    # evaluate model
                    _, _, loss_test = batch_train(model=model, xs=xs, ys=ys)
                model = model.train()
            
            # periodic pruning of sindy coefficients with L0 norm
            if sindy_weight > 0 and n_calls_to_train_model > warmup_steps and n_calls_to_train_model % sindy_threshold_frequency == 0 and n_calls_to_train_model != 0:
                if n_calls_to_train_model == warmup_steps+sindy_threshold_frequency:
                    print("\n"+"="*80)
                    print(f"SPICE model before {n_calls_to_train_model} epochs:")
                    print("="*80)
                    model.print_spice_model(ensemble_idx=4)
                    
                model.thresholding(threshold=sindy_threshold, base_threshold=0.1, n_terms_cutoff=1)
                
                print("\n"+"="*80)
                print(f"SPICE model after {n_calls_to_train_model} epochs:")
                print("="*80)
                model.print_spice_model(ensemble_idx=4)    
            
            # check for convergence
            dloss = last_loss - loss_test if dataset_test is not None else last_loss - loss_train
            convergence_value += recency_factor * (np.abs(dloss) - convergence_value)
            converged = convergence_value < convergence_threshold
            continue_training = not converged and n_calls_to_train_model < epochs
            last_loss = 0
            last_loss += loss_test if dataset_test is not None else loss_train
            
            msg = None
            # if verbose:
            msg = f'Epoch {n_calls_to_train_model}/{epochs} --- L(Train): {loss_train:.7f}'                
            if dataset_test is not None:
                msg += f'; L(Val): {loss_test:.7f}'
            msg += f'; Time: {time.time()-t_start:.2f}s; Convergence: {convergence_value:.2e}'
            if scheduler is not None:
                msg += f'; LR: {scheduler_warmup.get_last_lr()[-1] if n_calls_to_train_model < warmup_steps else scheduler.get_last_lr()[-1]:.2e}'
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) or isinstance(scheduler, ReduceOnPlateauWithRestarts):
                    msg += f"; Metric: {scheduler.best:.7f}; Bad epochs: {scheduler.num_bad_epochs}/{scheduler.patience}"
            if converged:
                msg += '\nModel converged!'
            elif n_calls_to_train_model >= epochs:
                msg += '\nMaximum number of training epochs reached.'
                if not converged:
                    msg += '\nModel did not converge yet.'
                        
            if scheduler is not None:
                if n_calls_to_train_model <= warmup_steps:
                    scheduler_warmup.step()
                else:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) or isinstance(scheduler, ReduceOnPlateauWithRestarts):
                        scheduler.step(metrics=loss_test if dataset_test is not None else loss_train)
                    # elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    #     scheduler.step(epoch=n_calls_to_train_model)
                    else:
                        scheduler.step()

            # save checkpoint
            if path_save_checkpoints and n_calls_to_train_model == save_at_epoch:
                torch.save(model.state_dict(), path_save_checkpoints.replace('.', f'_ep{n_calls_to_train_model}.'))
                save_at_epoch *= 2
                
        except KeyboardInterrupt:
            continue_training = False
            msg = 'Training interrupted. Continuing with further operations...'

        #if verbose:
        print(msg, end='\r')
    
    model.rnn_training_finished = True
    
    # Second stage: Refit SINDy coefficients on trained RNN hidden states (always run when sindy_weight > 0)
    if sindy_weight > 0:
        model = fit_sindy_second_stage(
            model=model,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            learning_rate=optimizer.param_groups[0]['lr'],
            epochs=sindy_epochs,
            threshold=sindy_threshold,
            batch_size=-1,
            verbose=verbose,
        )
        
    return model.eval(), optimizer, loss_train