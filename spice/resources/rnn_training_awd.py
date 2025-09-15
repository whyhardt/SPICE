import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import time
import numpy as np

from .rnn import BaseRNN
from .rnn_utils import DatasetRNN

# Create Mask
def apply_mask(preds, ss):
    mask = ss[..., :1] > -1
    return preds * mask

def fit_with_metaopt(
    model: BaseRNN,
    dataset_train: DatasetRNN,
    dataset_val: DatasetRNN = None,
    model_optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-7,
    epochs: int = 1,
    batch_size: int = -1,
    bagging: bool = False,
    scheduler: bool = False,
    n_steps: int = -1,
    verbose: bool = True,
    path_save_checkpoints: str = None,

    lambda_awd: float = 0.022,  # Default from paper experiments
    ):

    """
    Training loop with Adaptive Weight Decay implementation based on Ghiasi et al. (2023)
    
    Args:
        lambda_awd: The AWD hyperparameter (λ_awd in the paper)
    """
    # Set batch size
    if batch_size == -1:
        batch_size = len(dataset_train)

    # Set bagging
    if bagging:
        batch_size = max(batch_size, 64)
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=batch_size)
    else:
        raise NotImplementedError("Bagging=False is not implemented in this function.")

    # Leave at 0 for compatibility
    num_workers = 0
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    if dataset_val is not None:
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        raise ValueError("Validation dataset is required for training with meta-optimization.")

    # Set up learning rate scheduler
    warmup_steps = 1024 if epochs > 1024 else 1
    if scheduler and model_optimizer is not None:
        default_lr = model_optimizer.param_groups[0]['lr'] + 0

        def warmup_lr_lambda(current_step):
            scale = 1e-1 / default_lr
            if current_step < warmup_steps * 0.8:
                return 0.1 * scale  # Scaling factor during the first 80% of warmup
            elif current_step < warmup_steps:
                # Linearly anneal towards 1.0 in the last 20% of warmup steps
                progress = (current_step - warmup_steps * 0.8) / (warmup_steps * 0.2)
                return (0.1 + progress * (1.0 - 0.1)) * scale
            else:
                return 1.0  # Default learning rate scaling after warmup
        
        scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(model_optimizer, lr_lambda=warmup_lr_lambda)
                
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=model_optimizer, T_0=warmup_steps if warmup_steps > 0 else 64, T_mult=2)
    else:
        scheduler_warmup, scheduler = None, None

    # Epoch error handling
    if epochs == 0:
        raise ValueError("Epochs must be specified.")

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Define convergence variables
    convergence_value = 1
    last_loss = 1
    recency_factor = 0.5

    # Initialize losses and histories for logging/plotting
    train_loss = torch.tensor(0.)
    val_loss = torch.tensor(0.)

    history_train_loss = []
    history_val_loss = []
    history_lambda_wd = []

    # Step 2 in Algorithm:
    # Initialize moving average of the lambda we are adapting
    lambda_wd_bar = 0.0

    # Set up at which epoch to save
    save_at_epoch = warmup_steps

    # Training loop
    try:
        # Step 3 in Algorithm: 
        # Iterate over batches in epochs, here just epochs
        for epoch in range(epochs):
            t_start = time.time()
            train_iter = iter(dataloader_train)
            val_iter = iter(dataloader_val)

            # Data loading using iters
            try:
                xs, ys = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader_train)
                xs, ys = next(train_iter)
            xs, ys = xs.to(model.device), ys.to(model.device)

            try:
                xs_val, ys_val = next(val_iter)
            except StopIteration:
                val_iter = iter(dataloader_val)
                xs_val, ys_val = next(val_iter)
            xs_val, ys_val = xs_val.to(model.device), ys_val.to(model.device)

            model.train()
            model.set_initial_state(batch_size=len(xs))
            model_optimizer.zero_grad()

            # Ensure fresh computational graph
            state = model.get_state(detach=True)
            if hasattr(state, 'detach'):
                state = state.detach()

            # Step 4 in Algorithm: 
            # Get model predictions
            outputs = model(xs, state, batch_first=True)[0]
            outputs = apply_mask(outputs, xs)
            
            # Step 5 in Algorithm: 
            # Calculate the main loss (CrossEntropy)
            train_loss = loss_fn(outputs.reshape(-1, model._n_actions),
                               torch.argmax(ys.reshape(-1, model._n_actions), dim=1))

            # Step 6 in Algorithm:
            # Compute gradients of main loss w.r.t weights
            train_loss.backward()

            # AWD Section:
            # Step 7 in the Algorithm: 
            # Compute weight decay hyperparameter

            # Compute L2 gradient norm
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += (param.grad ** 2).sum().item()
            grad_norm = np.sqrt(grad_norm)
            
            # Compute L2 weight norm
            weight_norm = 0.0
            for param in model.parameters():
                weight_norm += (param.data ** 2).sum().item()
            weight_norm = np.sqrt(weight_norm)
            
            # Compute weight decay parameter
            if weight_norm > 1e-8:  # Error handling: Avoid division by zero
                lambda_wd_current = (lambda_awd * grad_norm) / weight_norm
            else:
                lambda_wd_current = 0.0

            # Step 8 in Algorithm: 
            # Compute exponential weighted/moving average (EMA) (using papers coefficients)
            lambda_wd_bar = 0.1 * lambda_wd_bar + 0.9 * lambda_wd_current

            # Step 9 in Algorithm: 
            # Update network parameters with adaptive weight decay
            # Manually add weight decay directly to gradients
            # This is equivalent to applying weight decay in the optimizer
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.add_(param.data, alpha=lambda_wd_bar)

            # Step 10 in Algorithm: 
            # Update model parameters, with weight decay included in the gradients
            model_optimizer.step()

            model.eval()
            model.set_initial_state(batch_size=len(xs_val))
            state = model.get_state(detach=True)
            val_preds = model(xs_val, state, batch_first=True)[0]
            val_preds = apply_mask(val_preds, xs_val)

            val_loss = loss_fn(
                val_preds.reshape(-1, model._n_actions),
                torch.argmax(ys_val.reshape(-1, model._n_actions), dim=1))


            history_train_loss.append(train_loss.item())
            history_val_loss.append(val_loss.item())
            history_lambda_wd.append(lambda_wd_bar)

            dloss = last_loss - val_loss
            convergence_value += recency_factor * (torch.abs(dloss).item() - convergence_value)
            converged = convergence_value < convergence_threshold
            last_loss = val_loss 

            msg = None
            if verbose:
                # Shortened num decimal points displayed
                msg = f"Epoch {epoch + 1}/{epochs} --- L(Train): {train_loss:.4f}"
                msg += f"; L(Val): {val_loss:.4f}"
                msg += f"; λ_wd: {lambda_wd_bar:.6f}"
                msg += f"; Time: {time.time()-t_start:.2f}"

                if converged:
                    msg += " --- Converged!"
                print(msg)

            if scheduler is not None:
                if epoch < warmup_steps:
                    scheduler_warmup.step()
                else:
                    scheduler.step()
                
            # Save checkpoint
            if path_save_checkpoints and (epoch + 1) == save_at_epoch:
                torch.save(model.state_dict(), path_save_checkpoints.replace(".", f"_ep{epoch + 1}."))
                save_at_epoch *= 2
            
    except KeyboardInterrupt:
        msg = "Training interrupted. Continuing with further operations..."
        
        histories = [history_train_loss, history_val_loss, history_lambda_wd]
        return model, model_optimizer, histories

    histories = [history_train_loss, history_val_loss, history_lambda_wd]
    return model, model_optimizer, histories