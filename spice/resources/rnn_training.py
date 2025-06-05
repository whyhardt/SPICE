import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import time
import numpy as np
from spice.resources.rnn import BaseRNN, CustomEmbedding
from spice.resources.rnn_utils import DatasetRNN


def gradient_penalty(f: nn.Module, e_i: torch.Tensor, e_j: torch.Tensor, factor=1.0):
    
    # one-hot encode
    e_i = f[0].one_hot_encode(e_i)
    e_j = f[0].one_hot_encode(e_j)
    
    # e_i, e_j: one-hot tensors, shape [batch, P]
    eps = torch.rand(e_i.size(0), 1).to(e_i.device)
    eps = eps.expand_as(e_i)
    
    # Interpolation in participant‐ID space
    e_hat = eps*e_i + (1-eps)*e_j
    e_hat.requires_grad_(True)

    # compute interpolated embedding
    emb_hat = f[1](f[0].get_embedding(e_hat))
    
    # Compute Jacobian norm
    grads = torch.autograd.grad(
        outputs=emb_hat,
        inputs=e_hat,
        grad_outputs=torch.ones_like(emb_hat),
        create_graph=True
    )[0]  # shape [batch, P]
    gp = (grads.view(grads.size(0), -1).norm(2, dim=1) ** 2).mean()
    return factor * gp


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
    n_steps: int = -1,
    l1_weight_decay: float = 1e-4,
    l2_weight_decay: float = 1e-4,
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
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
        
        mask = xs_step[..., :1] > -1
        
        state = model.get_state(detach=True)
        ys_pred = model(xs_step, state, batch_first=True)[0]
        
        ys_pred = ys_pred * mask
        ys_step = ys_step * mask
        
        loss_step = loss_fn(
            ys_pred.reshape(-1, model._n_actions), 
            torch.argmax(ys_step.reshape(-1, model._n_actions), dim=1),
            )
        
        loss_batch += loss_step
        iterations += 1
        
        if torch.is_grad_enabled():
            
            # L1 weight decay to enforce sparsification in the network (except for participant embedding)
            l1_reg = l1_weight_decay * torch.stack([
                param.abs().sum()
                for name, param in model.named_parameters()
                # if "embedding" not in name
                # if "embedding" in name
                ]).mean()
            
            # Regularization of the embedding space
            if l2_weight_decay > 0:
                if hasattr(model, 'participant_embedding') and isinstance(model.participant_embedding, nn.Sequential) and isinstance(model.participant_embedding[0], CustomEmbedding):
                    # gradient penalty between two participants
                    # sample two random distributions of participant indices as one-hot-encoded tensors
                    e_i = torch.randint(low=0, high=model.n_participants, size=(xs.shape[0],), dtype=torch.int64, device=xs_step.device)
                    e_j = torch.randint(low=0, high=model.n_participants, size=(xs.shape[0],), dtype=torch.int64, device=xs_step.device)
                    embedding_reg = gradient_penalty(model.participant_embedding, e_i, e_j, factor=l2_weight_decay)
                elif hasattr(model, 'participant_embedding') and ((isinstance(model.participant_embedding, nn.Sequential) and isinstance(model.participant_embedding[0], nn.Embedding)) or isinstance(model.participant_embedding, nn.Embedding)):
                    # L2 weight decay on participant embedding to enforce smoother gradients between participants and prevent overfitting
                    if model.embedding_size > 1:
                        embedding_reg = l2_weight_decay * torch.stack([
                            param.pow(2).sum()
                            for name, param in model.named_parameters()
                            if "embedding" in name
                            # if "embedding" not in name
                            ]).mean()
                    else:
                        embedding_reg = 0
                else:
                    embedding_reg = 0
            else:
                embedding_reg = 0
                
            loss = loss_step + l1_reg + embedding_reg
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model, optimizer, loss_batch.item()/iterations
    

def fit_model(
    model: BaseRNN,
    dataset_train: DatasetRNN,
    dataset_test: DatasetRNN = None,
    optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-7,
    epochs: int = 1,
    batch_size: int = -1,
    bagging: bool = False,
    scheduler: bool = False,
    n_steps: int = -1,
    l1_weight_decay: float = 1e-4,
    l2_weight_decay: float = 1e-4,
    verbose: bool = True,
    ):
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
        l1_weight_decay (float, optional): L1 weight decay for sparsification. Defaults to 1e-4.
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
    if dataset_test is not None:
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))
    
    # set up learning rate scheduler
    if scheduler and optimizer is not None:
        warmup_steps = 64 if epochs > 64 else 0 #int(epochs * 0.125/16)
        # Define the LambdaLR scheduler for warm-up
        def warmup_lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps)) * 10
            return 1.0  # No change after warm-up phase

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
    
    # start training
    while continue_training:
        try:
            loss_train = 0
            loss_test = 0
            t_start = time.time()
            n_calls_to_train_model += 1
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
                    l2_weight_decay=l2_weight_decay,
                )
                loss_train += loss_i
            loss_train /= iterations_per_epoch
            
            if dataset_test is not None:
                model.eval()
                with torch.no_grad():
                    xs, ys = next(iter(dataloader_test))
                    if xs.device != model.device:
                        xs = xs.to(model.device)
                        ys = ys.to(model.device)
                    # evaluate model
                    _, _, loss_test = batch_train(
                        model=model,
                        xs=xs,
                        ys=ys,
                        optimizer=optimizer,
                    )
                model.train()
            
            # check for convergence
            dloss = last_loss - loss_test if dataset_test is not None else last_loss - loss_train
            convergence_value += recency_factor * (np.abs(dloss) - convergence_value)
            converged = convergence_value < convergence_threshold
            continue_training = not converged and n_calls_to_train_model < epochs
            last_loss = 0
            last_loss += loss_test if dataset_test is not None else loss_train
            
            msg = None
            if verbose:
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
                    
        except KeyboardInterrupt:
            continue_training = False
            msg = 'Training interrupted. Continuing with further operations...'
        if verbose:
            print(msg)
            
    return model, optimizer, loss_train