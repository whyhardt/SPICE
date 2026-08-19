"""Terminal reporting and environment probing for the training pipelines."""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from typing import Tuple, Union, Optional
import shutil
from torch.nn.functional import mse_loss

from ..model import BaseModel
from ..spice_utils import SpiceDataset

# === DEBUG MODE: Set to False for live-updating display, True for line-by-line output ===
DEBUG_MODE = False


def _get_terminal_width() -> int:
    try:
        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 160
    return terminal_width

def _check_cuda_oom(exception) -> bool:
    return isinstance(exception, RuntimeError) and "out of memory" not in str(exception) and "CUBLAS_STATUS_ALLOC_FAILED" not in str(exception)

def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


def _print_training_status(
    len_last_print: int,
    model: BaseModel,
    n_calls: int,
    epochs: int,
    loss_train: float,
    loss_test_rnn: float,
    loss_test_sindy: float,
    time_elapsed: float,
    convergence_value: float,
    sindy_weight: float,
    lr=None,
    warmup_steps: int = 0,
    converged: bool = False,
    finished: bool = False,
    keep_log: bool = False,
    is_notebook: bool = False,
):
    """Print live-updating training status block."""
    
    # Build comprehensive status display
    terminal_width = _get_terminal_width()
    if is_notebook:
        terminal_width = terminal_width*2
    status_lines = []
    status_lines.append("=" * terminal_width)
    
    # Training progress bar (tqdm-style)
    postfix_parts = {}
    postfix_parts['L(Train)'] = f'{loss_train:.7f}'
    if loss_test_rnn is not None:
        postfix_parts['L(Val,RNN)'] = f'{loss_test_rnn:.7f}'
    if loss_test_sindy is not None:
        postfix_parts['L(Val,SINDy)'] = f'{loss_test_sindy:.7f}'
    if convergence_value is not None:
        postfix_parts['Conv'] = f'{convergence_value:.2e}'
    if lr is not None:
        postfix_parts['LR'] = f'{lr:.2e}'

    postfix_str = ', '.join(f'{k}={v}' for k, v in postfix_parts.items())
    bar_str = tqdm.format_meter(
        n=n_calls,
        total=epochs,
        elapsed=time_elapsed if time_elapsed is not None else 0,
        ncols=terminal_width,
        postfix=postfix_str,
    )
    status_lines.append(bar_str)
    
    max_len_module = max([len(module) for module in model.get_modules()])
    
    # Add SPICE model equations if SINDy is active
    if sindy_weight > 0:
        status_lines.append("-" * terminal_width)
        counts = model.count_spice_parameters()
        status_lines.append(
            f"SPICE Model (loadings/participant: {counts['loadings'][0, 0]:.0f}, "
            f"shared direction values: {counts['directions']:.0f}):"
        )
        status_lines.append(model.get_spice_model_string(participant_id=0))

        # Concept prevalence: how many of the P*X units hold each concept open. This
        # replaces the old per-term presence row -- term support is now a property of
        # the population-level dictionary, not something each participant owns.
        status_lines.append("-" * terminal_width)
        n_units = model.n_participants * model.n_experiments
        status_lines.append(f"Concept prevalence (number of models={n_units}):")
        for m in model.get_modules():
            gates = model.sindy_concept_gates[m].any(dim=0)  # (P, X, C)
            prevalence = gates.sum(dim=0).sum(dim=0).detach().cpu().numpy()
            alive = model.sindy_concept_support[m].any(dim=-1).detach().cpu().numpy()
            space_filler = " " + " " * (max_len_module - len(m)) if max_len_module > len(m) else " "
            entries = ", ".join(str(int(v)) for v, a in zip(prevalence, alive) if a and v > 0)
            status_lines.append(m + ":" + space_filler + (entries or "(no live concepts)"))

        # What each live concept is made of
        status_lines.append("-" * terminal_width)
        status_lines.append("Concept supports:")
        for m in model.get_modules():
            terms = model.sindy_candidate_terms[m]
            support = model.sindy_concept_support[m].detach().cpu().numpy()
            live = model.sindy_concept_gates[m].any(dim=0).any(dim=0).any(dim=0).detach().cpu().numpy()
            for index_concept in range(support.shape[0]):
                if not live[index_concept] or not support[index_concept].any():
                    continue
                owned = [terms[i] for i in range(len(terms)) if support[index_concept, i]]
                status_lines.append(f"  {m}[{index_concept}]: " + " | ".join(owned))
    status_lines.append("=" * terminal_width)
    
    # Convergence messages
    if converged:
        status_lines.append('Model converged!')
    elif finished:
        status_lines.append('Maximum number of training epochs reached.')
        if not converged:
            status_lines.append('Model did not converge yet.')
    
    msg = "\n".join(status_lines)
    current_line_count = len(status_lines)
    
    # Clear and reprint
    if not keep_log and n_calls > 1:
        if is_notebook:
            from IPython.display import clear_output
            clear_output(wait=True)
        else:
            os.system('clear' if os.name == 'posix' else 'cls')
    print(msg, flush=True)
    
    return current_line_count
