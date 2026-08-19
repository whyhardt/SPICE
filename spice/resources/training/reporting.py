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
    
    # Concept-structure summary. Deliberately one line per module rather than a dump of
    # equations: during training what you need to track is whether structure is
    # collapsing (concepts retiring), whether concepts are staying multi-term, and how
    # much per-participant freedom is left -- not the algebra of any one participant.
    if sindy_weight > 0:
        status_lines.append("-" * terminal_width)
        counts = model.count_spice_parameters()
        loadings_per_participant = counts['loadings']
        status_lines.append(
            f"SPICE concepts   free/participant: {loadings_per_participant.mean():.1f}"
            f" +/- {loadings_per_participant.std():.1f}"
            f"   shared: {counts['directions']:.0f}"
        )

        name_width = max(len(module) for module in model.get_modules())
        name_width = max(name_width, 6)
        status_lines.append(
            f"{'module':<{name_width}}  {'live':>7}  {'terms/c':>7}  {'held/p':>6}  prevalence %"
        )

        for module in model.get_modules():
            gates = model.sindy_concept_gates[module]          # (E, P, X, C)
            support = model.sindy_concept_support[module]      # (C, T)
            held = gates.any(dim=0)                            # (P, X, C)
            live = held.any(dim=0).any(dim=0) & support.any(dim=-1)   # (C,)

            n_live = int(live.sum())
            n_total = support.shape[0]
            n_units = model.n_participants * model.n_experiments

            if n_live == 0:
                status_lines.append(f"{module:<{name_width}}  {0:>3}/{n_total:<3}  {'-':>7}  {'-':>6}  (no live concepts)")
                continue

            terms_per_concept = support[live].sum(dim=-1).float().mean().item()
            held_per_unit = held.sum(dim=-1).float().mean().item()

            prevalence = (held.sum(dim=0).sum(dim=0)[live].float() / max(n_units, 1) * 100)
            prevalence = prevalence.sort(descending=True).values.tolist()

            # Keep the prevalence list inside the terminal, however many concepts survive
            budget = max(20, terminal_width - name_width - 30)
            rendered, shown = "", 0
            for value in prevalence:
                candidate = (rendered + " " if rendered else "") + f"{value:.0f}"
                if len(candidate) > budget:
                    break
                rendered, shown = candidate, shown + 1
            if shown < len(prevalence):
                rendered += f" +{len(prevalence) - shown}"

            status_lines.append(
                f"{module:<{name_width}}  {n_live:>3}/{n_total:<3}  "
                f"{terms_per_concept:>7.1f}  {held_per_unit:>6.1f}  {rendered}"
            )


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
