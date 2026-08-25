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


# One min/mean/max block: three 6-wide numbers, single-space separated.
_RANGE_WIDTH = 20
_RANGE_HEADER = f"{'min':>6} {'mean':>6} {'max':>6}"


def _format_value_range(values: torch.Tensor) -> str:
    """min/mean/max of a factor's in-use entries, as one fixed-width block.

    Both tracked quantities are non-negative -- Z by constraint, V by taking |V| -- so
    no sign column is reserved.
    """
    if values.numel() == 0:
        return f"{'-':^{_RANGE_WIDTH}}"
    return f"{values.min().item():>6.3f} {values.mean().item():>6.3f} {values.max().item():>6.3f}"


def _format_mean_std(values: torch.Tensor) -> str:
    """`mean+/-std` of a count, in a fixed 10-wide field."""
    values = values.float()
    spread = values.std() if values.numel() > 1 else torch.zeros_like(values).sum()
    return f"{values.mean().item():>4.1f}+/-{spread.item():<3.1f}"


# Width of one prevalence entry: a right-aligned integer percent plus its separator.
_PREVALENCE_WIDTH = 4


def _format_prevalence(fractions: torch.Tensor, budget: int) -> str:
    """Integer percent per entry, in index order.

    Deliberately *not* sorted. Index position is the identity here -- concept c in this
    epoch is concept c in the next one, and term t is a named library term -- so sorting
    by prevalence would make a decaying concept slide along the strip and destroy the
    one thing the display is for: watching a specific concept or term fade out.

    Percentages are truncated, not rounded, so 100 means literally all: a concept every
    participant but one holds reads 99, never 100.
    """
    entries = [f"{int(fraction * 100):>3}" for fraction in fractions.tolist()]
    if len(entries) * _PREVALENCE_WIDTH - 1 > budget:
        keep = max(1, (budget - len(f" +{len(entries)}")) // _PREVALENCE_WIDTH)
        return " ".join(entries[:keep]) + f" +{len(entries) - keep}"
    return " ".join(entries)


def _spice_parameter_postfix(model: BaseModel) -> dict:
    """`free_participant`/`shared` fields for a tqdm postfix.

    One count_spice_parameters() call, not three: it loops every module and reduces over
    the gate/support masks, so calling it per postfix field forces three device syncs per
    epoch for the same numbers.
    """
    counts = model.count_spice_parameters()
    loadings = counts['loadings']
    # std() over a single unit is nan (and warns); a one-participant fit has no spread.
    spread = loadings.std() if loadings.numel() > 1 else torch.zeros_like(loadings).sum()
    return {
        'free_participant': f"{loadings.mean():.2f}+/-{spread:.2f}",
        'shared': f"{counts['directions']:.0f}",
    }


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

        # Widths of every fixed column, so the two group headers can be centred over
        # their own blocks instead of being eyeballed into place.
        count_width = 10
        n_concepts_max = max(model.sindy_concept_support[m].shape[0] for m in model.get_modules())
        n_terms_max = max(model.sindy_concept_support[m].shape[1] for m in model.get_modules())

        fixed = name_width + 2 * (2 + count_width + 2 + _RANGE_WIDTH + 2)
        strip_budget = max(2 * _PREVALENCE_WIDTH * 2, terminal_width - fixed)
        # Split the remaining width in proportion to what each strip has to render.
        z_full = n_concepts_max * _PREVALENCE_WIDTH - 1
        v_full = n_terms_max * _PREVALENCE_WIDTH - 1
        z_budget = max(2 * _PREVALENCE_WIDTH, min(z_full, round(strip_budget * z_full /
                                                                max(z_full + v_full, 1))))
        v_budget = max(2 * _PREVALENCE_WIDTH, min(v_full, strip_budget - z_budget))

        z_block = count_width + 2 + _RANGE_WIDTH + 2 + z_budget
        v_block = count_width + 2 + _RANGE_WIDTH + 2 + v_budget
        status_lines.append(
            f"{'Modules':<{name_width}}  {'Loadings Z':<{z_block}}  {'Concepts V':<{v_block}}"
        )
        status_lines.append(
            f"{'':<{name_width}}  "
            f"{'c/p':^{count_width}}  {_RANGE_HEADER}  {'prevalence/participant'[:z_budget]:<{z_budget}}  "
            f"{'t/c':^{count_width}}  {_RANGE_HEADER}  {'prevalence/concept'[:v_budget]:<{v_budget}}"
        )

        for module in model.get_modules():
            gates = model.sindy_concept_gates[module]          # (E, P, X, C)
            # Same mask the model itself composes coefficients through (and the same one
            # count_spice_parameters counts): theory-excluded columns are not part of a
            # concept's support even when the learned support bit is still set.
            support = (model.sindy_concept_support[module]
                       & model.sindy_term_prior_mask[module].unsqueeze(0))   # (C, T)
            held = gates.any(dim=0)                            # (P, X, C)
            live = held.any(dim=0).any(dim=0) & support.any(dim=-1)   # (C,)

            n_live = int(live.sum())
            n_total = support.shape[0]
            n_units = max(model.n_participants * model.n_experiments, 1)

            if n_live == 0:
                empty = f"{'-':^{count_width}}  {'-':^{_RANGE_WIDTH}}"
                status_lines.append(
                    f"{module:<{name_width}}  {empty}  {_format_prevalence(torch.zeros(n_total), z_budget):<{z_budget}}  "
                    f"{empty}  {_format_prevalence(torch.zeros(support.shape[1]), v_budget):<{v_budget}}"
                )
                continue

            # Values of the two factors, restricted to what is actually in use: Z over
            # the gated loadings of live concepts, V over the supported terms of live
            # concepts. Unused entries are exact zeros and would only drag the statistics
            # toward zero as pruning proceeds, i.e. track sparsity instead of magnitude.
            # V is tracked as |V|: rows are unit-norm with arbitrary sign per coordinate,
            # so the signed mean sits at ~0 regardless of how the direction is moving,
            # while the mean magnitude tracks how concentrated the row is.
            loadings = model.effective_loadings(module).detach()[..., live]  # (E, P, X, C_live)
            loading_values = loadings[gates[..., live]]
            directions = model.effective_directions(module).detach()[live]   # (C_live, T)
            direction_values = directions[support[live]].abs()

            # Two different questions, one per factor: how many concepts a participant
            # holds / how widely each concept is held, versus how many terms a concept
            # spans / how widely each term is used across the dictionary.
            concepts_per_unit = held[..., live].sum(dim=-1).float()          # (P, X)
            # Over all C, not just the live ones: a retired concept leaves a '.' in place
            # rather than shifting every later concept one position along the strip.
            concept_prevalence = torch.where(
                live, held.sum(dim=0).sum(dim=0).float() / n_units, torch.zeros(()))
            terms_per_concept = support[live].sum(dim=-1).float()            # (C_live,)
            term_prevalence = support[live].sum(dim=0).float() / n_live      # (T,)

            status_lines.append(
                f"{module:<{name_width}}  {_format_mean_std(concepts_per_unit)}  {_format_value_range(loading_values)}  "
                f"{_format_prevalence(concept_prevalence, z_budget):<{z_budget}}  "
                f"{_format_mean_std(terms_per_concept)}  {_format_value_range(direction_values)}  "
                f"{_format_prevalence(term_prevalence, v_budget):<{v_budget}}"
            )

        status_lines.append(
            "prevalence  truncated %, in index order   |   prevalence/participant: each concept "
            "across participants   prevalence/concept: each term across concepts"
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
