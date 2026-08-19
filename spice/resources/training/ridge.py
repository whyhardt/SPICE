"""Closed-form ridge initialization, projected into the concept factorization."""

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
from .reporting import _check_cuda_oom
from .trajectories import _vectorize_state


def _ridge_solve_sindy(
    model: BaseModel,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    alpha: float = None,
) -> bool:
    """
    Lightweight ridge solve: snap SINDy coefficients to their MSE-optimal
    values without touching the optimizer state.

    Runs a frozen RNN forward pass over flattened (trial, session) pairs, chunked with
    automatic OOM backoff (mirrors _run_shooting_eval_batched) -- a single-shot pass over the
    whole flattened dataset can OOM outright on models with large per-sample state (e.g.
    SpiceDDM's evidence_pdf grid), even though the batched SGD loop elsewhere handles the same
    dataset fine. Inside each call_module(), sindy_ridge_accumulate() scatter-adds this chunk's
    contribution to each module's per-participant normal equations; sindy_ridge_finalize()
    solves them once all chunks have been accumulated.

    Returns:
        True if the ridge solve succeeded for all modules, False otherwise.
    """
    was_training = model.training
    prev_use_sindy = model.use_sindy
    prev_ridge_mode = model.ridge_mode

    prev_sindy_alpha = model.sindy_alpha
    if alpha is not None:
        model.sindy_alpha = alpha

    model.eval(use_sindy=False)
    input_state_buffer, _, xs_flat, _ = _vectorize_state(model, xs_train, ys_train)

    flat_total = xs_flat.shape[1]
    chunk_size = flat_total

    # torch.compile's graph for each RNN submodule is specialized to the batch shape(s) it was
    # first traced with (the single full-flat_total shape from the pre-chunking code path).
    # Re-tracing it against a second, smaller chunk shape here (from OOM backoff, or an
    # unevenly-divisible final chunk) has been observed to hit a Triton kernel launch bug
    # ("invalid argument"). Ridge solves are infrequent and already run under no_grad, so
    # correctness matters far more than speed here -- run the uncompiled path instead.
    prev_compile_flags = {name: m._compile for name, m in model.submodules_rnn.items()}
    for rnn_module in model.submodules_rnn.values():
        rnn_module._compile = False

    with torch.no_grad():
        model.ridge_mode = True
        model.train(use_sindy=True)
        for rnn_module in model.submodules_rnn.values():
            rnn_module.eval()

        while True:
            try:
                model.reset_ridge_accumulators()
                for start in range(0, flat_total, chunk_size):
                    end = min(start + chunk_size, flat_total)
                    chunk_state = {s: t[:, :, start:end].clone() for s, t in input_state_buffer.items()}
                    model(xs_flat[:, start:end].to(model.device), chunk_state)
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if _check_cuda_oom(e):
                    raise
                if chunk_size <= 1:
                    raise RuntimeError(f"Automatic batch size probing was unsuccessful for the SINDy ridge solve. Current chunk size is {chunk_size} but could still not be started. Please try again with a smaller ensemble size (current: {model.ensemble_size}).")
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)

        success = all(
            model.sindy_ridge_finalize(module_name)
            for module_name in model.submodules_rnn.keys()
        )

    for name, rnn_module in model.submodules_rnn.items():
        rnn_module._compile = prev_compile_flags[name]

    model.ridge_mode = prev_ridge_mode
    if was_training:
        model.train(use_sindy=prev_use_sindy)
    else:
        model.eval(use_sindy=prev_use_sindy)

    model.sindy_alpha = prev_sindy_alpha

    return success
