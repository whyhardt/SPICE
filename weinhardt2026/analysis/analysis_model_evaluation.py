import os
import math
import sys
from typing import Callable, Optional

import argparse
import importlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import binomtest

# standard methods and classes used for every model evaluation
from spice import SpiceEstimator, csv_to_dataset, SpiceDataset


def get_choice_probs(logits: torch.Tensor) -> torch.Tensor:
    # softmax normalization
    return torch.softmax(logits, dim=-1)


def log_likelihood(data: torch.tensor, probs: torch.tensor, **kwargs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1 
    
    # Sum over all data points
    # return torch.sum(torch.sum(data * torch.log(probs), axis=-1), axis=axis) / normalization
    # Ensure probabilities are within a valid range to prevent log(0)
    epsilon = 1e-9
    probs = torch.clip(probs, epsilon, 1 - epsilon)
    
    # Calculate log-likelihood for each observation
    log_likelihoods = data * torch.log(probs)
    
    # Sum log-likelihoods over all observations
    return log_likelihoods


def bayesian_information_criterion(data: torch.Tensor, probs: torch.Tensor, n_parameters: int, nll: torch.Tensor = None, **kwargs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if nll is None:
        nll = log_likelihood(data=data, probs=probs)
    
    # n_samples = (data[:, 0] != -1).sum()
    n_samples = (~torch.isnan(data[..., 0])).sum()
    return 2 * nll + n_parameters * torch.log(n_samples)


def akaike_information_criterion(data: torch.Tensor, probs: torch.Tensor, n_parameters: int, nll: torch.Tensor = None, **kwargs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if nll is None:
        nll = log_likelihood(data=data, probs=probs)
    
    return 2 * nll + 2 * n_parameters


def get_scores(probs: torch.Tensor, targets: torch.Tensor, n_parameters: int) -> tuple[tuple[float, float, float], torch.Tensor]:
    nll = -log_likelihood(data=targets, probs=probs)
    nll_sum = torch.nansum(nll)

    bic = bayesian_information_criterion(data=targets, probs=probs, n_parameters=n_parameters, nll=nll_sum)
    aic = akaike_information_criterion(data=targets, probs=probs, n_parameters=n_parameters, nll=nll_sum)


    return (nll_sum, aic, bic), nll


def get_participant_experiment_groups(dataset: SpiceDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each session to a (participant, experiment) group.

    Sessions belonging to the same participant/experiment share the same
    fitted parameters (per-subject benchmark parameters, or SPICE's
    per-participant SINDy coefficients), so BIC/AIC must be computed per
    group and then averaged across groups (see ``grouped_information_criteria``).

    Returns:
        unique_pairs: (G, 2) tensor of unique (participant_id, experiment_id) pairs.
        group_index: (B,) tensor mapping each session to its row in unique_pairs.
    """
    participant_ids = dataset.xs[:, 0, 0, -1].long().cpu()
    experiment_ids = dataset.xs[:, 0, 0, -2].long().cpu()
    pair_ids = torch.stack([participant_ids, experiment_ids], dim=1)
    unique_pairs, group_index = torch.unique(pair_ids, dim=0, return_inverse=True)
    return unique_pairs, group_index


def grouped_information_criteria(
    nll_per_session: torch.Tensor,
    n_trials_per_session: torch.Tensor,
    group_index: torch.Tensor,
    n_groups: int,
    n_parameters_per_group: torch.Tensor,
    n_actions_baseline: int,
) -> dict:
    """Compute BIC/AIC/ΔBIC-per-trial per (participant, experiment) group, then
    summarize as mean ± std across groups.

    Pooling NLL across the whole dataset while scaling the parameter count
    with the number of subjects (as is correct for per-subject-fit models,
    e.g. individually-fit benchmark parameters or SPICE's per-participant
    SINDy coefficients) makes such models lose against a fixed-size
    shared-weight model purely as dataset size grows, independent of actual
    fit quality. Computing BIC per group with that group's own trial count
    and own parameter count -- then averaging across groups -- removes that
    scaling artifact.
    """
    # All grouping/reduction happens on CPU: inputs may come from a mix of
    # devices (e.g. coefficients read off a CUDA model vs. CPU-side NLLs).
    nll_per_session = nll_per_session.cpu()
    n_trials_per_session = n_trials_per_session.float().cpu()
    group_index = group_index.cpu()
    n_parameters_per_group = n_parameters_per_group.float().cpu()

    nll_group = torch.zeros(n_groups, dtype=nll_per_session.dtype).scatter_add_(0, group_index, nll_per_session)
    n_trials_group = torch.zeros(n_groups).scatter_add_(0, group_index, n_trials_per_session)

    valid = n_trials_group > 0
    nll_group = nll_group[valid]
    n_trials_group = n_trials_group[valid]
    k_group = n_parameters_per_group[valid]

    bic_group = 2 * nll_group + k_group * torch.log(n_trials_group)
    aic_group = 2 * nll_group + 2 * k_group

    nll_random_group = n_trials_group * math.log(n_actions_baseline)
    bic_random_group = 2 * nll_random_group
    delta_bic_per_trial_group = (bic_random_group - bic_group) / n_trials_group

    def _mean_std(x: torch.Tensor) -> tuple[float, float]:
        return x.mean().item(), (x.std().item() if x.numel() > 1 else 0.0)

    bic_mean, bic_std = _mean_std(bic_group)
    aic_mean, aic_std = _mean_std(aic_group)
    dbic_mean, dbic_std = _mean_std(delta_bic_per_trial_group)

    return {
        'nll_total': nll_group.sum().item(),
        'bic_mean': bic_mean, 'bic_std': bic_std,
        'aic_mean': aic_mean, 'aic_std': aic_std,
        'delta_bic_per_trial_mean': dbic_mean, 'delta_bic_per_trial_std': dbic_std,
    }


def joint_information_criteria(nll_total: float, n_trials: float, n_parameters_total: float) -> tuple[float, float]:
    """AIC and BIC of the whole dataset as one model.

    The NLL is summed over all trials of all participants and every participant's own parameters are part
    of the joint model (``n_parameters_total``: per-participant parameters summed over participants, shared
    parameters counted once):
        AIC = 2 NLL + 2 K,    BIC = 2 NLL + K log N.
    """
    aic = 2 * nll_total + 2 * n_parameters_total
    bic = 2 * nll_total + n_parameters_total * math.log(n_trials)
    return aic, bic


def get_participant_index(dataset: SpiceDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each session to its participant (all sessions and experiments of a participant are pooled)."""
    participant_ids = dataset.xs[:, 0, 0, -1].long().cpu()
    return torch.unique(participant_ids, return_inverse=True)


def participant_win_counts(
    score_per_participant: dict[str, np.ndarray],
    reference: str = 'SPICE-EQ',
    compared: tuple[str, ...] = ('Benchmark',),
    higher_is_better: bool = True,
) -> dict[str, dict]:
    """Per-participant comparison of ``reference`` against the ``compared`` models.

    By default only the interpretable benchmark is compared: against black-box models (GRU, SPICE-RNN)
    the count says little about the recovered mechanism.

    Counts the participants for whom ``reference`` scores better, tests the count with a two-sided sign test
    (ties dropped) and reports the median per-participant difference, signed so that positive favours
    ``reference``. Participants without a score (NaN) are skipped.

    Args:
        score_per_participant: model name -> (P,) score per participant (e.g. trial likelihood or MSE).
        reference: model the others are compared against.
        higher_is_better: True for likelihoods, False for errors.

    Returns:
        model name -> {'wins', 'losses', 'n', 'p', 'median_delta'} for every compared model that was evaluated.
    """
    if reference not in score_per_participant:
        return {}
    results = {}
    for name, score in score_per_participant.items():
        if name == reference or name not in compared:
            continue
        delta = score_per_participant[reference] - score
        if not higher_is_better:
            delta = -delta
        delta = delta[~np.isnan(delta)]
        wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
        results[name] = {
            'wins': wins,
            'losses': losses,
            'n': len(delta),
            'p': binomtest(wins, wins + losses, 0.5).pvalue if wins + losses > 0 else float('nan'),
            'median_delta': float(np.median(delta)) if len(delta) > 0 else float('nan'),
        }
    return results


def win_count_columns(comparison: Optional[dict], reference: str, delta_label: str, prefix: str = '') -> dict:
    """Table columns for one model's row from ``participant_win_counts`` (NaN for models not compared)."""
    if comparison is None:
        return {f'{prefix}{reference} better': float('nan'), f'{prefix}Participants': float('nan'),
                f'{prefix}Sign test p': float('nan'), f'{prefix}Median Δ {delta_label}': float('nan')}
    return {f'{prefix}{reference} better': comparison['wins'], f'{prefix}Participants': comparison['n'],
            f'{prefix}Sign test p': comparison['p'], f'{prefix}Median Δ {delta_label}': comparison['median_delta']}


def format_number(value: float) -> str:
    """Integers without decimals (e.g. parameter counts), every other number rounded to 3 decimals."""
    if not np.isfinite(value):
        return str(value)
    return f'{value:.0f}' if float(value).is_integer() else f'{value:.3f}'


def format_table(df: pd.DataFrame) -> str:
    """Evaluation table for the terminal, transposed (models as columns, metrics as rows) so every column fits; NaN as '-'."""
    return df.T.to_string(float_format=format_number, na_rep='-')


# How the members of a SPICE ensemble are turned into one reported score (see analysis_model_evaluation)
SPICE_SELECTIONS = ('averaged-model', 'averaged-logit', 'best-on-train')

# Information criteria are only reported for the interpretable models
INFORMATION_CRITERIA_MODELS = ('Benchmark', 'SPICE-EQ')


def _prepare_split(dataset: SpiceDataset, trial_filter: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> dict:
    """Masked targets, considered trials and session groupings of one data split."""
    valid_mask = ~torch.isnan(dataset.xs[:, :, 0, 0])
    if trial_filter is not None:
        valid_mask = valid_mask & trial_filter(dataset.ys)
    valid_mask = valid_mask.cpu()

    # Masked targets: excluded trials set to NaN so nansum skips them
    targets = dataset.ys.clone().cpu()
    targets[~valid_mask] = float('nan')

    unique_pairs, group_index = get_participant_experiment_groups(dataset)
    participants, participant_index = get_participant_index(dataset)
    n_trials_per_session = valid_mask.sum(dim=1).float()

    return {
        'xs': dataset.xs,
        'targets': targets,
        'n_trials_per_session': n_trials_per_session,
        'unique_pairs': unique_pairs,
        'group_index': group_index,
        'participant_index': participant_index,
        'n_trials_per_participant': torch.zeros(len(participants)).scatter_add_(0, participant_index, n_trials_per_session),
    }


def _score_split(probs: torch.Tensor, split: dict) -> tuple[dict, np.ndarray]:
    """Trial likelihood and NLL of one model on one split.

    Returns:
        scores: metric name -> value.
        trial_lik_per_participant: (P,) trial likelihood per participant (for ``participant_win_counts``).
    """
    nll = -log_likelihood(data=split['targets'], probs=probs)
    nll_per_session = nll.sum(dim=-1).nansum(dim=1)[..., 0]
    n_trials_per_session = split['n_trials_per_session']

    n_trials_per_participant = split['n_trials_per_participant']
    nll_per_participant = torch.zeros(len(n_trials_per_participant), dtype=nll_per_session.dtype).scatter_add_(0, split['participant_index'], nll_per_session)
    trial_lik_per_participant = torch.where(
        n_trials_per_participant > 0, torch.exp(-nll_per_participant / n_trials_per_participant.clamp(min=1)), float('nan')).numpy()

    trial_lik_per_session = np.exp(-nll_per_session.numpy() / n_trials_per_session.numpy())

    scores = {
        'Trial Lik.': torch.exp(-nll_per_session.sum() / n_trials_per_session.sum()).item(),
        'Trial Lik. (std)': np.nanstd(trial_lik_per_session),   # sessions without considered trials are NaN
        'NLL': nll_per_session.sum().item(),
    }
    return scores, trial_lik_per_participant


# Parameter counts of a model on a split: unique (participant, experiment) pairs (G, 2) ->
# (parameter count per pair (G,), parameters of the joint model over all pairs)
ParameterCount = Callable[[torch.Tensor], tuple[torch.Tensor, float]]


def _shared_parameters(n_parameters: float) -> ParameterCount:
    """A model whose parameters all participants share: counted once in the joint model."""
    return lambda unique_pairs: (torch.full((unique_pairs.shape[0],), float(n_parameters)), float(n_parameters))


def _per_participant_parameters(n_parameters: float) -> ParameterCount:
    """A model with its own ``n_parameters`` per participant: summed over the split's participants."""
    return lambda unique_pairs: (
        torch.full((unique_pairs.shape[0],), float(n_parameters)),
        float(n_parameters) * len(torch.unique(unique_pairs[:, 0])),
    )


def _sindy_parameters(spice_model: SpiceEstimator, counts: torch.Tensor) -> ParameterCount:
    """SPICE-EQ: active coefficients of each (participant, experiment), summed over the distinct coefficient sets.

    Dataset ids are mapped onto the model's parameters (a group model maps every participant to 0,
    so its coefficients are counted once).
    """
    def count(unique_pairs: torch.Tensor) -> tuple[torch.Tensor, float]:
        participants = spice_model.model.map_participant_ids(unique_pairs[:, 0])
        experiments = spice_model.model.map_experiment_ids(unique_pairs[:, 1])
        model_pairs = torch.unique(torch.stack([participants, experiments], dim=1), dim=0)
        return counts[participants, experiments], counts[model_pairs[:, 0], model_pairs[:, 1]].sum().item()
    return count


def _spice_candidates(spice_model: SpiceEstimator, splits: dict, spice_selection: str) -> dict:
    """SPICE-RNN and SPICE-EQ candidates: one ensemble-averaged-logit candidate, or one per ensemble member.

    Returns:
        model name -> list of (split name -> choice probs, ParameterCount).
    """
    spice_rnn_parameters = 0
    for module in spice_model.get_modules():
        spice_rnn_parameters += sum(p.numel() for p in spice_model.model.submodules_rnn[module].parameters())
    spice_rnn_parameters += spice_model.model.embedding_size

    logits = {}   # model name -> split name -> (E, B, T, W, A)
    for name, use_sindy in (('SPICE-EQ', True), ('SPICE-RNN', False)):
        print(f"Computing choice probabilities with {name} model...")
        spice_model.eval(use_sindy=use_sindy)
        logits[name] = {}
        for split_name, split in splits.items():
            spice_model.model.init_state(batch_size=split['xs'].shape[0])
            logits[name][split_name] = spice_model(split['xs'].to(spice_model.device))[0].detach().cpu()
    spice_model.use_sindy(True)

    if spice_selection == 'averaged-logit':
        n_parameters = {
            'SPICE-RNN': _shared_parameters(spice_rnn_parameters),
            'SPICE-EQ': _sindy_parameters(spice_model, spice_model.count_sindy_coefficients().float().cpu()),
        }
        return {
            name: [({split_name: get_choice_probs(split_logits.mean(dim=0)) for split_name, split_logits in logits[name].items()}, n_parameters[name])]
            for name in ('SPICE-RNN', 'SPICE-EQ')
        }

    coefficient_counts = spice_model.count_sindy_coefficients(aggregate=False).float().cpu()   # (E, P, X)
    members = range(coefficient_counts.shape[0])
    n_parameters = {
        'SPICE-RNN': [_shared_parameters(spice_rnn_parameters) for _ in members],
        'SPICE-EQ': [_sindy_parameters(spice_model, coefficient_counts[member]) for member in members],
    }
    return {
        name: [
            ({split_name: get_choice_probs(split_logits[member]) for split_name, split_logits in logits[name].items()}, n_parameters[name][member])
            for member in members
        ]
        for name in ('SPICE-RNN', 'SPICE-EQ')
    }


def _evaluate_candidate(probs: dict, n_parameters: ParameterCount, splits: dict) -> dict:
    """Scores of one candidate model on every split, plus its parameter counts and AIC/BIC on the training split."""
    n_parameters_per_group, n_parameters_total = n_parameters(splits['train']['unique_pairs'])
    evaluation = {split_name: _score_split(probs=probs[split_name], split=split) for split_name, split in splits.items()}
    aic, bic = joint_information_criteria(
        nll_total=evaluation['train'][0]['NLL'],
        n_trials=splits['train']['n_trials_per_session'].sum().item(),
        n_parameters_total=n_parameters_total,
    )
    evaluation.update({
        'n_parameters': n_parameters_per_group.float().cpu(),
        'n_parameters_total': n_parameters_total,
        'AIC': aic,
        'BIC': bic,
    })
    return evaluation


@torch.no_grad()
def analysis_model_evaluation(
    dataset_train: SpiceDataset,
    dataset_test: SpiceDataset,
    spice_model: SpiceEstimator = None,
    benchmark_model: torch.nn.Module = None,
    gru_model: torch.nn.Module = None,
    spice_selection: str = 'best-on-train',
    verbose: bool = False,
    output_dir: Optional[str] = None,
    trial_filter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    benchmark_group_model: bool = False,
    ):
    """Evaluate the models on the training and the test split in one table.

    Training split: trial likelihood, NLL, AIC and BIC. Test split: trial likelihood and NLL.
    ``n_parameters`` (± std) is the parameter count per (participant, experiment) group of the training
    split; ``n_parameters (total)`` is the parameter count of the joint model that enters AIC/BIC.

    Args:
        spice_selection: How the members of the SPICE ensemble enter the reported scores (applied to
            SPICE-RNN and SPICE-EQ separately):
              'averaged-model': every member is scored on its own (its own logits and coefficient count);
                  the members' scores are averaged.
              'averaged-logit': the members' logits are averaged at every time step and the averaged
                  model is scored (SPICE-EQ counts the coefficients active in any member).
              'best-on-train': every member is scored on its own; the member with the highest training
                  likelihood is reported, on the training and on the test split.
        benchmark_group_model: Set True when the benchmark is a group model (one parameter set shared
            by all participants): its parameters are then counted once in the joint model instead of
            once per participant. SPICE's parameter sharing is read off the model (a SPICE group model,
            n_participants=1, counts its coefficients once).
        trial_filter: Optional function ``(ys: Tensor) -> BoolTensor (B, T)``
            that returns True for trials to **include** in the evaluation (applied to both splits).
            When None (default), all non-NaN trials are included.

    AIC/BIC treat the whole training split as one model (``joint_information_criteria``): the NLL is
    summed over all N trials of all participants, and K counts every participant's own parameters
    (per-participant parameters summed over participants, shared parameters once):
    AIC = 2 NLL + 2 K, BIC = 2 NLL + K log N. They are reported only for the interpretable models
    (Benchmark, SPICE-EQ).

    **Information criteria belong on training data only.** The ``k log n`` term
    exists to stand in for the generalization gap; held-out likelihood already
    *is* that gap, so scoring a test split with BIC charges for parsimony twice
    and systematically favours whichever model is sparsest. On dezfouli2019 this
    inverted a ranking: a 5.6-parameter model beat a 10.7-parameter one on
    held-out BIC while losing to it on both the in-sample criterion and raw
    held-out likelihood. Select on training BIC; report held-out NLL.

    When SPICE and the benchmark are evaluated, the benchmark's row also reports, per split, for how many
    participants SPICE-EQ reaches the higher trial likelihood (``SPICE-EQ better`` out of ``Participants``),
    a two-sided sign test on that count, and the median per-participant likelihood difference (SPICE-EQ minus
    the benchmark); see ``participant_win_counts``. With 'averaged-model' each participant's likelihood is
    averaged over the members before counting. To compare across seeds, average each participant's
    likelihood over seeds before counting, so that seed noise does not enter the count.
    """
    if spice_selection not in SPICE_SELECTIONS:
        raise ValueError(f"spice_selection must be one of {SPICE_SELECTIONS}, got '{spice_selection}'.")

    splits = {
        'train': _prepare_split(dataset_train, trial_filter),
        'test': _prepare_split(dataset_test, trial_filter),
    }

    # ------------------------------------------------------------
    # Compute choice probs + parameter counts
    # model name -> list of candidates (split name -> choice probs, ParameterCount)
    # ------------------------------------------------------------
    candidates = {}

    if benchmark_model is not None:
        print("Computing choice probabilities with benchmark model...")
        benchmark_parameters = benchmark_model.count_parameters() if hasattr(benchmark_model, 'count_parameters') else len([p for p in benchmark_model.parameters()])
        probs = {split_name: get_choice_probs(benchmark_model(split['xs'])[0]).detach().cpu() for split_name, split in splits.items()}
        count = _shared_parameters if benchmark_group_model else _per_participant_parameters
        candidates['Benchmark'] = [(probs, count(benchmark_parameters))]

    if gru_model is not None:
        print("Computing choice probabilities with GRU model...")
        gru_model.eval()
        gru_parameters = sum(p.numel() for p in gru_model.parameters())
        probs = {split_name: get_choice_probs(gru_model(split['xs'])[0]).detach().cpu() for split_name, split in splits.items()}
        candidates['GRU'] = [(probs, _shared_parameters(gru_parameters))]

    if spice_model is not None:
        candidates.update(_spice_candidates(spice_model, splits, spice_selection))

    # ------------------------------------------------------------
    # Evaluate each model; reduce SPICE's members to one row
    # ------------------------------------------------------------
    rows = {}
    trial_lik_per_participant = {split_name: {} for split_name in splits}
    for name, model_candidates in candidates.items():
        evaluations = [_evaluate_candidate(probs, n_parameters, splits) for probs, n_parameters in model_candidates]

        if spice_selection == 'best-on-train' and len(evaluations) > 1:
            best = int(np.argmax([evaluation['train'][0]['Trial Lik.'] for evaluation in evaluations]))
            print(f"{name}: best-on-train ensemble member {best}")
            evaluations = [evaluations[best]]

        n_parameters = torch.cat([evaluation['n_parameters'] for evaluation in evaluations])
        rows[name] = {
            'n_parameters': n_parameters.mean().item(),
            'n_parameters (std)': n_parameters.std().item() if n_parameters.numel() > 1 else 0.0,
            'n_parameters (total)': np.mean([evaluation['n_parameters_total'] for evaluation in evaluations]),
        }
        for split_name, prefix in (('train', 'Train '), ('test', 'Test ')):
            scores = {metric: np.mean([evaluation[split_name][0][metric] for evaluation in evaluations]) for metric in evaluations[0][split_name][0]}
            rows[name].update({f'{prefix}{metric}': scores[metric] for metric in ('Trial Lik.', 'Trial Lik. (std)', 'NLL')})
            if split_name == 'train':
                for metric in ('AIC', 'BIC'):
                    rows[name][metric] = np.mean([evaluation[metric] for evaluation in evaluations]) if name in INFORMATION_CRITERIA_MODELS else float('nan')
            trial_lik_per_participant[split_name][name] = np.nanmean(
                np.stack([evaluation[split_name][1] for evaluation in evaluations]), axis=0)

    # ------------------------------------------------------------
    # Per-participant comparison against SPICE-EQ
    # ------------------------------------------------------------
    for split_name, prefix in (('train', 'Train '), ('test', 'Test ')):
        comparisons = participant_win_counts(trial_lik_per_participant[split_name], reference='SPICE-EQ', higher_is_better=True)
        if comparisons:
            for name in rows:
                rows[name].update(win_count_columns(comparisons.get(name), reference='SPICE-EQ', delta_label='lik', prefix=prefix))

    # ------------------------------------------------------------
    # Printing model performance table
    # ------------------------------------------------------------

    df = pd.DataFrame(rows).T.reindex(['Benchmark', 'GRU', 'SPICE-RNN', 'SPICE-EQ'])

    if verbose:
        print(format_table(df))

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'model_evaluation.csv'))

    return df



# ============================================================================
# MSE-based model evaluation (for continuous action spaces)
# ============================================================================

def _get_predictions(model, xs: torch.Tensor) -> torch.Tensor:
    """Run forward pass and return predictions with shape (E, B, T, W, A).

    SpiceEstimator / BaseModel outputs keep their ensemble axis; GRU / benchmark outputs (B, T, W, A)
    get an ensemble axis of 1.
    """
    if isinstance(model, SpiceEstimator):
        model.model.init_state(batch_size=xs.shape[0])
        xs = xs.to(model.device)
    preds, _ = model(xs)
    if isinstance(preds, tuple):
        preds = preds[0]
    if preds.dim() == 4:
        preds = preds.unsqueeze(0)
    return preds.detach().cpu()


def _compute_mse_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_fn: Callable = None,
) -> dict:
    """Compute MSE, RMSE, MAE, R² from masked predictions and targets.

    Args:
        predictions: (B, T, W, A) raw model outputs.
        targets: (B, T, W, F) ground truth. F may exceed A when a custom
            loss_fn requires extra columns (e.g. clamping metadata).
        valid_mask: (B, T) bool mask for non-padded trials.
        loss_fn: Optional loss function ``(prediction, target) → scalar``.
            When provided, MSE is computed via this function and targets
            may have more features than predictions. When ``None``
            (default), plain MSE is used and targets must match
            predictions in the last dimension.

    Returns:
        Dict with aggregate and per-session metrics.
    """
    n_sessions = predictions.shape[0]
    n_actions = predictions.shape[-1]
    n_target_features = targets.shape[-1]

    # Flatten valid predictions
    mask_pred = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(predictions)
    pred_valid = predictions[mask_pred].reshape(-1, n_actions)

    # Flatten valid targets (may have more features than predictions)
    mask_tgt = valid_mask.unsqueeze(-1).unsqueeze(-1).expand(
        *valid_mask.shape, predictions.shape[2], n_target_features)
    tgt_valid = targets[mask_tgt].reshape(-1, n_target_features)

    if loss_fn is not None:
        mse = loss_fn(pred_valid, tgt_valid).item()
        tgt_actual = tgt_valid[:, :n_actions]
    else:
        tgt_actual = tgt_valid
        mse = ((pred_valid - tgt_actual) ** 2).mean().item()

    # Derived metrics (always from raw prediction errors against actual targets)
    errors = pred_valid - tgt_actual
    rmse = mse ** 0.5
    mae = errors.abs().mean().item()

    ss_res = mse * pred_valid.numel()
    ss_tot = ((tgt_actual - tgt_actual.mean(dim=0)) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # Per-session metrics
    mse_per_session = torch.full((n_sessions,), float('nan'))
    for s in range(n_sessions):
        s_mask = valid_mask[s]
        if s_mask.sum() == 0:
            continue
        s_pred = predictions[s, s_mask, 0]  # (n_valid, A)
        s_tgt = targets[s, s_mask, 0]       # (n_valid, F)
        if loss_fn is not None:
            mse_per_session[s] = loss_fn(s_pred, s_tgt)
        else:
            mse_per_session[s] = ((s_pred - s_tgt) ** 2).mean()

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse_per_session': mse_per_session,
    }


def _benchmark_parameters_mse(benchmark_model: torch.nn.Module) -> float:
    """Parameter count per participant of a continuous benchmark model."""
    if hasattr(benchmark_model, 'count_parameters'):
        return benchmark_model.count_parameters()
    n_params = sum(p.numel() for p in benchmark_model.parameters())
    if getattr(benchmark_model, 'n_participants', 1) > 1:
        # Fall back assumes per-participant nn.Parameters (shape (n_participants,));
        # report the per-participant count, not the total across all participants.
        n_params /= benchmark_model.n_participants
    return n_params


def _spice_candidates_mse(spice_model: SpiceEstimator, datasets: dict, spice_selection: str) -> dict:
    """SPICE-RNN and SPICE-EQ candidates for the MSE evaluation (see ``_spice_candidates``).

    Returns:
        model name -> list of (split name -> (B, T, W, A) predictions, ParameterCount).
    """
    spice_rnn_parameters = 0
    for module in spice_model.get_modules():
        spice_rnn_parameters += sum(p.numel() for p in spice_model.model.submodules_rnn[module].parameters())
    spice_rnn_parameters += spice_model.model.embedding_size

    predictions = {}   # model name -> split name -> (E, B, T, W, A)
    for name, use_sindy in (('SPICE-EQ', True), ('SPICE-RNN', False)):
        print(f"Evaluating {name} model...")
        spice_model.eval(use_sindy=use_sindy)
        predictions[name] = {split_name: _get_predictions(spice_model, dataset.xs) for split_name, dataset in datasets.items()}
    spice_model.use_sindy(True)

    if spice_selection == 'averaged-logit':
        n_parameters = {
            'SPICE-RNN': _shared_parameters(spice_rnn_parameters),
            'SPICE-EQ': _sindy_parameters(spice_model, spice_model.count_sindy_coefficients().float().cpu()),
        }
        return {
            name: [({split_name: split_preds.mean(dim=0) for split_name, split_preds in predictions[name].items()}, n_parameters[name])]
            for name in ('SPICE-RNN', 'SPICE-EQ')
        }

    coefficient_counts = spice_model.count_sindy_coefficients(aggregate=False).float().cpu()   # (E, P, X)
    members = range(coefficient_counts.shape[0])
    n_parameters = {
        'SPICE-RNN': [_shared_parameters(spice_rnn_parameters) for _ in members],
        'SPICE-EQ': [_sindy_parameters(spice_model, coefficient_counts[member]) for member in members],
    }
    return {
        name: [
            ({split_name: split_preds[member] for split_name, split_preds in predictions[name].items()}, n_parameters[name][member])
            for member in members
        ]
        for name in ('SPICE-RNN', 'SPICE-EQ')
    }


def _gaussian_information_criteria(mse: float, n_trials: int, n_parameters_total: float) -> tuple[float, float]:
    """AIC and BIC of the whole dataset as one model with a Gaussian likelihood (MLE variance ``mse`` over all trials)."""
    nll_total = 0.5 * n_trials * (1 + math.log(2 * math.pi) + math.log(max(mse, 1e-30)))
    return joint_information_criteria(nll_total=nll_total, n_trials=n_trials, n_parameters_total=n_parameters_total)


@torch.no_grad()
def analysis_model_evaluation_mse(
    dataset_train: SpiceDataset,
    dataset_test: SpiceDataset,
    spice_model: SpiceEstimator = None,
    benchmark_model: torch.nn.Module = None,
    gru_model: torch.nn.Module = None,
    spice_selection: str = 'best-on-train',
    output_dir: Optional[str] = None,
    verbose: bool = True,
    loss_fn: Callable = None,
    benchmark_group_model: bool = False,
) -> pd.DataFrame:
    """Evaluate models on continuous prediction tasks using MSE-based metrics, on the training and the test split.

    Counterpart to ``analysis_model_evaluation`` for continuous action spaces.
    Training split: MSE, RMSE, MAE, R², AIC and BIC. Test split: MSE, RMSE, MAE and R².
    AIC/BIC treat the whole training split as one model (as in ``analysis_model_evaluation``): a Gaussian
    likelihood with one MLE variance over all trials, and every participant's own parameters counted
    (``n_parameters (total)``). They are only reported for the interpretable models (Benchmark, SPICE-EQ) and only on the
    training split: an information criterion on a test split charges for parsimony twice.

    When SPICE and the benchmark are evaluated, the benchmark's row also reports, per split, for how many
    participants SPICE-EQ reaches the lower MSE (trial-weighted over the participant's sessions), a two-sided
    sign test on that count, and the median per-participant MSE difference (the benchmark minus SPICE-EQ, so
    positive favours SPICE-EQ); see ``participant_win_counts``.

    Args:
        dataset_train: Training dataset (SpiceDataset with continuous targets).
        dataset_test: Test dataset.
        spice_model: Fitted SpiceEstimator (optional).
        benchmark_model: Fitted benchmark nn.Module (optional).
        gru_model: Fitted GRU nn.Module (optional).
        spice_selection: How the members of the SPICE ensemble enter the reported scores (applied to
            SPICE-RNN and SPICE-EQ separately):
              'averaged-model': every member is scored on its own; the members' scores are averaged.
              'averaged-logit': the members' predictions are averaged at every time step and the
                  averaged model is scored (SPICE-EQ counts the coefficients active in any member).
              'best-on-train': every member is scored on its own; the member with the lowest training
                  MSE (highest Gaussian likelihood) is reported, on the training and on the test split.
        output_dir: If provided, save summary CSV and a bar plot of the test metrics to this directory.
        verbose: Print results to console.
        loss_fn: Optional loss function ``(prediction, target) → scalar``.
            When provided, MSE is computed via this function (e.g.
            ``clamped_angular_mse``). Defaults to plain MSE.
        benchmark_group_model: Set True when the benchmark is a group model (one parameter set shared
            by all participants): its parameters are then counted once instead of once per participant.

    Returns:
        DataFrame with n_parameters and the train/test metrics per model.
    """
    if spice_selection not in SPICE_SELECTIONS:
        raise ValueError(f"spice_selection must be one of {SPICE_SELECTIONS}, got '{spice_selection}'.")

    datasets = {'train': dataset_train, 'test': dataset_test}
    valid_masks = {split_name: ~torch.isnan(dataset.xs[:, :, 0, 0]).cpu() for split_name, dataset in datasets.items()}
    targets = {split_name: dataset.ys.cpu() for split_name, dataset in datasets.items()}

    # ------------------------------------------------------------
    # Predictions + parameter counts
    # model name -> list of candidates (split name -> (B, T, W, A) predictions, parameter count)
    # ------------------------------------------------------------
    candidates = {}

    if benchmark_model is not None:
        print("Evaluating benchmark model...")
        benchmark_model.eval()
        preds = {split_name: _get_predictions(benchmark_model, dataset.xs).mean(dim=0) for split_name, dataset in datasets.items()}
        count = _shared_parameters if benchmark_group_model else _per_participant_parameters
        candidates['Benchmark'] = [(preds, count(_benchmark_parameters_mse(benchmark_model)))]

    if gru_model is not None:
        print("Evaluating GRU model...")
        gru_model.eval()
        preds = {split_name: _get_predictions(gru_model, dataset.xs).mean(dim=0) for split_name, dataset in datasets.items()}
        candidates['GRU'] = [(preds, _shared_parameters(sum(p.numel() for p in gru_model.parameters())))]

    if spice_model is not None:
        candidates.update(_spice_candidates_mse(spice_model, datasets, spice_selection))

    n_valid_train = valid_masks['train'].sum().item()
    unique_pairs_train, _ = get_participant_experiment_groups(dataset_train)

    # ------------------------------------------------------------
    # Evaluate each model; reduce SPICE's members to one entry
    # ------------------------------------------------------------
    scalar_metrics = ('mse', 'rmse', 'mae', 'r2')
    models = {}   # model name -> (split name -> metrics, parameter counts and train AIC/BIC)
    for name, model_candidates in candidates.items():
        evaluations = []
        for preds, n_parameters in model_candidates:
            metrics = {split_name: _compute_mse_metrics(preds[split_name], targets[split_name], valid_masks[split_name], loss_fn=loss_fn) for split_name in datasets}
            n_parameters_per_group, n_parameters_total = n_parameters(unique_pairs_train)
            aic, bic = _gaussian_information_criteria(metrics['train']['mse'], n_valid_train, n_parameters_total)
            evaluations.append({'metrics': metrics, 'n_parameters': n_parameters_per_group.float(), 'n_parameters_total': n_parameters_total, 'AIC': aic, 'BIC': bic})

        if spice_selection == 'best-on-train' and len(evaluations) > 1:
            best = int(np.argmin([evaluation['metrics']['train']['mse'] for evaluation in evaluations]))
            print(f"{name}: best-on-train ensemble member {best}")
            evaluations = [evaluations[best]]

        metrics = {
            split_name: {
                **{metric: np.mean([evaluation['metrics'][split_name][metric] for evaluation in evaluations]) for metric in scalar_metrics},
                'mse_per_session': torch.stack([evaluation['metrics'][split_name]['mse_per_session'] for evaluation in evaluations]).nanmean(dim=0),
            }
            for split_name in datasets
        }
        n_parameters = torch.cat([evaluation['n_parameters'] for evaluation in evaluations])
        models[name] = (metrics, {
            'n_parameters': n_parameters.mean().item(),
            'n_parameters (std)': n_parameters.std().item() if n_parameters.numel() > 1 else 0.0,
            'n_parameters (total)': np.mean([evaluation['n_parameters_total'] for evaluation in evaluations]),
            'AIC': np.mean([evaluation['AIC'] for evaluation in evaluations]),
            'BIC': np.mean([evaluation['BIC'] for evaluation in evaluations]),
        })

    # ------------------------------------------------------------
    # Per-participant comparison against SPICE-EQ: trial-weighted mean of the per-session MSE (lower is better)
    # ------------------------------------------------------------
    comparisons = {}
    for split_name, dataset in datasets.items():
        participants, participant_index = get_participant_index(dataset)
        n_trials_per_session = valid_masks[split_name].sum(dim=1).float()
        n_trials_per_participant = torch.zeros(len(participants)).scatter_add_(0, participant_index, n_trials_per_session)
        mse_per_participant = {}
        for name, (metrics, _) in models.items():
            weighted = torch.nan_to_num(metrics[split_name]['mse_per_session'] * n_trials_per_session)
            sse = torch.zeros(len(participants)).scatter_add_(0, participant_index, weighted)
            mse_per_participant[name] = torch.where(n_trials_per_participant > 0, sse / n_trials_per_participant.clamp(min=1), float('nan')).numpy()
        comparisons[split_name] = participant_win_counts(mse_per_participant, reference='SPICE-EQ', higher_is_better=False)

    # ------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------
    rows = []
    for name, (metrics, summary) in models.items():
        row = {'Model': name, **{key: summary[key] for key in ('n_parameters', 'n_parameters (std)', 'n_parameters (total)')}}
        for split_name, prefix in (('train', 'Train '), ('test', 'Test ')):
            mse_per_session = metrics[split_name]['mse_per_session']
            row.update({
                f'{prefix}MSE': metrics[split_name]['mse'],
                f'{prefix}MSE (std)': mse_per_session[~mse_per_session.isnan()].std().item(),
                f'{prefix}RMSE': metrics[split_name]['rmse'],
                f'{prefix}MAE': metrics[split_name]['mae'],
                f'{prefix}R²': metrics[split_name]['r2'],
            })
            if split_name == 'train':
                interpretable = name in INFORMATION_CRITERIA_MODELS
                row['AIC'] = summary['AIC'] if interpretable else float('nan')
                row['BIC'] = summary['BIC'] if interpretable else float('nan')
        for split_name, prefix in (('train', 'Train '), ('test', 'Test ')):
            if comparisons[split_name]:
                row.update(win_count_columns(comparisons[split_name].get(name), reference='SPICE-EQ', delta_label='MSE', prefix=prefix))
        rows.append(row)

    df = pd.DataFrame(rows).set_index('Model')

    if verbose:
        print("\nModel Evaluation (MSE):")
        print(format_table(df))

    # --- Plot ---
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'model_evaluation_mse.csv'))
        _plot_mse_comparison({name: metrics['test'] for name, (metrics, _) in models.items()}, output_dir)

    return df


def _plot_mse_comparison(models: dict, output_dir: str) -> None:
    """Bar chart of MSE per model + per-session violin overlay (models: name -> metrics of one split)."""
    model_names = list(models.keys())
    n_models = len(model_names)
    colors = plt.cm.tab10.colors[:n_models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: Aggregate metrics bar chart ---
    ax = axes[0]
    metric_names = ['MSE', 'RMSE', 'MAE']
    x = np.arange(len(metric_names))
    width = 0.8 / n_models

    for i, name in enumerate(model_names):
        metrics = models[name]
        vals = [metrics['mse'], metrics['rmse'], metrics['mae']]
        ax.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.8)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Error')
    ax.set_title('Aggregate Metrics (test)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 2: Per-session MSE violin ---
    ax = axes[1]
    data = []
    for name in model_names:
        mse_ps = models[name]['mse_per_session']
        data.append(mse_ps[~mse_ps.isnan()].numpy())

    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
    for j, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[j])
        body.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('gray')
    parts['cmedians'].set_linestyle('--')

    ax.set_xticks(range(1, n_models + 1))
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel('MSE per session')
    ax.set_title('Per-Session MSE Distribution')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_evaluation_mse.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(output_dir, 'model_evaluation_mse.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)