import os
import sys

import argparse
import importlib
import numpy as np
import pandas as pd
import torch

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


@torch.no_grad()
def analysis_model_evaluation(
    dataset: SpiceDataset,
    spice_model: SpiceEstimator = None,
    benchmark_model: torch.nn.Module = None,
    gru_model: torch.nn.Module = None,
    verbose: bool = False,
    ):
    
    # ------------------------------------------------------------
    # Compute choice probs
    # ------------------------------------------------------------
    
    if benchmark_model is not None:
        print("Computing choice probabilities with benchmark model...")
        benchmark_parameters = benchmark_model.count_parameters() if hasattr(benchmark_model, 'count_parameters') else len([p for p in benchmark_model.parameters()])
        benchmark_predictions, _ = benchmark_model(dataset.xs)
        benchmark_choice_probs = get_choice_probs(benchmark_predictions).detach().cpu()
    else:
        benchmark_parameters = torch.nan
        
    # setup GRU model
    if gru_model is not None:
        print("Computing choice probabilities with GRU model...")
        gru_model.eval()
        gru_parameters = sum(p.numel() for p in gru_model.parameters())
        gru_predictions, _ = gru_model(dataset.xs)
        gru_choice_probs = get_choice_probs(gru_predictions).detach().cpu()
    else:
        gru_parameters = torch.nan
        
    # setup SPICE model
    if spice_model is not None:
        spice_parameters = spice_model.count_sindy_coefficients()
        
        spice_rnn_parameters = 0
        for module in spice_model.get_modules():
            spice_rnn_parameters += sum(p.numel() for p in spice_model.model.submodules_rnn[module].parameters())
        spice_rnn_parameters += spice_model.model.embedding_size
        
        # use spice
        print("Computing choice probabilities with SPICE model...")
        spice_model.eval(aggregate=False)
        
        spice_predictions, _ = spice_model(dataset.xs.to(spice_model.device))           
        spice_choice_probs = get_choice_probs(spice_predictions.mean(dim=0)).detach().cpu()
        
        # spice_model.use_sindy(False)
        spice_model.model.init_state(batch_size=dataset.xs.shape[0])
        spice_rnn_predictions, _ = spice_model(dataset.xs.to(spice_model.device))           
        spice_rnn_choice_probs = get_choice_probs(spice_rnn_predictions.mean(dim=0)).detach().cpu()
        # spice_model.use_sindy(True)
    else:
        spice_parameters = torch.nan
        spice_rnn_parameters = torch.nan
        
    # ------------------------------------------------------------
    # Evaluation pipeline
    # ------------------------------------------------------------
    
    scores = torch.zeros((4, 3))
    metric_participant = torch.zeros((len(scores), len(dataset)))
    
    considered_trials_participant = (~torch.isnan(dataset.xs[:, :, 0, 0])).sum(dim=1)
    considered_trials = considered_trials_participant.sum()
    
    # SPICE model
    if spice_model is not None:
        participant_ids = dataset.xs[:, 0, 0, -1].cpu().numpy()
        experiment_ids = dataset.xs[:, 0, 0, -2].cpu().numpy()

        scores_spice, nll_per_sample = get_scores(targets=dataset.ys, probs=spice_choice_probs, n_parameters=spice_parameters[participant_ids, experiment_ids].mean().item())
        scores[3] += torch.tensor(scores_spice)
        metric_participant[3] = nll_per_sample.sum(dim=-1).nansum(dim=1)[..., 0]
        parameters_participant = spice_parameters[participant_ids, experiment_ids].unsqueeze(0)
        
        scores_spice_rnn, nll_per_sample = get_scores(targets=dataset.ys, probs=spice_rnn_choice_probs, n_parameters=spice_rnn_parameters)
        scores[2] += torch.tensor(scores_spice_rnn)
        metric_participant[2] = nll_per_sample.sum(dim=-1).nansum(dim=1)[..., 0]     
    else:
        parameters_participant = torch.zeros((1, len(dataset)))
        
    # Benchmark model
    if benchmark_model is not None:
        scores_benchmark, nll_per_sample = get_scores(targets=dataset.ys, probs=benchmark_choice_probs, n_parameters=benchmark_parameters)
        scores[0] += torch.tensor(scores_benchmark)
        metric_participant[0] = nll_per_sample.sum(dim=-1).nansum(dim=1)[..., 0]
        
    # GRU model
    if gru_model is not None:
        scores_gru, nll_per_sample = get_scores(targets=dataset.ys, probs=gru_choice_probs, n_parameters=gru_parameters)
        scores[1] += torch.tensor(scores_gru)
        metric_participant[1] = nll_per_sample.sum(dim=-1).nansum(dim=1)[..., 0]
        
    # ------------------------------------------------------------
    # Post processing
    # ------------------------------------------------------------

    # compute trial-level metrics (and NLL -> Likelihood)
    # scores = scores / considered_trials
    avg_trial_likelihood = torch.exp(-scores[:, 0] / considered_trials)

    avg_trial_likelihood_participant = np.exp(- metric_participant / considered_trials_participant)
    avg_trial_likelihood_participant_std = avg_trial_likelihood_participant.std(dim=1)
    parameter_participant_std = parameters_participant.std(dim=1)[0]

    # compute average number of parameters
    n_parameters = torch.tensor([
        benchmark_parameters, 
        gru_parameters,
        spice_rnn_parameters, 
        torch.mean(spice_parameters) if isinstance(spice_parameters, torch.Tensor) else spice_parameters,
        ])
    n_parameters_std = torch.tensor([
        0,
        0,
        0,
        parameter_participant_std,
    ])

    scores = torch.concatenate((
        avg_trial_likelihood.reshape(-1, 1), 
        avg_trial_likelihood_participant_std.reshape(-1, 1), 
        n_parameters.reshape(-1, 1), 
        n_parameters_std.reshape(-1, 1),
        scores[:, :1], 
        scores[:, 1:],
        ), dim=1)
    
    # ------------------------------------------------------------
    # Printing model performance table
    # ------------------------------------------------------------

    df = pd.DataFrame(
        data=scores,
        index=['Benchmark', 'GRU', 'SPICE-RNN', 'SPICE'],
        columns = ('Trial Lik.', '(std)', 'n_parameters', '(std)', 'NLL', 'AIC', 'BIC'),
        )
    
    if verbose:
        print(df)
    
    return df