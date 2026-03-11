import os
import sys

import argparse
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from typing import Iterable, Optional, Tuple

# standard methods and classes used for every model evaluation
from spice import SpiceEstimator, csv_to_dataset, split_data_along_sessiondim, BaseRNN, SpiceConfig, SpiceDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from weinhardt2025.benchmarking import benchmarking_qlearning


def prepare_benchmark(path_model: str, dataset: SpiceDataset, model_module: str = None, model_class: BaseRNN = None, n_reward_features: int = None) -> torch.nn.Module:
    # --- load benchmark or GRU model ---
    n_actions = dataset.ys.shape[-1]
    n_participants = dataset.xs[..., -1].unique().shape[0]
    
    if model_module is not None and model_class is None:
        mod = importlib.import_module(model_module)
        model_class = mod.Model
    elif model_module is None and model_class is not None:
        pass
    else:
        raise ValueError("You have to give either (model_module) OR (model_class AND model_config).")
    
    model = model_class(
        n_actions=n_actions,
        n_participants=n_participants,
        n_reward_features=n_reward_features,
        )
    
    state_dict = torch.load(path_model, map_location='cpu')
    model.load_state_dict(state_dict)
    return model.eval()


def prepare_spice(path_model: str, dataset: SpiceDataset, model_module: str = None, model_class: BaseRNN = None, model_config: SpiceConfig = None, n_reward_features: int = None) -> SpiceEstimator:
    # --- load SPICE model via precoded module ---
    if model_module is not None and model_class is None and model_config is None:
        mod = importlib.import_module(model_module)
        rnn_class = mod.SpiceModel
        spice_config = mod.CONFIG
    elif model_module is None and model_class is not None and model_config is not None:
        rnn_class = model_class
        spice_config = model_config
    else:
        raise ValueError("You have to give either (model_module) OR (model_class AND model_config).")
    
    n_actions = dataset.ys.shape[-1]
    n_participants = dataset.xs[..., -1].unique().shape[0]
    
    spice_estimator = SpiceEstimator(
        spice_class=rnn_class,
        spice_config=spice_config,
        n_actions=n_actions,
        n_participants=n_participants,
        n_reward_features=n_reward_features,
        sindy_library_polynomial_degree=2,
    )
    spice_estimator.load_spice(path_model=path_model)
    spice_estimator.model.eval()
    return spice_estimator


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
    log_likelihoods = data * torch.log(probs)# + (1 - data) * torch.log(1 - probs)
    # log_likelihoods = data * torch.log(probs)
    # log_likelihoods = torch.sum(data * torch.log(probs), axis=-1)
    
    # Sum log-likelihoods over all observations
    return torch.sum(log_likelihoods)


def bayesian_information_criterion(data: torch.Tensor, probs: torch.Tensor, n_parameters: int, nll: torch.Tensor = None, **kwargs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if nll is None:
        nll = log_likelihood(data=data, probs=probs)
    
    n_samples = (data[:, 0] != -1).sum()
    return 2 * nll + n_parameters * torch.log(n_samples)


def akaike_information_criterion(data: torch.Tensor, probs: torch.Tensor, n_parameters: int, nll: torch.Tensor = None, **kwargs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if nll is None:
        nll = log_likelihood(data=data, probs=probs)
    
    return 2 * nll + 2 * n_parameters


def get_scores(probs: torch.Tensor, targets: torch.Tensor, n_parameters: int) -> Tuple[float, float, float]:
    nll = -log_likelihood(data=targets, probs=probs)
    bic = bayesian_information_criterion(data=targets, probs=probs, n_parameters=n_parameters, nll=nll)
    aic = akaike_information_criterion(data=targets, probs=probs, n_parameters=n_parameters, nll=nll)
    return nll, aic, bic


def analysis_model_evaluation(
    dataset: SpiceDataset,
    list_test_sessions: Optional[Iterable[int]] = None,
    
    spice_path: str = None,    
    spice_module: str = None,
    spice_class: BaseRNN = None,
    spice_config: SpiceConfig = None,
    spice_model: BaseRNN = None,
    
    benchmark_path: str = None,
    benchmark_module: str = None,
    benchmark_class: torch.nn.Module = None,
    benchmark_model: torch.nn.Module = None,
    
    gru_path: str = None,
    gru_module: str = None,
    gru_class: torch.nn.Module = None,
    gru_model: torch.nn.Module = None,
    
    verbose: bool = False,
    ):
    
    # ------------------------------------------------------------
    # General setup
    # ------------------------------------------------------------
    
    participant_ids = dataset.xs[:, 0, -1].unique().cpu().numpy()
    n_participants = int(max(participant_ids)+1)
    n_actions = dataset.ys.shape[-1]
    
    # ------------------------------------------------------------
    # Dataset splitting
    # ------------------------------------------------------------

    if list_test_sessions is not None:
        _, dataset_test = split_data_along_sessiondim(dataset, list_test_sessions=list_test_sessions)
    else:
        _, dataset_test = dataset, dataset
        
    # NaN data mask
    nan_mask = ~torch.isnan(dataset_test.xs[..., :n_actions].sum(dim=-1))

    # ------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------
    with torch.no_grad():
        # setup benchmark model
        if benchmark_model is None and benchmark_path is not None and (benchmark_module is not None or benchmark_class is not None):
            benchmark_model = prepare_benchmark(path_model=benchmark_path, dataset=dataset, model_module=benchmark_module, model_class=benchmark_class, n_reward_features=n_reward_features)
        if benchmark_model is not None:
            print("Computing choice probabilities with benchmark model...")
            benchmark_parameters = len([p for p in benchmark_model.parameters()]) / n_participants
            benchmark_predictions, _ = benchmark_model(dataset_test.xs)
            benchmark_choice_probs = get_choice_probs(benchmark_predictions).detach().cpu()
        else:
            benchmark_parameters = torch.nan
            
        # setup GRU model
        if gru_model is None and gru_path is not None and (gru_module is not None or gru_class is not None):
            gru_model = prepare_benchmark(path_model=gru_path, dataset=dataset, model_module=gru_module, model_class=gru_class, n_reward_features=1)
        if gru_model is not None:
            print("Computing choice probabilities with GRU model...")
            gru_parameters = sum(p.numel() for p in gru_model.parameters())
            gru_predictions, _ = gru_model(dataset_test.xs)
            gru_choice_probs = get_choice_probs(gru_predictions).detach().cpu()
        else:
            gru_parameters = torch.nan
            
        # setup rnn agent
        if spice_model is None and spice_path is not None and (spice_module is not None or (spice_class is not None and spice_config is not None)):
            spice_model = prepare_spice(path_model=spice_path, dataset=dataset, model_module=spice_module, model_class=spice_class, model_config=spice_config)
        if spice_model is not None:
            spice_parameters = spice_model.count_sindy_coefficients()
            
            spice_rnn_parameters = 0
            for module in spice_model.get_modules():
                spice_rnn_parameters += sum(p.numel() for p in spice_model.model.submodules_rnn[module].parameters())
            spice_rnn_parameters += spice_model.model.embedding_size
            
            # use spice
            print("Computing choice probabilities with SPICE model...")
            spice_rnn_predictions, spice_predictions = spice_model.predict(dataset_test.xs.to(spice_model.device))           
            spice_rnn_predictions, spice_predictions = torch.tensor(spice_rnn_predictions), torch.tensor(spice_predictions)
            spice_rnn_choice_probs = get_choice_probs(spice_rnn_predictions).detach().cpu()
            spice_choice_probs = get_choice_probs(spice_predictions).detach().cpu()
        else:
            spice_parameters = torch.nan
            spice_rnn_parameters = torch.nan
            
    # ------------------------------------------------------------
    # Evaluation pipeline
    # ------------------------------------------------------------

    scores = torch.zeros((4, 3))
    metric_participant = torch.zeros((len(scores), len(dataset_test)))
    parameters_participant = torch.zeros((1, len(dataset_test)))
    considered_trials_participant = torch.zeros(len(dataset_test))
    considered_trials = 0
    
    for index_data in range(len(dataset_test)):
        # use whole session to include warm-up phase; make sure to exclude warm-up phase when computing metrics
        pid = dataset_test.xs[index_data, 0, 0, -1].int().item()
        
        # get number of actual trials
        n_trials = nan_mask[index_data].sum()
        considered_trials_participant[index_data] += n_trials
        considered_trials += n_trials
        
        # SPICE model
        if spice_model is not None:
            scores_spice = torch.tensor(
                get_scores(targets=dataset_test.ys[index_data, nan_mask[index_data]], 
                           probs=spice_choice_probs[index_data, nan_mask[index_data]], 
                           n_parameters=spice_parameters[pid],
                           ))
            scores[3] += scores_spice
            
            scores_spice_rnn = torch.tensor(
                get_scores(targets=dataset_test.ys[index_data, nan_mask[index_data]], 
                           probs=spice_rnn_choice_probs[index_data, nan_mask[index_data]], 
                           n_parameters=spice_rnn_parameters,
                           ))
            scores[2] += scores_spice_rnn
            
            metric_participant[2, index_data] = scores_spice_rnn[0]      
            metric_participant[3, index_data] = scores_spice[0]      
            parameters_participant[0, index_data] = spice_parameters[pid]
        
        # Benchmark model
        if benchmark_model is not None:
            scores_benchmark = torch.tensor(
                get_scores(targets=dataset_test.ys[index_data, nan_mask[index_data]], 
                           probs=benchmark_choice_probs[index_data, nan_mask[index_data]], 
                           n_parameters=benchmark_parameters,
                           ))
            scores[0] += scores_benchmark
            metric_participant[0, index_data] = scores_benchmark[0]
                  
        # GRU model
        if gru_model is not None:
            scores_gru = torch.tensor(
                get_scores(targets=dataset_test.ys[index_data, nan_mask[index_data]], 
                           probs=gru_choice_probs[index_data, nan_mask[index_data]], 
                           n_parameters=gru_parameters,
                           ))
            scores[1] += scores_gru
            metric_participant[1, index_data] = scores_gru[0]
        
    # ------------------------------------------------------------
    # Post processing
    # ------------------------------------------------------------

    # compute trial-level metrics (and NLL -> Likelihood)
    scores = scores / considered_trials
    avg_trial_likelihood = torch.exp(-scores[:, 0])

    metric_participant_std = (metric_participant/considered_trials_participant).std(dim=1)
    avg_trial_likelihood_participant = np.exp(- metric_participant / considered_trials_participant)
    avg_trial_likelihood_participant_std = avg_trial_likelihood_participant.std(dim=1)
    parameter_participant_std = parameters_participant.std(dim=1)[0]

    # compute average number of parameters
    n_parameters = torch.tensor([
        benchmark_parameters, 
        gru_parameters,
        spice_rnn_parameters, 
        torch.mean(spice_parameters),
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
        scores[:, :1], 
        metric_participant_std.reshape(-1, 1), 
        scores[:, 1:], 
        n_parameters.reshape(-1, 1), 
        n_parameters_std.reshape(-1, 1),
        ), dim=1)
    
    # ------------------------------------------------------------
    # Printing model performance table
    # ------------------------------------------------------------

    df = pd.DataFrame(
        data=scores,
        index=['Benchmark', 'GRU', 'RNN', 'SPICE'],
        columns = ('Trial Lik.', '(std)', 'NLL', '(std)', 'AIC', 'BIC', 'n_parameters', '(std)'),
        )
    
    if verbose:
        print(df)
    
    return df


if __name__=='__main__':
    p = argparse.ArgumentParser(
        description="Model evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data", default=None,
                   help="Path to the experiment data CSV")
    p.add_argument("--test_sessions", default=None,
                   help="Sessions to test the models against (comma-separated list)")
    p.add_argument("--spice_model", default=None,
                   help="Path to the trained SPICE model (.pkl)")
    p.add_argument("--spice-module", default="spice.precoded.workingmemory_rewardbinary",
                   help="Name of the SPICE model module (default: spice.precoded.workingmemory_rewardbinary)")
    p.add_argument("--benchmark_model", default=None,
                   help="Path to the trained benchmark model (.pkl)")
    p.add_argument("--benchmark-module", default="weinhardt2025.benchmarking.benchmarking_qlearning",
                   help="Name of the benchmark model module (default: weinhardt2025.benchmarking.benchmarking_qlearning)")
    p.add_argument("--gru_model", default=None,
                   help="Path to the trained SPICE model (.pkl)")
    p.add_argument("--gru-module", default="weinhardt2025.benchmarking.benchmarking_gru",
                   help="Name of the SPICE model module (default: weinhardt2025.benchmarking.benchmarking_gru)")
    args  = p.parse_args()
    
    
    args.data = 'weinhardt2025/data/dezfouli2019/dezfouli2019.csv'
    args.test_sessions = (3,6,9)
    args.spice_model = 'weinhardt2025/params/dezfouli2019/spice_dezfouli2019.pkl'
    args.gru_model = 'weinhardt2025/params/dezfouli2019/gru_dezfouli2019.pkl'
    args.verbose = True    
    
    dataset = csv_to_dataset(
        file=args.data,
    )
    
    analysis_model_evaluation(
        dataset=dataset,
        list_test_sessions=args.test_sessions,
        
        spice_path=args.spice_model,
        spice_module=args.spice_module,
        
        benchmark_path=args.benchmark_model,
        benchmark_module=args.benchmark_module,
        
        gru_path=args.gru_model,
        gru_module=args.gru_module,
        
        verbose=True,
    )
    