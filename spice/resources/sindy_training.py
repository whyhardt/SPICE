from typing import List, Dict, Tuple, Iterable
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

from resources.sindy_utils import remove_bad_participants
from resources.fit_sindy import fit_sindy_pipeline
from resources.rnn_utils import DatasetRNN,split_data_along_timedim
from resources.bandits import AgentNetwork, AgentSpice, get_update_dynamics
from resources.model_evaluation import bayesian_information_criterion as loss_metric, log_likelihood
from resources.optimizer_selection import optimize_for_participant


def module_pruning(agent_spice: AgentSpice, dataset: DatasetRNN, participant_ids: Iterable[int], verbose: bool = False):
    """Check for badly fitted participants in the SPICE models w.r.t. the SPICE-RNN and return only the IDs of the well-fitted participants.

    Args:
        agent_spice (AgentSpice): _description_
        dataset_test (DatasetRNN): _description_
        participant_ids (Iterable[int]): _description_
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        AgentSpice: pruned SPICE agent
    """
    if verbose:
        print("\nPruning unnecessary SPICE modules...")
    
    n_removed_modules = 0
    for pid in tqdm(participant_ids):
        # Skip if participant is not in the SPICE model
        if pid not in agent_spice._model.submodules_sindy[list(agent_spice._model.submodules_sindy.keys())[0]]:
            if verbose:
                print("Participant ID", pid, "not found in SPICE model. Skipped that participant.")
            continue
        
        agent_spice.new_sess(participant_id=pid)
        keys_betas = agent_spice.get_betas().keys()
        if not any([agent_spice.get_betas()[key] == 0 for key in keys_betas]):
            continue
        
        # Calculate normalized log likelihood for SPICE and RNN
        mask_participant_id = dataset.xs[:, 0, -1] == pid
        if not mask_participant_id.any():
            if verbose:
                print("Participant ID", pid, "not found in dataset. Skipped that participant.")
            continue
            
        participant_data = DatasetRNN(*dataset[mask_participant_id])
        
        # Get probabilities from SPICE and RNN models
        agent_spice.new_sess(participant_id=pid)
        
        # Calculate action probabilities for original model
        probs_spice = get_update_dynamics(experiment=participant_data.xs, agent=agent_spice)[1]
        # n_trials_test = len(probs_spice)
        
        # Calculate action probabilities for pruned models
        for module in agent_spice.get_modules():
            agent_spice_modified = deepcopy(agent_spice)
            module_modified = agent_spice_modified.get_modules()[module][pid]
            
            # set all coefficients of the currently investigated module to 0
            module_modified.optimizer.coef_ = np.zeros_like(module_modified.optimizer.coef_)

            # put modified module back into agent
            # agent_spice_modified._model.submodules_sindy[module][pid] = module_modified
            
            # get new action probabilities and compare to original ones
            probs_spice_modified = get_update_dynamics(experiment=participant_data.xs, agent=agent_spice_modified)[1]
            if np.sum(np.abs(probs_spice - probs_spice_modified)) == 0 and np.sum(agent_spice.get_modules()[module][pid].coefficients()) != 0:
                if verbose:
                    print("Found unnecessary module", module, "for participant", pid)
                agent_spice._model.submodules_sindy[module][pid].optimizer.coef_ = np.zeros_like(module_modified.optimizer.coef_)
                n_removed_modules += 1
                
    if verbose:
        print("Removed modules:", n_removed_modules)
        
    return agent_spice
        

def fit_spice(
    rnn_modules: List[np.ndarray],
    control_signals: List[np.ndarray], 
    agent_rnn: AgentNetwork,
    data: DatasetRNN = None,
    polynomial_degree: int = 2, 
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    optimizer_type: str = "SR3_L1",
    optimizer_threshold: float = 0.05,
    optimizer_alpha: float = 0.1,
    participant_id: int = None,
    shuffle: bool = False,
    dataprocessing: Dict[str, List] = None,
    n_trials_off_policy: int = 1000,
    n_sessions_off_policy: int = 1,
    n_trials_same_action_off_policy: int = 5,
    train_test_ratio: float = 1.0,
    deterministic: bool = True,
    get_loss: bool = False,
    verbose: bool = False,
    use_optuna: bool = False,
    filter_bad_participants: bool = False,
    pruning: bool = False,
    ) -> Tuple[AgentSpice, float]:
    """Fit a SPICE agent by replacing RNN modules with SINDy equations.

    Args:
        rnn_modules (List[np.ndarray]): List of RNN module names to be replaced with SINDy
        control_signals (List[np.ndarray]): List of control signal names
        agent (AgentNetwork): The trained RNN agent
        data (DatasetRNN, optional): Dataset for training/evaluation. Defaults to None.
        polynomial_degree (int, optional): Polynomial degree for SINDy. Defaults to 2.
        library_setup (Dict[str, List[str]], optional): Dictionary mapping features to library components. Defaults to {}.
        filter_setup (Dict[str, Tuple[str, float]], optional): Dictionary mapping features to filter conditions. Defaults to {}.
        optimizer_type (str, optional): Type of optimizer to use. Defaults to "SR3_L1".
        optimizer_threshold (float, optional): Threshold for optimizer. Defaults to 0.05.
        optimizer_alpha (float, optional): Alpha parameter for optimizer. Defaults to 1.
        participant_id (int, optional): Specific participant ID to process. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        dataprocessing (Dict[str, List], optional): Data processing options. Defaults to None.
        n_trials_off_policy (int, optional): Number of off-policy trials. Defaults to 2048.
        n_sessions_off_policy (int, optional): Number of off-policy sessions. Defaults to 1.
        deterministic (bool, optional): Whether to use deterministic mode. Defaults to True.
        get_loss (bool, optional): Whether to compute loss. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        use_optuna (bool, optional): Whether to use Optuna for optimizer selection. Defaults to False.

    Returns:
        Tuple[AgentSpice, float]: The SPICE agent and its loss
    """
    
    data = split_data_along_timedim(data, split_ratio=train_test_ratio)[0]
    
    if participant_id is not None:
        participant_ids = [participant_id]
    elif data is not None and participant_id is None:
        participant_ids = data.xs[..., -1].unique().int().cpu().numpy()
    else:
        raise ValueError("Either data or participant_id are required.")
        
    sindy_modules = {rnn_module: {} for rnn_module in rnn_modules}
    optuna_pids = []
    likelihoods_rnn = []
    likelihoods_spice_before_optuna = []
    likelihoods_spice_after_optuna = []
    for pid in tqdm(participant_ids):
        
        if data is not None:
            mask_participant_id = data.xs[:, 0, -1] == pid
            data_pid = DatasetRNN(*data[mask_participant_id])
        
        try:
        # just fit the SINDy modules with the given parameters 
            sindy_modules_id = fit_sindy_pipeline(
                participant_id=pid,
                agent=agent_rnn,
                data=data_pid,
                rnn_modules=rnn_modules,
                control_signals=control_signals,
                sindy_library_setup=library_setup,
                sindy_filter_setup=filter_setup,
                sindy_dataprocessing=dataprocessing,
                optimizer_type=optimizer_type,
                optimizer_alpha=optimizer_alpha,
                optimizer_threshold=optimizer_threshold,
                polynomial_degree=polynomial_degree,
                shuffle=shuffle,
                n_sessions_off_policy=n_sessions_off_policy,
                n_trials_off_policy=n_trials_off_policy,
                n_trials_same_action_off_policy=n_trials_same_action_off_policy,
                catch_convergence_warning=False,
                verbose=verbose,
            )
            
            spice_modules_id = {rnn_module: {} for rnn_module in rnn_modules}
            for rnn_module in rnn_modules:
                spice_modules_id[rnn_module][pid] = sindy_modules_id[rnn_module]
            agent_spice_id = AgentSpice(model_rnn=agent_rnn._model, sindy_modules=spice_modules_id, n_actions=agent_rnn._n_actions)
            probs_rnn = get_update_dynamics(agent=agent_rnn, experiment=data_pid.xs[0])[1]
            probs_spice = get_update_dynamics(agent=agent_spice_id, experiment=data_pid.xs[0])[1]
            lik_rnn = np.exp(log_likelihood(data=data_pid.xs[0, :probs_rnn.shape[0], :agent_rnn._n_actions].numpy(), probs=probs_rnn) / probs_rnn.size)
            lik_spice_before_optuna = np.exp(log_likelihood(data=data_pid.xs[0, :probs_rnn.shape[0], :agent_rnn._n_actions].numpy(), probs=probs_spice) / probs_spice.size)
            if lik_rnn - lik_spice_before_optuna > 0.03 or np.isnan(lik_spice_before_optuna):
                likelihoods_rnn.append(np.round(lik_rnn, 5))
                likelihoods_spice_before_optuna.append(np.round(lik_spice_before_optuna, 5))
                raise RuntimeError(f"Bad fit of SPICE model.\nLikelihoods before optuna fitting: RNN = {np.round(lik_rnn, 5)}; SPICE = {np.round(lik_spice_before_optuna, 5)}; Diff = {np.round(lik_rnn-lik_spice_before_optuna, 5)}\nStarting optuna...")
            
        except Exception as e:
            # If using Optuna, find the best optimizer configuration for this participant
            print(e)
            optuna_pids.append(pid)
            if use_optuna:
                print(f"Using optuna to find a better set of pysindy parameters for participant {pid}...")
                # Find optimal optimizer and parameters for this participant
                sindy_config = optimize_for_participant(
                    participant_id=pid,
                    agent_rnn=agent_rnn,
                    data=data_pid,
                    metric_rnn=lik_rnn,
                    rnn_modules=rnn_modules,
                    control_signals=control_signals,
                    library_setup=library_setup,
                    filter_setup=filter_setup,
                    polynomial_degree=polynomial_degree,
                    n_sessions_off_policy=n_sessions_off_policy,
                    n_trials_optuna=50,  # Adjust as needed
                    verbose=verbose
                )
                
                optuna_optimizer_type = sindy_config["optimizer_type"]
                optuna_optimizer_alpha = sindy_config["optimizer_alpha"]
                optuna_optimizer_threshold = sindy_config["optimizer_threshold"]
                optuna_n_trials_off_policy = sindy_config["n_trials_off_policy"]
                optuna_n_trials_same_action_off_policy = sindy_config["n_trials_same_action_off_policy"]
                
                if verbose:
                    print(f"\nUsing optimized parameters for participant {pid}:")
                    print(f"\tOptimizer type: {optuna_optimizer_type}")
                    print(f"\tAlpha: {optuna_optimizer_alpha}")
                    print(f"\tThreshold: {optuna_optimizer_threshold}")
                    print(f"\tOff-polcy trials: {optuna_n_trials_off_policy}")
                    print(f"\tSame action in off-polcy trials: {optuna_n_trials_same_action_off_policy}")
                    
                sindy_modules_optuna = fit_sindy_pipeline(
                participant_id=pid,
                agent=agent_rnn,
                data=data_pid,
                rnn_modules=rnn_modules,
                control_signals=control_signals,
                sindy_library_setup=library_setup,
                sindy_filter_setup=filter_setup,
                sindy_dataprocessing=dataprocessing,
                optimizer_type=optuna_optimizer_type,
                optimizer_alpha=optuna_optimizer_alpha,
                optimizer_threshold=optuna_optimizer_threshold,
                polynomial_degree=polynomial_degree,
                shuffle=shuffle,
                n_sessions_off_policy=n_sessions_off_policy,
                n_trials_off_policy=optuna_n_trials_off_policy,
                n_trials_same_action_off_policy=optuna_n_trials_same_action_off_policy,
                catch_convergence_warning=False,
                verbose=verbose,
                )
                
                spice_modules_optuna = {rnn_module: {} for rnn_module in rnn_modules}
                for rnn_module in rnn_modules:
                    spice_modules_optuna[rnn_module][pid] = sindy_modules_optuna[rnn_module]
                agent_spice_optuna = AgentSpice(model_rnn=agent_rnn._model, sindy_modules=spice_modules_optuna, n_actions=agent_rnn._n_actions)
                probs_spice = get_update_dynamics(agent=agent_spice_optuna, experiment=data_pid.xs[0])[1]
                lik_spice_after_optuna = np.exp(log_likelihood(data=data_pid.xs[0, :probs_rnn.shape[0], :agent_rnn._n_actions].numpy(), probs=probs_spice) / probs_spice.size)
                
                if lik_rnn - lik_spice_after_optuna > 0.1 or np.isnan(lik_spice_after_optuna):
                    
                    print(f"Did not find satisfying solution after 50 optuna trials.\nSPICE = {np.round(lik_spice_before_optuna, 5)} -> {np.round(lik_spice_after_optuna, 5)}\nStarting again with 100 trials...")
                    
                    # Find optimal optimizer and parameters for this participant
                    sindy_config = optimize_for_participant(
                        participant_id=pid,
                        agent_rnn=agent_rnn,
                        data=data_pid,
                        metric_rnn=lik_rnn,
                        rnn_modules=rnn_modules,
                        control_signals=control_signals,
                        library_setup=library_setup,
                        filter_setup=filter_setup,
                        polynomial_degree=polynomial_degree,
                        n_sessions_off_policy=n_sessions_off_policy,
                        n_trials_optuna=100,  # Adjust as needed
                        verbose=verbose
                    )
                    
                    optuna_optimizer_type = sindy_config["optimizer_type"]
                    optuna_optimizer_alpha = sindy_config["optimizer_alpha"]
                    optuna_optimizer_threshold = sindy_config["optimizer_threshold"]
                    optuna_n_trials_off_policy = sindy_config["n_trials_off_policy"]
                    optuna_n_trials_same_action_off_policy = sindy_config["n_trials_same_action_off_policy"]
                    
                    if verbose:
                        print(f"\nUsing optimized parameters for participant {pid}:")
                        print(f"\tOptimizer type: {optuna_optimizer_type}")
                        print(f"\tAlpha: {optuna_optimizer_alpha}")
                        print(f"\tThreshold: {optuna_optimizer_threshold}")
                        print(f"\tOff-polcy trials: {optuna_n_trials_off_policy}")
                        print(f"\tSame action in off-polcy trials: {optuna_n_trials_same_action_off_policy}")
                        
                    sindy_modules_optuna = fit_sindy_pipeline(
                    participant_id=pid,
                    agent=agent_rnn,
                    data=data_pid,
                    rnn_modules=rnn_modules,
                    control_signals=control_signals,
                    sindy_library_setup=library_setup,
                    sindy_filter_setup=filter_setup,
                    sindy_dataprocessing=dataprocessing,
                    optimizer_type=optuna_optimizer_type,
                    optimizer_alpha=optuna_optimizer_alpha,
                    optimizer_threshold=optuna_optimizer_threshold,
                    polynomial_degree=polynomial_degree,
                    shuffle=shuffle,
                    n_sessions_off_policy=n_sessions_off_policy,
                    n_trials_off_policy=optuna_n_trials_off_policy,
                    n_trials_same_action_off_policy=optuna_n_trials_same_action_off_policy,
                    catch_convergence_warning=False,
                    verbose=verbose,
                    )
                    
                    spice_modules_optuna = {rnn_module: {} for rnn_module in rnn_modules}
                    for rnn_module in rnn_modules:
                        spice_modules_optuna[rnn_module][pid] = sindy_modules_optuna[rnn_module]
                    agent_spice_optuna = AgentSpice(model_rnn=agent_rnn._model, sindy_modules=spice_modules_optuna, n_actions=agent_rnn._n_actions)
                    probs_spice = get_update_dynamics(agent=agent_spice_optuna, experiment=data_pid.xs[0])[1]
                    lik_spice_after_optuna = np.exp(log_likelihood(data=data_pid.xs[0, :probs_rnn.shape[0], :agent_rnn._n_actions].numpy(), probs=probs_spice) / probs_spice.size)
                
                likelihoods_spice_after_optuna.append(np.round(lik_spice_after_optuna, 5))
                print(f"Likelihoods after optuna fitting: RNN = {np.round(lik_rnn, 5)}; SPICE =  {np.round(lik_spice_before_optuna, 5)} -> {np.round(lik_spice_after_optuna, 5)}, Diff = {np.round(lik_rnn-lik_spice_before_optuna, 5)} -> {np.round(lik_rnn-lik_spice_after_optuna, 5)}")
                
                if lik_spice_after_optuna > lik_spice_before_optuna or np.isnan(lik_spice_before_optuna):
                    sindy_modules_id = sindy_modules_optuna
                
        for rnn_module in rnn_modules:
            sindy_modules[rnn_module][pid] = sindy_modules_id[rnn_module]

    print("Optuna-optimized participants:", optuna_pids)
    print("Likelihoods of RNN, SPICE (before), SPICE (after):")
    df = {
        'PID': optuna_pids,
        'RNN': likelihoods_rnn,
        'SPICE': likelihoods_spice_before_optuna,
        'SPICE (Optuna)': likelihoods_spice_after_optuna,
    }
    print(pd.DataFrame(data=df))
    
    # set up a SINDy-based agent by replacing the RNN-modules with the respective SINDy-model
    agent_spice = AgentSpice(model_rnn=deepcopy(agent_rnn._model), sindy_modules=sindy_modules, n_actions=agent_rnn._n_actions, deterministic=deterministic)
    
    # pruning unnecessary SINDy modules (i.e. modules which have no influence on the output but add up parameters)
    if pruning:
        agent_spice = module_pruning(
            agent_spice=agent_spice,
            dataset=data,
            participant_ids=participant_ids,
            verbose=True,
            )
    
    # remove badly fitted participants
    if filter_bad_participants:
        agent_spice, participant_ids = remove_bad_participants(
            agent_spice=agent_spice,
            agent_rnn=agent_rnn,
            dataset=data,
            participant_ids=participant_ids,
            trial_likelihood_difference_threshold=0.20,
            verbose=verbose,
        )
    
    # compute loss
    loss = None
    if get_loss and data is None:
        raise ValueError("When get_loss is True, data must be given to compute the loss. Off-policy data won't be considered to compute the loss.")
    elif get_loss and data is not None:
        loss = 0
        n_trials_total = 0
        mapping_modules_values = {module: 'x_value_choice' if 'choice' in module else 'x_value_reward' for module in agent_spice._model.submodules_sindy}
        n_parameters = agent_spice.count_parameters(mapping_modules_values=mapping_modules_values)
        for pid in participant_ids:
            xs, ys = data.xs.cpu().numpy(), data.ys.cpu().numpy()
            probs = get_update_dynamics(experiment=xs[pid], agent=agent_spice)[1]
            loss += loss_metric(data=ys[pid, :len(probs)], probs=probs, n_parameters=n_parameters[pid])
            n_trials_total += len(probs)
        loss = loss/n_trials_total
        
    return agent_spice, loss