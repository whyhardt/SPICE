import optuna
import numpy as np

from .fit_sindy import fit_sindy_pipeline
from .bandits import AgentSpice, AgentNetwork, get_update_dynamics, Bandits
from .sindy_utils import DatasetRNN


def optimize_for_participant(
    participant_id: int,
    agent_rnn: AgentNetwork,
    data: DatasetRNN,
    metric_rnn: float,
    rnn_modules: list,
    control_signals: list,
    library_setup: dict,
    filter_setup: dict,
    polynomial_degree: int,
    optimizer_type: str,
    simulation_environment: Bandits,
    n_sessions_off_policy: int,
    n_trials_off_policy: int,
    n_trials_same_action_off_policy: int,
    n_trials_optuna: int = 50,
    timeout: int = 600,  # ? 10 minutes timeout
    threshold: float = 0.01,
    verbose: bool = False,
):
    """
    Use Optuna to find the best optimizer type and hyperparameters for a specific participant.
    
    Args:
        variables: RNN module variables for the participant
        control: Control signals for the participant
        rnn_modules: List of RNN module names
        control_signals: List of control signal names
        library_setup: Dictionary mapping features to library components
        filter_setup: Dictionary mapping features to filter conditions
        polynomial_degree: Polynomial degree for the feature library
        n_trials: Number of Optuna trials to run
        timeout: Maximum time in seconds to run optimization
        verbose: Whether to print verbose output
        
    Returns:
        dict: Best optimizer configuration with type and parameters
    """
    
    threshold = 0.01
    
    def objective(trial):
        
        # Sample optimizer type
        # optimizer_type = trial.suggest_categorical("optimizer_type", ["STLSQ", "SR3_L1"])#, "SR3_weighted_l1"])
        
        # Sample optimizer hyperparameters
        optimizer_alpha = trial.suggest_float("optimizer_alpha", 0.01, 1.0, log=True)
        optimizer_threshold = trial.suggest_float("optimizer_threshold", 0.01, 0.2, log=True)
        
        # Sample off-policy parameters
        # n_sessions_off_policy = trial.suggest_categorical("n_sessions_off_policy", [0, 1])
        # n_trials_off_policy = trial.suggest_categorical("n_trials_off_policy", [1000, 2000])
        # n_trials_same_action_off_policy = trial.suggest_categorical("n_trials_same_action_off_policy", [5, 10, 20])
        
        # just fit the SINDy modules with the given parameters 
        sindy_modules = fit_sindy_pipeline(
            participant_id=participant_id,
            agent=agent_rnn,
            data=data,
            rnn_modules=rnn_modules,
            control_signals=control_signals,
            sindy_library_setup=library_setup,
            sindy_filter_setup=filter_setup,
            sindy_dataprocessing=None,
            optimizer_type=optimizer_type,
            optimizer_alpha=optimizer_alpha,
            optimizer_threshold=optimizer_threshold,
            polynomial_degree=polynomial_degree,
            shuffle=True,
            simulation_environment=simulation_environment,
            n_sessions_off_policy=n_sessions_off_policy,
            n_trials_off_policy=n_trials_off_policy,
            n_trials_same_action_off_policy=n_trials_same_action_off_policy,
            catch_convergence_warning=False,
            verbose=verbose,
        )
        
        spice_modules = {module: {} for module in sindy_modules}        
        for rnn_module in rnn_modules:
            spice_modules[rnn_module][participant_id] = sindy_modules[rnn_module]
        
        agent_spice = AgentSpice(model_rnn=agent_rnn._model, sindy_modules=spice_modules, n_actions=agent_rnn._n_actions)
        
        # compute loss
        probs_spice = get_update_dynamics(experiment=data.xs[0], agent=agent_spice)[1]
        
        # loss: Difference between average trial likelihoods of RNN and SPICE -> SPICE can become even better than RNN; But that does not make sense on off-policy data
        # lik_spice = np.exp(log_likelihood(data.xs[0, :probs_spice.shape[0], :agent_rnn._n_actions].numpy(), probs=probs_spice) / probs_spice.size)
        # loss = metric_rnn - lik_spice
        
        # loss: MSE between predicted trial probabilities
        loss_reconstruction = np.power(metric_rnn - probs_spice, 2).mean()
        # loss_parameter = bayesian_information_criterion(data.xs[0, :len(probs_spice), :agent_spice._n_actions].numpy(), probs_spice, n_parameters=agent_spice.count_parameters()[agent_spice.get_participant_ids()[0]])/len(probs_spice)
        penalty_parameters = 0
        penalty_parameters += np.sum([np.sum(np.abs(sindy_modules[module].coefficients())) for module in sindy_modules])
        loss = loss_reconstruction + penalty_parameters * 1e-3
        
        if loss == np.nan:
            loss = 1e3
                    
        return loss
    
    threshold_no_improvement = 50
    
    def callback_no_improvement(study, trial):
        """
        Callback to stop the study if there is no improvement over the last `patience` trials.

        Args:
            study: The Optuna study object.
            trial: The Optuna trial object.
            patience (int): Number of trials to wait for an improvement.
        """
        # Store the state in the study's `user_attrs`
        if "last_best_value" not in study.user_attrs:
            study.set_user_attr("last_best_value", study.best_value)
            study.set_user_attr("trial_count", 0)
            return

        last_best_value = study.user_attrs["last_best_value"]
        trial_count = study.user_attrs["trial_count"]

        # Check for improvement
        if study.best_value < last_best_value:
            study.set_user_attr("last_best_value", study.best_value)
            study.set_user_attr("trial_count", 0)
        else:
            trial_count += 1
            study.set_user_attr("trial_count", trial_count)

            # Stop the study if patience is exceeded
            if trial_count >= threshold_no_improvement:
                study.stop()
        
    def callback_threshold_reached(study, trial):
        try:
            if study.best_value < threshold:  # Use the threshold argument
                study.stop()
        except ValueError as e:
            if "No trials are completed yet" in str(e):
                pass
        
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective, 
        n_trials=n_trials_optuna, 
        timeout=timeout, 
        show_progress_bar=True, 
        callbacks=[
            callback_threshold_reached, 
            # callback_no_improvement,
            ],
    )
    
    return {
        # "optimizer_type": study.best_params["optimizer_type"],
        "optimizer_alpha": study.best_params["optimizer_alpha"],
        "optimizer_threshold": study.best_params["optimizer_threshold"],
        # "n_sessions_off_policy": study.best_params["n_sessions_off_policy"],
        # "n_trials_off_policy": study.best_params["n_trials_off_policy"],
        # "n_trials_same_action_off_policy": study.best_params["n_trials_same_action_off_policy"],
    }