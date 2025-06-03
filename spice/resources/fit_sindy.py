from typing import List, Dict, Tuple
import numpy as np
from math import comb

import pysindy as ps

from .sindy_utils import remove_control_features, conditional_filtering
from .rnn_utils import DatasetRNN
from .sindy_utils import generate_off_policy_data, create_dataset
from .bandits import AgentNetwork

def fit_sindy(
    variables: List[np.ndarray], 
    control: List[np.ndarray] = None, 
    rnn_modules: List[str] = None,
    control_signals: List[str] = None, 
    polynomial_degree: int = 1, 
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    optimizer_type: str = "SR3_L1",
    optimizer_threshold: float = 0.05,
    optimizer_alpha: float = 1,
    catch_convergence_warning: bool = False,
    verbose: bool = False,
    ):
    
    # get all x-features
    x_features = rnn_modules
    # get all control features
    c_features = control_signals
    
    # make sure that all x_features are in the library_setup
    for feature in x_features:
        if feature not in library_setup:
            library_setup[feature] = []
    
    # train one sindy model per variable
    sindy_models = {feature: None for feature in x_features}
    loss = 0
    for index_feature, x_feature in enumerate(x_features):
        if verbose:
            print(f'\nSINDy model for {x_feature}:')
        
        # sort signals into corresponding arrays    
        x_i = [x[:, index_feature].reshape(-1, 1) for x in variables] # get current x-feature as target variable
        control_i = control
        
        feature_names_i = [x_feature] + c_features
        
        # filter target variable and control features according to filter conditions
        if x_feature in filter_setup:
            if not isinstance(filter_setup[x_feature][0], list):
                # check that filter_setup[x_feature] is a list of filter-conditions 
                filter_setup[x_feature] = [filter_setup[x_feature]]
            for filter_condition in filter_setup[x_feature]:
                x_i, control_i, feature_names_i = conditional_filtering(
                    x_train=x_i, 
                    control=control_i, 
                    feature_names=feature_names_i, 
                    feature_filter=filter_condition[0], 
                    condition=filter_condition[1], 
                    remove_feature_filter=False
                )
        
        # remove unnecessary control features according to library setup
        control_i = remove_control_features(control_i, feature_names_i[1:], library_setup[x_feature])
        feature_names_i = [x_feature] + library_setup[x_feature]
        
        # add a dummy control feature if no control features are remaining - otherwise sindy breaks
        # if control_i is None or len(control_i) == 0:
        #     raise NotImplementedError('Having no control signal in a module is currently not implemented')
        #     control_i = None
        #     feature_names_i = feature_names_i + ['dummy']
        
        # Set up increasing thresholds with polynomial degree for SR3_weighted_l1
        if optimizer_type == "SR3_weighted_l1":
            n_polynomial_combinations = np.array([comb(len(feature_names_i) + d, d) for d in range(polynomial_degree+1)])
            thresholds = np.zeros((1, n_polynomial_combinations[-1]))
            index = 0
            for d in range(len(n_polynomial_combinations)):
                thresholds[0, index:n_polynomial_combinations[d]] = d * optimizer_threshold
                index = n_polynomial_combinations[d]

        # Create optimizer based on type
        if optimizer_type == "STLSQ":
            optimizer = ps.STLSQ(alpha=optimizer_alpha, threshold=optimizer_threshold)
        elif optimizer_type == "SR3_L1":
            optimizer = ps.SR3(
                thresholder="L1",
                nu=optimizer_alpha,
                threshold=optimizer_threshold,
                verbose=verbose,
                max_iter=100
            )
        else:  # "SR3_weighted_l1" (default)
            optimizer = ps.SR3(
                thresholder="weighted_l1",
                nu=optimizer_alpha,
                threshold=optimizer_threshold,
                thresholds=thresholds,
                verbose=verbose,
                max_iter=100
            )
            
        # Setup sindy model for current x-feature
        sindy_models[x_feature] = ps.SINDy(
            optimizer=optimizer,
            feature_library=ps.PolynomialLibrary(polynomial_degree),
            discrete_time=True,
            feature_names=feature_names_i,
        )

        # fit sindy model
        sindy_models[x_feature].fit(x_i, u=control_i, t=1, multiple_trajectories=True, ensemble=False)
        
        if catch_convergence_warning and sindy_models[x_feature].optimizer.iters >= sindy_models[x_feature].optimizer.max_iter-1:
            raise RuntimeError("SINDy optimizer did not converge.")
        
        if verbose:
            sindy_models[x_feature].print()
    
    return sindy_models


def fit_sindy_pipeline(
    participant_id: int,
    agent: AgentNetwork,
    data: DatasetRNN,
    rnn_modules: List,
    control_signals: List,
    sindy_library_setup: Dict,
    sindy_filter_setup: Dict,
    sindy_dataprocessing: Dict,
    optimizer_type: str,
    optimizer_alpha: float,
    optimizer_threshold: float,
    polynomial_degree: int,
    shuffle: bool,
    n_sessions_off_policy: int,
    n_trials_off_policy: int,
    n_trials_same_action_off_policy: int,
    catch_convergence_warning: bool,
    verbose: bool,
    ):
    
    # get participant data
    if n_sessions_off_policy == 0 and data is not None:
        mask_participant_id = dataset_fit.xs[:, 0, -1] == participant_id
        dataset_fit = DatasetRNN(*data[mask_participant_id])
    elif n_sessions_off_policy > 0:
        dataset_fit = generate_off_policy_data(
            participant_id=participant_id, 
            n_trials_off_policy=n_trials_off_policy,
            n_trials_same_action_off_policy=n_trials_same_action_off_policy,
            n_sessions_off_policy=n_sessions_off_policy,
            n_actions=agent._n_actions,
            )
    else:
        raise ValueError("One of the arguments data or n_sessions_off_policy (> 0) must be given. If n_sessions_off_policy > 0 the SINDy modules will be fitted on the off-policy data regardless of data. If n_sessions_off_policy = 0 then data will be used to fit the SINDy modules.")
    
    # extract all necessary data from the RNN (memory state) and align with the control inputs (action, reward)
    rnn_variables, control_parameters, _, _ = create_dataset(
        agent=agent,
        data=dataset_fit,
        rnn_modules=rnn_modules,
        control_signals=control_signals,
        shuffle=shuffle,
        dataprocessing=sindy_dataprocessing,
    )
    
    # fit one SINDy-model per RNN-module
    sindy_modules = fit_sindy(
        variables=rnn_variables,
        control=control_parameters,
        rnn_modules=rnn_modules,
        control_signals=control_signals,
        polynomial_degree=polynomial_degree,
        library_setup=sindy_library_setup,
        filter_setup=sindy_filter_setup,
        optimizer_type=optimizer_type,
        optimizer_alpha=optimizer_alpha,
        optimizer_threshold=optimizer_threshold,
        catch_convergence_warning=catch_convergence_warning,
        verbose=verbose,
    )
    
    return sindy_modules