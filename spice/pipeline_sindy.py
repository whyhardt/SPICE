import sys
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

sys.path.append('resources')
from resources.bandits import AgentQ, BanditsDrift, BanditsSwitch, plot_session, create_dataset as create_dataset_bandits
from resources.rnn import RLRNN_eckstein2022
from resources.rnn_utils import parameter_file_naming
from resources.sindy_utils import check_library_setup, save_spice, SindyConfig_eckstein2022
from resources.sindy_training import fit_spice
from utils.convert_dataset import convert_dataset
from utils.plotting import plot_session as plot_session
from utils.setup_agents import setup_agent_rnn

warnings.filterwarnings("ignore")

def main(
    class_rnn: type = None,
    model: str = None,
    data: str = None,
    save: bool = False,
    additional_inputs_data: List[str] = None,
    
    # generated dataset parameters
    participant_id: int = None,
    
    # sindy config
    rnn_modules: List[str] = None,
    control_parameters: List[str] = None,
    library_setup: Dict[str, list] = None,
    filter_setup: Dict[str, list] = None,
    dataprocessing_setup: Dict[str, list] = None,
    
    # sindy parameters
    optimizer_type: str = "SR3_L1",
    optimizer_threshold = 0.05,
    optimizer_alpha = 0.1,
    polynomial_degree = 2,
    n_trials_off_policy = 1024,
    n_sessions_off_policy = 1,
    n_trials_same_action_off_policy = 5,
    verbose = True,
    use_optuna = False,
    filter_bad_participants = False,  # Added parameter to control filtering
    pruning = False,
    train_test_ratio = 1.0,
    optuna_threshold = 0.03,
    optuna_n_trials = 50,
    
    # ground truth parameters
    beta_reward = 3.,
    alpha = 0.25,
    alpha_penalty = -1.,
    forget_rate = 0.,
    confirmation_bias = 0.,
    beta_choice = 0.,
    alpha_choice = 0.,
    alpha_counterfactual = 0.,
    parameter_variance = 0.,
    
    # environment parameters
    n_actions = 2,
    sigma = .2,
    counterfactual = False,
    
    get_loss: bool = False,
    analysis: bool = False,
    # reward_range: List[float] = [0, 1],
    ):

    # ---------------------------------------------------------------------------------------------------
    # SINDy-agent setup
    # ---------------------------------------------------------------------------------------------------
    
    # # tracked variables and control signals in the RNN
    # rnn_modules = ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen']
    # control_parameters = ['c_action', 'c_reward_chosen', 'c_value_reward', 'c_value_choice']
    
    # # library setup: 
    # # which terms are allowed as control inputs in each SINDy model
    # # key is the SINDy model name, value is a list of allowed control inputs from the list of control signals 
    # library_setup = {
    #     'x_learning_rate_reward': ['c_reward_chosen', 'c_value_reward', 'c_value_choice'],
    #     'x_value_reward_not_chosen': ['c_reward_chosen', 'c_value_choice'],
    #     'x_value_choice_chosen': ['c_value_reward'],
    #     'x_value_choice_not_chosen': ['c_value_reward'],
    # }

    # # data-filter setup: 
    # # which samples are allowed as training samples in each SINDy model based on the given filter condition (conditions are always equality conditions)
    # # key is the SINDy model name, value is a list with a triplet of values:
    # #   1. str: feature name to be used as a filter
    # #   2. numeric: the numeric filter condition
    # #   3. bool: remove feature from control inputs --> TODO: check if this is necessary or makes things just more complicated
    # # Multiple conditions can also be given as a list of triplets.
    # # Example:
    # #   'x_value_choice_not_chosen': ['c_action', 0, True] means that for the SINDy model 'x_value_choice_not_chosen', only samples where the feature 'c_action' == 0 are used for training the SINDy model. 
    # #   The control parameter 'c_action' is removed afterwards from the list of control signals for training of the model
    # filter_setup = {
    #     # 'x_value_reward_chosen': ['c_action', 1, True], -> Remove this one as well
    #     'x_learning_rate_reward': ['c_action', 1, True],
    #     'x_value_reward_not_chosen': ['c_action', 0, True],
    #     'x_value_choice_chosen': ['c_action', 1, True],
    #     'x_value_choice_not_chosen': ['c_action', 0, True],
    # }

    # # data pre-processing setup:
    # # define the processing steps for each variable and control signal.
    # # possible processing steps are: 
    # #   1. Trimming: Remove the first 25% of the samples along the time-axis. This is useful if the RNN begins with a variable at 0 but then accumulates first first to a specific default value, i.e. the range changes from (0, p) to (q, q+p). That way the data is cleared of the accumulation process. Trimming will be active for all variables, if it is active for one. 
    # #   2. Offset-Clearing: Clearup any offset by determining the minimal value q of a variable and move the value range from (q, q+p) -> (0, p). This step makes SINDy equations less complex and aligns them more with RL-Theory
    # #   3. Normalization: Scale the value range of a variable to x_max - x_min = 1. Offset-Clearing is recommended to achieve a value range of (0, 1) 
    # # The processing steps are passed in the form of a binary triplet in this order: (Trimming, Offset-Clearing, Normalization) 
    # dataprocessing_setup = {
    #     'x_learning_rate_reward': [0, 0, 0],
    #     'x_value_reward_not_chosen': [0, 0, 0],
    #     'x_value_choice_chosen': [1, 1, 0],
    #     'x_value_choice_not_chosen': [1, 1, 0],
    #     # 'c_action': [0, 0, 0],
    #     # 'c_reward': [0, 0, 0],
    #     'c_value_reward': [0, 0, 0],
    #     'c_value_choice': [1, 1, 0],
    # }
    
    sindy_feature_list = rnn_modules + control_parameters

    if not check_library_setup(library_setup, sindy_feature_list, verbose=True):
        raise ValueError('Library setup does not match feature list.')
        
    # ---------------------------------------------------------------------------------------------------
    # RNN Setup
    # ---------------------------------------------------------------------------------------------------
    
    # set up rnn agent and expose q-values to train sindy
    if model is None:
        file_rnn = parameter_file_naming('params/params', beta_reward=beta_reward, alpha_reward=alpha, alpha_penalty=alpha_penalty, beta_choice=beta_choice, alpha_choice=alpha_choice, forget_rate=forget_rate, confirmation_bias=confirmation_bias, alpha_counterfactual=alpha_counterfactual, variance=parameter_variance, verbose=True)
    else:
        file_rnn = model
    
    agent_rnn = setup_agent_rnn(
        class_rnn=class_rnn,
        path_model=file_rnn,
        list_sindy_signals=rnn_modules+control_parameters,
    )
    
    # ---------------------------------------------------------------------------------------------------
    # Data setup
    # ---------------------------------------------------------------------------------------------------
    
    agent = None
    if data is None:
        # set up ground truth agent and environment
        environment = BanditsDrift(sigma=sigma, n_actions=n_actions, counterfactual=counterfactual)
        agent = AgentQ(
            n_actions=n_actions, 
            beta_reward=beta_reward, 
            alpha_reward=alpha, 
            alpha_penalty=alpha_penalty, 
            beta_choice=beta_choice, 
            alpha_choice=alpha_choice, 
            forget_rate=forget_rate, 
            confirmation_bias=confirmation_bias, 
            alpha_counterfactual=alpha_counterfactual,
            )
        
        if participant_id is not None or participant_id != 0:
            participant_id = 0
        participant_ids = np.arange(agent_rnn._model.n_participants, dtype=int)
        dataset, _, _ = create_dataset_bandits(agent, environment, 200, len(participant_ids))
    else:
        # get data from experiments for later evaluation
        dataset, _, df, _ = convert_dataset(data, additional_inputs=additional_inputs_data)
        participant_ids = dataset.xs[..., -1].unique().int().cpu().numpy()

    choices = dataset.xs[..., 0] == 1
    reward_range = [dataset.xs[..., n_actions][choices].min(), dataset.xs[..., n_actions][choices].max()]
    print(reward_range)
    # ---------------------------------------------------------------------------------------------------
    # SINDy training
    # ---------------------------------------------------------------------------------------------------
    
    if use_optuna and verbose:
        print("\nUsing Optuna to find optimal optimizer configuration for each participant")
    
    # setup the SINDy-agent
    agent_spice, loss_spice = fit_spice(
        rnn_modules=rnn_modules,
        control_signals=control_parameters,
        agent_rnn=agent_rnn,
        data=dataset,
        n_sessions_off_policy=n_sessions_off_policy,
        n_trials_off_policy=n_trials_off_policy,
        n_trials_same_action_off_policy=n_trials_same_action_off_policy,
        polynomial_degree=polynomial_degree,
        library_setup=library_setup,
        filter_setup=filter_setup,
        dataprocessing=dataprocessing_setup,
        optimizer_type=optimizer_type,
        optimizer_threshold=optimizer_threshold,
        optimizer_alpha=optimizer_alpha,
        get_loss=get_loss,
        participant_id=participant_id,
        shuffle=False,
        verbose=verbose,
        use_optuna=use_optuna,
        filter_bad_participants=filter_bad_participants,
        pruning=pruning,
        train_test_ratio=train_test_ratio,
        optuna_threshold=optuna_threshold,
        optuna_n_trials=optuna_n_trials,
        )
    
    # If agent_spice is None, we couldn't fit the model, so return early
    if len(participant_ids) == 0:
        print("ERROR: Failed to fit SPICE model. Returning None.")
        return None, None, None
    
    # save spice modules
    if save:
        file_spice = file_rnn.split(os.path.sep)
        if 'rnn' in file_spice[-1]:
            file_spice[-1] = file_spice[-1].replace('rnn', 'spice')
        else:
            file_spice[-1] = 'spice_' + file_spice[-1]
        file_spice = os.path.join(*file_spice)
        save_spice(agent_spice=agent_spice, file=file_spice)
        print("Saved SPICE parameters to file " + file_spice)
    
    # ---------------------------------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------------------------------
    
    if analysis and len(participant_ids) > 0:
        participant_id_test = participant_id if participant_id is not None else participant_ids[0]
        mask_participant_id = dataset.xs[:, 0, -1] == participant_id_test
        experiment_test = dataset.xs[mask_participant_id][0]
        
        agent_rnn.new_sess(participant_id=participant_id_test)
        agent_spice.new_sess(participant_id=participant_id_test)
        
        # print sindy equations from tested sindy agent
        print('\nDiscovered SPICE models:')
        for module in rnn_modules:
            if participant_id_test in agent_spice._model.submodules_sindy[module]:
                agent_spice._model.submodules_sindy[module][participant_id_test].print()
            else:
                print(f"No module available for {module}, participant {participant_id_test}")
        print('\n')
        
        # set up ground truth agent by getting parameters from dataset if specified
        if data is not None and agent is None and analysis and 'mean_beta_reward' in df.columns:
            agent = AgentQ(
                beta_reward = df['beta_reward'].values[(df['session']==participant_id_test).values][0],
                alpha_reward = df['alpha_reward'].values[(df['session']==participant_id_test).values][0],
                alpha_penalty = df['alpha_penalty'].values[(df['session']==participant_id_test).values][0],
                confirmation_bias = df['confirmation_bias'].values[(df['session']==participant_id_test).values][0],
                forget_rate = df['forget_rate'].values[(df['session']==participant_id_test).values][0],
                beta_choice = df['beta_choice'].values[(df['session']==participant_id_test).values][0],
                alpha_choice = df['alpha_choice'].values[(df['session']==participant_id_test).values][0],
            )
        
        # get analysis plot
        if agent is not None:
            agents = {'groundtruth': agent, 'rnn': agent_rnn, 'sindy': agent_spice}
            plt_title = r'$GT:\beta_{reward}=$'+str(np.round(agent._beta_reward, 2)) + r'; $\beta_{choice}=$'+str(np.round(agent._beta_choice, 2))+'\n'
        else:
            agents = {'rnn': agent_rnn, 'sindy': agent_spice}
            plt_title = ''
        
        fig, axs = plot_session(agents, experiment_test, reward_range=reward_range)
        betas = agent_spice.get_betas()
        plt_title += r'SINDy: $\beta_{reward}=$'+str(np.round(betas['x_value_reward'], 2)) + r'; $\beta_{choice}=$'+str(np.round(betas['x_value_choice'], 2))
        
        fig.suptitle(plt_title)
        plt.show()
        # plt.savefig('example_recovered_dynamics.png', dpi=500)
    
    features = {}
    # for module in agent_spice._model.submodules_sindy:
    #     features[module] = {}
    #     for pid in agent_spice._model.submodules_sindy[module]:
    #         features_i = np.array(agent_spice._model.submodules_sindy[module][pid].get_feature_names())
    #         coeffs_i = agent_spice._model.submodules_sindy[module][pid].coefficients()[0]
    #         index_u = [not 'dummy' in f for f in features_i]
    #         features_i = features_i[index_u]
    #         coeffs_i = coeffs_i[index_u]
    #         # features[model][pid] = tuple(features_i)
    #         features[module][pid] = tuple(coeffs_i)
    
    # features['beta_reward'] = {}
    # features['beta_choice'] = {}
    # for pid in participant_ids:
    #     pid = int(pid)
    #     if pid in agent_spice._model.submodules_sindy[rnn_modules[0]]:
    #         agent_spice.new_sess(participant_id=pid, additional_embedding_inputs=)
    #         betas = agent_spice.get_betas()
    #         features['beta_reward'][pid] = betas['x_value_reward']
    #         features['beta_choice'][pid] = betas['x_value_choice']
        
    return agent_spice, features, loss_spice


if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SINDy pipeline with optimizer selection")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--data", type=str, default=None, help="Path to data file")
    parser.add_argument("--participant_id", type=int, default=None, help="Participant ID")
    parser.add_argument("--polynomial_degree", type=int, default=2, help="Polynomial degree")
    parser.add_argument("--optimizer_type", type=str, default="SR3_weighted_l1", choices=["STLSQ", "SR3_L1", "SR3_weighted_l1"], help="Optimizer type")
    parser.add_argument("--optimizer_alpha", type=float, default=0.1, help="Optimizer alpha")
    parser.add_argument("--optimizer_threshold", type=float, default=0.05, help="Optimizer threshold")
    parser.add_argument("--optuna_threshold", type=float, default=0.1, help="Threshold to use Optuna: Difference of action probabilities between RNN and discovered SPICE model")
    parser.add_argument("--optuna_n_trials", type=int, default=50, help="Number of Optuna trials to optimize SPICE fitting configuration")
    parser.add_argument("--use_optuna", action="store_true", help="Use Optuna to find the best optimizer for each participant")
    parser.add_argument("--filter_bad_participants", action="store_true", help="Remove badly fitted participants")
    parser.add_argument("--n_trials_off_policy", type=int, default=1000, help="Number of trials for off-policy data")
    parser.add_argument("--n_trials_same_action_off_policy", type=int, default=5, help="Number of same actions in a row for off-policy data")
    parser.add_argument("--n_sessions_off_policy", type=int, default=1, help="Number of sessions for off-policy data")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--analysis", action="store_true", help="Perform analysis")
    parser.add_argument("--get_loss", action="store_true", help="Compute loss")
    parser.add_argument("--save", action="store_true", help="Pickle SPICE parameters")
    parser.add_argument('--train_test_ratio', type=str, default="1.0", help='Ratio of training data; Can also be a comma-separated list of integeres to indicate testing sessions.')
    
    args = parser.parse_args()
    
    # convert train_test_ratio to number of list of numbers
    if ',' in args.train_test_ratio:
        train_test_ratio = [int(x) for x in args.train_test_ratio.split(',')]
    else:
        train_test_ratio = float(args.train_test_ratio)
    
    class_rnn = RLRNN_eckstein2022
    sindy_config = SindyConfig_eckstein2022
    
    agent_spice, features, loss = main(
        class_rnn=class_rnn,
        model=args.model,
        data=args.data,
        save=args.save,
        participant_id=args.participant_id,
        polynomial_degree=args.polynomial_degree,
        optimizer_type=args.optimizer_type,
        optimizer_alpha=args.optimizer_alpha,
        optimizer_threshold=args.optimizer_threshold,
        n_trials_off_policy=args.n_trials_off_policy,
        n_trials_same_action_off_policy=args.n_trials_same_action_off_policy,
        n_sessions_off_policy=args.n_sessions_off_policy,
        verbose=args.verbose,
        analysis=args.analysis,
        get_loss=args.get_loss,
        use_optuna=args.use_optuna,
        filter_bad_participants=args.filter_bad_participants,
        train_test_ratio=args.train_test_ratio,
        optuna_threshold=args.optuna_threshold,
        optuna_n_trials=args.optuna_n_trials,
        **sindy_config,
    )
    
    if loss is not None:
        print(f"Loss: {loss}")