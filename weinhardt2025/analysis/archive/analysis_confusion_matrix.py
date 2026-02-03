import sys, os

import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import matplotlib.colors as mcolors

from spice.utils.convert_dataset import csv_to_dataset
from spice.utils.agent import Agent, get_update_dynamics
from spice.resources.rnn import RLRNN_eckstein2022 as RLRNN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from weinhardt2025.utils.model_evaluation import log_likelihood, bayesian_information_criterion, akaike_information_criterion
from weinhardt2025.benchmarking import benchmarking_eckstein2022, benchmarking_dezfouli2019


#----------------------------------------------------------------------------------------------
# CONFIGURATION CONFUSION MATRIX FILES
#----------------------------------------------------------------------------------------------

# dataset settings
# study = "eckstein2022"
# models = ["ApBr", "ApAnBrBcfBch"]
# path_model_benchmark = f'params/{study}/mcmc_{study}_SIMULATED_FITTED.nc'

study = "dezfouli2019"
models = ["PhiBeta", "PhiChiBetaKappaC"]
path_model_benchmark = f'params/{study}/gql_{study}_SIMULATED_FITTED.pkl'
setup_agent_benchmark = benchmarking_dezfouli2019.setup_agent_gql
Dezfouli2019GQL = benchmarking_dezfouli2019.Dezfouli2019GQL

spice_model_name = "spice"
models.append(spice_model_name)

# file settings
path_data = f'data/{study}/{study}_simulated_SIMULATED_test.csv'
path_model_spice = f'params/{study}/rnn_{study}_SIMULATED.pkl'

n_actions = 2

#----------------------------------------------------------------------------------------------
# HELPER FUNCTION
#----------------------------------------------------------------------------------------------

# Create a truncated colormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

#----------------------------------------------------------------------------------------------
# SETUP MODELS
#----------------------------------------------------------------------------------------------

simulated_models = deepcopy(models)
if 'spice' in simulated_models:
    simulated_models.remove('spice')
    simulated_models.append('rnn')

print("Simulated models:", simulated_models)
print("Fitted models:", models)

agents = {}
for simulated_model in simulated_models:
    agents[simulated_model] = {}
    for fitted_model in models:
        agents[simulated_model][fitted_model] = None
        
        if fitted_model.lower() != spice_model_name:
            agent = setup_agent_benchmark(
                path_model=path_model_benchmark.replace('SIMULATED', simulated_model).replace('FITTED', fitted_model)
                )
            n_params = agent[1]
        else:
            if spice_model_name == "rnn":
                agent = setup_agent_rnn(
                    class_rnn=RLRNN, 
                    path_model=path_model_spice.replace('SIMULATED', simulated_model), 
                    list_sindy_signals=SindyConfig['rnn_modules']+SindyConfig['control_parameters'],
                    )
                n_params = 12.647059  # avg n params of eckstein2022-SPICE models
            elif spice_model_name == "spice":
                agent = setup_agent_spice(
                    class_rnn=RLRNN,
                    path_spice=path_model_spice.replace('SIMULATED', simulated_model).replace('rnn', 'spice', 1),
                    path_rnn=path_model_spice.replace('SIMULATED', simulated_model),
                    path_data=path_data.replace('SIMULATED', fitted_model),
                    rnn_modules=SindyConfig['rnn_modules'],
                    control_parameters=SindyConfig['control_parameters'],
                    sindy_library_setup=SindyConfig['library_setup'],
                    sindy_filter_setup=SindyConfig['filter_setup'],
                    sindy_dataprocessing=SindyConfig['dataprocessing_setup'],
                    sindy_library_polynomial_degree=1,
                )
                agent.new_sess()
                n_params = agent.count_parameters()
        agents[simulated_model][fitted_model] = (agent, n_params)


#----------------------------------------------------------------------------------------------
# PIPELINE CONFUSION MATRIX
#----------------------------------------------------------------------------------------------

metrics = ['nll', 'aic', 'bic']
confusion_matrix = {metric: np.zeros((len(models), len(models))) for metric in metrics}

for index_simulated_model, simulated_model in enumerate(simulated_models):
    
    # get data and choice probabilities from simulated model
    dataset = csv_to_dataset(file=path_data.replace('SIMULATED', simulated_model))
    n_sessions = len(dataset)
    metrics_session = {metric: np.zeros((n_sessions, len(models))) for metric in metrics}

    for index_fitted_model, fitted_model in enumerate(models):
        
        print(f"Comparing fitted model {fitted_model} to simulated data from {simulated_model}...")
        
        # agent setup for fitted model
        agent, n_parameters = agents[simulated_model][fitted_model]
        
        for session in tqdm(range(n_sessions)):
            # get choice probabilities from agent for data from simulated model
            choice_probs_fitted = get_update_dynamics(experiment=dataset.xs[session], agent=agent if isinstance(agent, AgentSpice) or isinstance(agent, Agent) else agent[0][session])[1]
            choices = dataset.xs[session, :len(choice_probs_fitted), :n_actions].cpu().numpy()
            
            if fitted_model == 'spice':
                n_params = n_parameters[session]
            else:
                n_params = n_parameters
                
            ll = log_likelihood(choices, choice_probs_fitted)
            
            metrics_session['nll'][session, index_fitted_model] = -ll
            metrics_session['bic'][session, index_fitted_model] = bayesian_information_criterion(choices, choice_probs_fitted, n_params, ll=ll)
            metrics_session['aic'][session, index_fitted_model] = akaike_information_criterion(choices, choice_probs_fitted, n_params, ll=ll)
    
    for metric in metrics:
        
        # get "best model"-counts for each model for each session
        best_model = np.argmin(metrics_session[metric], axis=-1)
        unique, counts = np.unique(best_model, return_counts=True)
        
        # Ensure all models have a count in the dictionary
        counts_dict = {model_index: 0 for model_index in range(len(models))}
        counts_dict.update(dict(zip(unique, counts / n_sessions)))
        
        confusion_matrix[metric][index_simulated_model] += np.array([counts_dict[key] for key in counts_dict])
    
# Plot confusion matrix

def plot_confusion_matrices(confusion_matrix, models, metrics, study, cmap):
    """Plot and save confusion matrices with proper figure management"""
    
    # Create annotated version
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6), constrained_layout=True)
    
    for index_metric, metric in enumerate(metrics):
        sns.heatmap(
            confusion_matrix[metric], 
            annot=True, 
            xticklabels=models, 
            yticklabels=models, 
            cmap=cmap,
            vmax=1, 
            vmin=0, 
            ax=axes[index_metric],
            cbar=True,
            fmt='.3f'  # Format annotation numbers
        )
        axes[index_metric].set_xlabel("Fitted Model")
        axes[index_metric].set_ylabel("Simulated Model")
        axes[index_metric].set_title("Confusion Matrix: " + metric.upper())
    
    # Save annotated version
    plt.savefig(f'analysis/plots_confusion_matrix/confusion_matrix_{study}_annotated.png', 
                dpi=500, bbox_inches='tight')
    plt.close(fig)  # Close the figure explicitly
    
    # Create non-annotated version
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6), constrained_layout=True)
    
    for index_metric, metric in enumerate(metrics):
        sns.heatmap(
            confusion_matrix[metric], 
            annot=False, 
            xticklabels=models, 
            yticklabels=models, 
            cmap=cmap,
            vmax=1, 
            vmin=0, 
            ax=axes[index_metric],
            cbar=True
        )
        axes[index_metric].set_xlabel("Fitted Model")
        axes[index_metric].set_ylabel("Simulated Model")
        axes[index_metric].set_title("Confusion Matrix: " + metric.upper())
    
    # Save non-annotated version
    plt.savefig(f'analysis/plots_confusion_matrix/confusion_matrix_{study}_not_annotated.png', 
                dpi=500, bbox_inches='tight')
    plt.close(fig)  # Close the figure explicitly

# Call the function
cmap = truncate_colormap(plt.cm.viridis, minval=0.5, maxval=1.0)
plot_confusion_matrices(confusion_matrix, models, metrics, study, cmap)

print(f"Confusion matrices saved successfully for {study}")