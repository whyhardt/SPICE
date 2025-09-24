import sys, os
import torch
import matplotlib.pyplot as plt

from spice.utils.plotting import plot_session
from spice.utils.convert_dataset import convert_dataset
from spice.resources.rnn_utils import split_data_along_sessiondim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from weinhardt2025.benchmarking.benchmarking_lstm import RLLSTM, AgentLSTM

path_data = 'ganesh2024a/data/GBSlider_ganesh2024a_xs_withRand.csv'
path_lstm = 'ganesh2024a/params/lstm_ganesh2024a.pkl'
state_dict = torch.load(path_lstm,weights_only=True)

n_actions = 2
n_cells = 8
device = torch.device('cpu')
model = RLLSTM(n_cells, n_actions, additional_inputs=1)
model.load_state_dict(state_dict)

agent = AgentLSTM(model_rnn=model)

dataset = convert_dataset(
    file=path_data,
    df_participant_id='subjID',
    df_choice='choice',
    df_reward='reward',
    df_block='blocks',
    additional_inputs=['contrast_difference'],
    timeshift_additional_inputs=True,
)[0]

dataset_training, dataset_test = split_data_along_sessiondim(dataset=dataset, list_test_sessions=[2,4,6])

fig,axs = plot_session({'rnn': agent}, experiment=dataset_training.xs[0], signals_to_plot=[], display_choice=0)
plt.show()

fig,axs = plot_session({'rnn': agent}, experiment=dataset_training.xs[0], signals_to_plot=[], display_choice=1)
plt.show()