import sys, os
import torch
import matplotlib.pyplot as plt

from spice.utils.plotting import plot_session
from spice.utils.convert_dataset import convert_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from weinhardt2025.benchmarking.benchmarking_lstm import RLLSTM, AgentLSTM

path_data = 'ganesh2024a/data/ganesh2024a_merged_rewards_rand.csv'
path_lstm = 'ganesh2024a/params/lstm_ganesh2024a.pkl'
state_dict = torch.load(path_lstm,weights_only=True)

n_actions = 2
n_cells = 4
model = RLLSTM(n_cells, n_actions, additional_inputs=1)
model.load_state_dict(state_dict)

agent = AgentLSTM(model_rnn=model)

dataset = convert_dataset(
    file=path_data,
    df_participant_id='subjID',
    df_choice='chose_right',
    additional_inputs=['contrast_difference'],
)[0]

fig,axs = plot_session({'rnn': agent}, experiment=dataset.xs[0], signals_to_plot=[])
plt.show()