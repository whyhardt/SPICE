import matplotlib.pyplot as plt
import numpy as np

from spice.utils.convert_dataset import convert_dataset
from spice.utils.plotting import plot_session
from spice.estimator import SpiceEstimator
from spice_config import CONTR_DIFF_CONFIG, RNN_ContrDiff


path_data = 'ganesh2024a/data/ganesh2024a_short.csv'
path_model = 'ganesh2024a/params/rnn_ganesh2024a.pkl'

dataset = convert_dataset(
    file=path_data, 
    df_block='blocks',
    df_choice='chose_right',
    df_reward='reward_right',
    df_participant_id='subjID',
    additional_inputs=['contrast_difference'],
    )[0]

n_participants = len(dataset.xs[..., -1].unique())
test_blocks = [3, 6, 9]

estimator = SpiceEstimator(
    rnn_class=RNN_ContrDiff,
    spice_config=CONTR_DIFF_CONFIG,
    n_participants=n_participants,
    epochs=1,
    bagging=True,
    learning_rate=1e-2,
    scheduler=True,
    train_test_ratio=test_blocks,
    spice_library_polynomial_degree=2,
    use_optuna=True,
    save_path_rnn=path_model,
    save_path_spice=path_model.replace('rnn', 'spice'),
    n_sessions_off_policy=0,
)

estimator.fit(dataset.xs, dataset.ys)

fig, axs = plot_session(
    agents={'rnn': estimator.rnn_agent, 'spice': estimator.spice_agent}, 
    experiment=dataset.xs[0],
    signals_to_plot=['x_value_reward', 'x_value_choice', 'x_value_state']
    )
plt.show()