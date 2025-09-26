import matplotlib.pyplot as plt

from spice.utils.convert_dataset import convert_dataset
from spice.utils.plotting import plot_session
from spice.estimator import SpiceEstimator
from spice_config import CONTRAST_CONFIG, RNN_ContrDiff


# path_data = 'ganesh2024a/data/GBSlider_ganesh2024a_xs_withRand.csv'
path_data = 'ganesh2024a/data/GBSlider_ganesh2024a_xs_withChosehighLow.csv'
path_model = 'ganesh2024a/params/rnn_ganesh2024a.pkl'

dataset = convert_dataset(
    file=path_data, 
    df_block='blocks',
    df_participant_id='subjID',
    df_choice='chose_high',
    additional_inputs=['contrast_difference'],
    timeshift_additional_inputs=True,
    )[0]

n_participants = len(dataset.xs[..., -1].unique())
test_blocks = [3, 6, 9]

estimator = SpiceEstimator(
    rnn_class=RNN_ContrDiff,
    spice_config=CONTRAST_CONFIG,
    n_participants=n_participants,
    epochs=512,  # Quick test
    bagging=True,
    learning_rate=1e-2,  # Standard learning rate
    scheduler=True,
    train_test_ratio=test_blocks,
    spice_library_polynomial_degree=2,
    use_optuna=False,  # Disable for faster testing
    save_path_rnn=path_model,
    save_path_spice=path_model.replace('rnn', 'spice'),
    n_sessions_off_policy=1,
)

# estimator.fit(dataset.xs, dataset.ys)
estimator.load_rnn_model(path_model)
estimator.load_spice_model(path_model.replace('rnn', 'spice'))

estimator.print_spice_model()

fig, axs = plot_session(
    agents={
        'rnn': estimator.rnn_agent,
        'spice': estimator.spice_agent,
        },
    experiment=dataset.xs[16],
    signals_to_plot=['x_value_reward', 'x_value_choice'],
    display_choice=1,
    )
plt.show()