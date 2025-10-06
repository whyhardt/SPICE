import sys, os, warnings
import numpy as np
import torch
import pandas as pd
from torch.backends import cudnn
from tqdm import tqdm

warnings.filterwarnings("ignore")
cudnn.enabled = False

# --- Repo path ---
repo_root = r"C:\Users\alleebels\Desktop\malte-uni\SPICE"
if repo_root not in sys.path:
    sys.path.append(repo_root)

from spice import pipeline_rnn_autoreg, pipeline_sindy
from spice.resources.old_rnn import RLRNN_dezfouli2019, RLRNN_eckstein2022
from spice.resources import sindy_utils
from spice.utils.convert_dataset import convert_dataset
from spice.utils.setup_agents import setup_agent_rnn, setup_agent_spice
from spice.resources.model_evaluation import get_scores
from spice.resources.bandits import get_update_dynamics
from spice.resources.rnn_utils import split_data_along_timedim, split_data_along_sessiondim

# --- Seeds ---
np.random.seed(186)
torch.manual_seed(186)

# --- Configs ---
datasets = ["eckstein2022", "dezfouli2019"]
epochs = 8192  # For testing; use 8192 later

lambda_awd_list = [0.022, 0.1, 0.22, 0.5, 1, 5]
initial_reg_params = [0.0001, 0.001]
outer_lrs = [0.1, 0.01]

metaopt_order = ["awd", "imaml"]
use_test = True  # Match original script's behavior

# --- Helper for filenames ---
def float_to_str(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "_")

# --- Main loop ---
for metaopt_type in metaopt_order:
    print(f"\n=== Starting metaopt type: {metaopt_type} ===")
    for dataset in datasets:
        print(f"\n--- Dataset: {dataset} ---")
        path_data = os.path.join(repo_root, 'data', dataset, f'{dataset}.csv')
        additional_inputs = None

        train_test_ratio = 0.8 if dataset == 'eckstein2022' else [3, 6, 9]
        class_rnn = RLRNN_eckstein2022 if dataset == 'eckstein2022' else RLRNN_dezfouli2019
        sindy_config = sindy_utils.SindyConfig_eckstein2022 if dataset == 'eckstein2022' else sindy_utils.SindyConfig_dezfouli2019

        os.makedirs(f'params/{dataset}', exist_ok=True)
        summary_file = f'params/{dataset}/results_summary.csv'
        results_summary = pd.read_csv(summary_file).to_dict('records') if os.path.exists(summary_file) else []

        # --- AWD loop ---
        if metaopt_type == 'awd':
            for lambda_awd in lambda_awd_list:
                lam_str = float_to_str(lambda_awd)
                path_model_rnn = f'params/{dataset}/AWD_{dataset}_ep{epochs}_lawd-{lam_str}_rnn.pkl'
                path_model_spice = path_model_rnn.replace('_rnn.pkl', '_spice.pkl')

                if any(r.get('lambda_awd') == lambda_awd for r in results_summary if r['metaopt_type'] == 'awd'):
                    print(f"AWD λ={lambda_awd} already done, skipping...")
                    continue

                print(f"Training AWD λ={lambda_awd} on {dataset}...")
                try:
                    model, test_loss, histories = pipeline_rnn_autoreg.main(
                        dropout=0.25,
                        train_test_ratio=train_test_ratio,
                        checkpoint=False,
                        epochs=epochs,
                        scheduler=True,
                        learning_rate=1e-2,
                        metaopt_type='awd',
                        lambda_awd=lambda_awd,
                        meta_update_interval=50,
                        inner_steps=3,
                        outer_lr=None,
                        hypergradient_steps=3,
                        initial_reg_param=None,
                        n_steps=-1,
                        embedding_size=32,
                        batch_size=-1,
                        sequence_length=-1,
                        bagging=True,
                        class_rnn=class_rnn,
                        model=path_model_rnn,
                        data=path_data,
                        additional_inputs_data=additional_inputs,
                        n_sessions=128,
                        n_trials=200,
                        sigma=0.2,
                        beta_reward=3.,
                        alpha_reward=0.25,
                        alpha_penalty=0.5,
                        forget_rate=0.,
                        confirmation_bias=0.,
                        beta_choice=0.,
                        alpha_choice=1.,
                        counterfactual=False,
                        alpha_counterfactual=0.,
                        save_checkpoints=True,
                        analysis=False,
                        participant_id=0
                    )
                except Exception as e:
                    print(f"AWD λ={lambda_awd} failed on {dataset}: {e}")
                    continue

                last_train_loss, last_val_loss = histories[0][-1], histories[1][-1]

                # --- Run SINDy ---
                sindy_status = 'success'
                try:
                    pipeline_sindy.main(
                        class_rnn=class_rnn,
                        model=path_model_rnn,
                        data=path_data,
                        additional_inputs_data=additional_inputs,
                        save=True,
                        participant_id=None,
                        filter_bad_participants=False,
                        use_optuna=True,
                        pruning=False,
                        train_test_ratio=train_test_ratio,
                        polynomial_degree=3,
                        optimizer_alpha=0.1,
                        optimizer_threshold=0.05,
                        n_trials_off_policy=1000,
                        n_sessions_off_policy=1,
                        n_trials_same_action_off_policy=5,
                        optuna_threshold=0.1,
                        optuna_n_trials=50,
                        optimizer_type='SR3_weighted_l1',
                        verbose=False,
                        analysis=False,
                        get_loss=False,
                        **sindy_config
                    )
                except Exception as e:
                    print(f"SINDy failed for AWD λ={lambda_awd} on {dataset}: {e}")
                    sindy_status = 'failed'

                # --- Evaluation ---
                dataset_full = convert_dataset(path_data, additional_inputs=None)[0]
                participant_ids = dataset_full.xs[:, 0, -1].unique().cpu().numpy()
                if isinstance(train_test_ratio, float):
                    dataset_train, dataset_test = split_data_along_timedim(dataset_full, split_ratio=train_test_ratio)
                else:
                    dataset_train, dataset_test = split_data_along_sessiondim(dataset_full, list_test_sessions=train_test_ratio)
                    if not use_test:
                        dataset_test = dataset_train

                data_input = dataset_test.xs
                agent_rnn = setup_agent_rnn(class_rnn=class_rnn, path_model=path_model_rnn)
                agent_spice = setup_agent_spice(class_rnn=class_rnn, path_rnn=path_model_rnn, path_spice=path_model_spice) \
                              if sindy_status == 'success' and os.path.exists(path_model_spice) else None

                # Dynamic slicing
                n_actions_rnn = agent_rnn._n_actions
                data_test_rnn = dataset_test.xs[..., :n_actions_rnn]
                data_test_spice = None
                if agent_spice:
                    n_actions_spice = agent_spice._n_actions
                    data_test_spice = dataset_test.xs[..., :n_actions_spice]

                nll_rnn_total, nll_spice_total, considered_trials = 0.0, 0.0, 0
                for i in range(len(dataset_test)):
                    try:
                        pid = dataset_test.xs[i, 0, -1].int().item()
                        if pid not in participant_ids:
                            continue

                        # RNN
                        probs_rnn = get_update_dynamics(data_input[i], agent_rnn)[1]
                        n_trials = len(probs_rnn)
                        data_ys = data_test_rnn[i, :n_trials].cpu().numpy()
                        index_start, index_end = 0, n_trials
                        scores_rnn = get_scores(data_ys[index_start:index_end], probs_rnn[index_start:index_end],
                                                n_parameters=sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad))
                        nll_rnn_total += scores_rnn[0]

                        # SPICE
                        if agent_spice:
                            additional_inputs_embedding = data_input[0, agent_spice._n_actions*2:-3]
                            agent_spice.new_sess(participant_id=pid, additional_embedding_inputs=additional_inputs_embedding)
                            probs_spice = get_update_dynamics(data_input[i], agent_spice)[1]
                            data_ys_spice = data_test_spice[i, :len(probs_spice)].cpu().numpy()
                            scores_spice = get_scores(data_ys_spice[index_start:index_end], probs_spice[index_start:index_end],
                                                      n_parameters=agent_spice.count_parameters()[pid])
                            nll_spice_total += scores_spice[0]

                        considered_trials += (index_end - index_start)
                    except Exception as e:
                        print(f"Skipping participant {i} due to error: {e}")
                        continue

                nll_rnn_normalized = nll_rnn_total / considered_trials
                nll_spice_normalized = nll_spice_total / considered_trials if agent_spice else np.nan
                avg_trial_likelihood_rnn = np.exp(-nll_rnn_normalized)
                avg_trial_likelihood_spice = np.exp(-nll_spice_normalized) if agent_spice else np.nan

                results_summary.append({
                    'metaopt_type': 'awd',
                    'dataset': dataset,
                    'lambda_awd': lambda_awd,
                    'initial_reg_param': None,
                    'outer_lr': None,
                    'last_train_loss': last_train_loss,
                    'last_val_loss': last_val_loss,
                    'test_loss': test_loss,
                    'sindy_status': sindy_status,
                    'avg_trial_likelihood_rnn': avg_trial_likelihood_rnn,
                    'avg_trial_likelihood_spice': avg_trial_likelihood_spice
                })
                pd.DataFrame(results_summary).to_csv(summary_file, index=False)

        # --- iMAML loop ---
        elif metaopt_type == 'imaml':
            for initial_reg_param in initial_reg_params:
                for outer_lr in outer_lrs:
                    reg_str, lr_str = float_to_str(initial_reg_param), float_to_str(outer_lr)
                    path_model_rnn = f'params/{dataset}/iMAML_{dataset}_ep{epochs}_metalr-{lr_str}_in-{reg_str}_rnn.pkl'
                    path_model_spice = path_model_rnn.replace('_rnn.pkl', '_spice.pkl')

                    if any(r.get('initial_reg_param') == initial_reg_param and r.get('outer_lr') == outer_lr for r in results_summary if r['metaopt_type'] == 'imaml'):
                        print(f"iMAML reg={initial_reg_param}, lr={outer_lr} already done, skipping...")
                        continue

                    print(f"Training iMAML reg={initial_reg_param}, lr={outer_lr} on {dataset}...")
                    try:
                        model, test_loss, histories = pipeline_rnn_autoreg.main(
                            dropout=0.25,
                            train_test_ratio=train_test_ratio,
                            checkpoint=False,
                            epochs=epochs,
                            scheduler=True,
                            learning_rate=1e-2,
                            metaopt_type='imaml',
                            lambda_awd=None,
                            meta_update_interval=50,
                            inner_steps=3,
                            outer_lr=outer_lr,
                            hypergradient_steps=3,
                            initial_reg_param=initial_reg_param,
                            n_steps=-1,
                            embedding_size=32,
                            batch_size=-1,
                            sequence_length=-1,
                            bagging=True,
                            class_rnn=class_rnn,
                            model=path_model_rnn,
                            data=path_data,
                            additional_inputs_data=additional_inputs,
                            n_sessions=128,
                            n_trials=200,
                            sigma=0.2,
                            beta_reward=3.,
                            alpha_reward=0.25,
                            alpha_penalty=0.5,
                            forget_rate=0.,
                            confirmation_bias=0.,
                            beta_choice=0.,
                            alpha_choice=1.,
                            counterfactual=False,
                            alpha_counterfactual=0.,
                            save_checkpoints=True,
                            analysis=False,
                            participant_id=0
                        )
                    except Exception as e:
                        print(f"iMAML reg={initial_reg_param}, lr={outer_lr} failed on {dataset}: {e}")
                        continue

                    last_train_loss, last_val_loss = histories[0][-1], histories[1][-1]

                    # --- Run SINDy ---
                    sindy_status = 'success'
                    try:
                        pipeline_sindy.main(
                            class_rnn=class_rnn,
                            model=path_model_rnn,
                            data=path_data,
                            additional_inputs_data=additional_inputs,
                            save=True,
                            participant_id=None,
                            filter_bad_participants=False,
                            use_optuna=True,
                            pruning=False,
                            train_test_ratio=train_test_ratio,
                            polynomial_degree=3,
                            optimizer_alpha=0.1,
                            optimizer_threshold=0.05,
                            n_trials_off_policy=1000,
                            n_sessions_off_policy=1,
                            n_trials_same_action_off_policy=5,
                            optuna_threshold=0.1,
                            optuna_n_trials=50,
                            optimizer_type='SR3_weighted_l1',
                            verbose=False,
                            analysis=False,
                            get_loss=False,
                            **sindy_config
                        )
                    except Exception as e:
                        print(f"SINDy failed for iMAML reg={initial_reg_param}, lr={outer_lr} on {dataset}: {e}")
                        sindy_status = 'failed'

                    # --- Evaluation ---
                    dataset_full = convert_dataset(path_data, additional_inputs=None)[0]
                    participant_ids = dataset_full.xs[:, 0, -1].unique().cpu().numpy()
                    if isinstance(train_test_ratio, float):
                        dataset_train, dataset_test = split_data_along_timedim(dataset_full, split_ratio=train_test_ratio)
                    else:
                        dataset_train, dataset_test = split_data_along_sessiondim(dataset_full, list_test_sessions=train_test_ratio)
                        if not use_test:
                            dataset_test = dataset_train

                    data_input = dataset_test.xs
                    agent_rnn = setup_agent_rnn(class_rnn=class_rnn, path_model=path_model_rnn)
                    agent_spice = setup_agent_spice(class_rnn=class_rnn, path_rnn=path_model_rnn, path_spice=path_model_spice) \
                                  if sindy_status == 'success' and os.path.exists(path_model_spice) else None

                    n_actions_rnn = agent_rnn._n_actions
                    data_test_rnn = dataset_test.xs[..., :n_actions_rnn]
                    data_test_spice = None
                    if agent_spice:
                        n_actions_spice = agent_spice._n_actions
                        data_test_spice = dataset_test.xs[..., :n_actions_spice]

                    nll_rnn_total, nll_spice_total, considered_trials = 0.0, 0.0, 0
                    for i in range(len(dataset_test)):
                        try:
                            pid = dataset_test.xs[i, 0, -1].int().item()
                            if pid not in participant_ids:
                                continue

                            # RNN
                            probs_rnn = get_update_dynamics(data_input[i], agent_rnn)[1]
                            n_trials = len(probs_rnn)
                            data_ys = data_test_rnn[i, :n_trials].cpu().numpy()
                            index_start, index_end = 0, n_trials
                            scores_rnn = get_scores(data_ys[index_start:index_end], probs_rnn[index_start:index_end],
                                                    n_parameters=sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad))
                            nll_rnn_total += scores_rnn[0]

                            # SPICE
                            if agent_spice:
                                additional_inputs_embedding = data_input[0, agent_spice._n_actions*2:-3]
                                agent_spice.new_sess(participant_id=pid, additional_embedding_inputs=additional_inputs_embedding)
                                probs_spice = get_update_dynamics(data_input[i], agent_spice)[1]
                                data_ys_spice = data_test_spice[i, :len(probs_spice)].cpu().numpy()
                                scores_spice = get_scores(data_ys_spice[index_start:index_end], probs_spice[index_start:index_end],
                                                          n_parameters=agent_spice.count_parameters()[pid])
                                nll_spice_total += scores_spice[0]

                            considered_trials += (index_end - index_start)
                        except Exception as e:
                            print(f"Skipping participant {i} due to error: {e}")
                            continue

                    nll_rnn_normalized = nll_rnn_total / considered_trials
                    nll_spice_normalized = nll_spice_total / considered_trials if agent_spice else np.nan
                    avg_trial_likelihood_rnn = np.exp(-nll_rnn_normalized)
                    avg_trial_likelihood_spice = np.exp(-nll_spice_normalized) if agent_spice else np.nan

                    results_summary.append({
                        'metaopt_type': 'imaml',
                        'dataset': dataset,
                        'lambda_awd': None,
                        'initial_reg_param': initial_reg_param,
                        'outer_lr': outer_lr,
                        'last_train_loss': last_train_loss,
                        'last_val_loss': last_val_loss,
                        'test_loss': test_loss,
                        'sindy_status': sindy_status,
                        'avg_trial_likelihood_rnn': avg_trial_likelihood_rnn,
                        'avg_trial_likelihood_spice': avg_trial_likelihood_spice
                    })
                    pd.DataFrame(results_summary).to_csv(summary_file, index=False)

print("\n=== Hyperparameter search complete ===")
