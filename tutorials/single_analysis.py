import sys, os
import numpy as np
import torch
from tqdm import tqdm

# --- Repo path ---
repo_root = r"C:\Users\Malte\Desktop\SPICE"
if repo_root not in sys.path:
    sys.path.append(repo_root)

from spice.resources.old_rnn import RLRNN_dezfouli2019, RLRNN_eckstein2022
from spice.utils.convert_dataset import convert_dataset
from spice.utils.setup_agents import setup_agent_rnn, setup_agent_spice
from spice.resources.model_evaluation import get_scores
from spice.resources.bandits import get_update_dynamics
from spice.resources.rnn_utils import split_data_along_timedim, split_data_along_sessiondim

def evaluate_model(path_model_rnn, use_test=True):
    # --- Infer dataset from path ---
    dataset_name = "eckstein2022" if "eckstein" in path_model_rnn else "dezfouli2019"
    path_data = os.path.join(repo_root, 'data', dataset_name, f'{dataset_name}.csv')
    
    # Dataset-specific settings
    train_test_ratio = 0.8 if dataset_name == 'eckstein2022' else [3, 6, 9]
    class_rnn = RLRNN_eckstein2022 if dataset_name == 'eckstein2022' else RLRNN_dezfouli2019
    
    # Load dataset
    dataset_full = convert_dataset(path_data)[0]
    participant_ids = dataset_full.xs[:, 0, -1].unique().cpu().numpy()
    if isinstance(train_test_ratio, float):
        dataset_train, dataset_test = split_data_along_timedim(dataset_full, split_ratio=train_test_ratio)
    else:
        dataset_train, dataset_test = split_data_along_sessiondim(dataset_full, list_test_sessions=train_test_ratio)
        if not use_test:
            dataset_test = dataset_train
    
    data_input = dataset_test.xs
    
    # Load agents
    agent_rnn = setup_agent_rnn(class_rnn=class_rnn, path_model=path_model_rnn)
    path_model_spice = path_model_rnn.replace('_rnn.pkl', '_spice.pkl')
    agent_spice = setup_agent_spice(class_rnn=class_rnn, path_rnn=path_model_rnn, path_spice=path_model_spice) \
                  if os.path.exists(path_model_spice) else None
    
    # Prepare data slices
    n_actions_rnn = agent_rnn._n_actions
    data_test_rnn = dataset_test.xs[..., :n_actions_rnn]
    
    nll_rnn_total, nll_spice_total = 0.0, 0.0
    considered_trials_rnn, considered_trials_spice = 0, 0
    
    for i in tqdm(range(len(dataset_test))):
        pid = dataset_test.xs[i, 0, -1].int().item()
        if pid not in participant_ids:
            continue
        
        # --- RNN ---
        probs_rnn = get_update_dynamics(data_input[i], agent_rnn)[1]
        n_trials = len(probs_rnn)
        data_ys = data_test_rnn[i, :n_trials].cpu().numpy()
        scores_rnn = get_scores(data_ys, probs_rnn,
                                n_parameters=sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad))
        nll_rnn_total += scores_rnn[0]
        considered_trials_rnn += n_trials
        
        # --- SPICE ---
        if agent_spice:
            try:
                agent_spice.new_sess(participant_id=pid)  # handle embeddings internally
                probs_spice = get_update_dynamics(data_input[i], agent_spice)[1]
                if len(probs_spice) == 0:
                    continue
                data_ys_spice = data_test_rnn[i, :len(probs_spice)].cpu().numpy()
                scores_spice = get_scores(data_ys_spice, probs_spice,
                                          n_parameters=agent_spice.count_parameters().get(pid, 0))
                nll_spice_total += scores_spice[0]
                considered_trials_spice += len(probs_spice)
            except Exception:
                continue
    
    avg_trial_likelihood_rnn = np.exp(-nll_rnn_total / considered_trials_rnn) if considered_trials_rnn > 0 else np.nan
    avg_trial_likelihood_spice = np.exp(-nll_spice_total / considered_trials_spice) if considered_trials_spice > 0 else np.nan
    
    print(f"\nDataset: {dataset_name}")
    print(f"RNN avg trial likelihood: {avg_trial_likelihood_rnn:.4f}")
    print(f"SPICE avg trial likelihood: {avg_trial_likelihood_spice:.4f}")


# --- Example usage ---
rnn_path = r"C:\Users\Malte\Desktop\SPICE\tutorials\params\eckstein2022\iMAML_eckstein2022_ep8192_metalr-0_1000_in-0_0001_rnn.pkl"
evaluate_model(rnn_path)
