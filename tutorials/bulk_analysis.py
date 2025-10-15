import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from spice.utils.convert_dataset import convert_dataset
from spice.utils.setup_agents import setup_agent_rnn, setup_agent_spice
from spice.resources.model_evaluation import get_scores
from spice.resources.bandits import get_update_dynamics
from spice.resources.rnn_utils import split_data_along_timedim, split_data_along_sessiondim
from spice.resources.old_rnn import RLRNN_dezfouli2019, RLRNN_eckstein2022


# === Dataset-specific configs ===
RNN_CLASSES = {
    "dezfouli2019": RLRNN_dezfouli2019,
    "eckstein2022": RLRNN_eckstein2022
}

TRAIN_TEST_RATIOS = {
    "dezfouli2019": [3, 6, 9],
    "eckstein2022": 0.8
}


def evaluate_models(model_paths, output_dir="evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for rnn_path in tqdm(model_paths, desc="Evaluating models"):
        dataset_name = "dezfouli2019" if "dezfouli2019" in rnn_path else "eckstein2022"
        class_rnn = RNN_CLASSES[dataset_name]
        spice_path = rnn_path.replace("_rnn.pkl", "_spice.pkl")

        if not os.path.exists(rnn_path):
            print(f"RNN model not found: {rnn_path}, skipping...")
            continue
        if not os.path.exists(spice_path):
            spice_path = None

        # === Load dataset ===
        dataset_full = convert_dataset(f"data/{dataset_name}/{dataset_name}.csv", additional_inputs=None)[0]
        participant_ids = dataset_full.xs[:, 0, -1].unique().cpu().numpy()

        # === Train/test split ===
        ratio = TRAIN_TEST_RATIOS[dataset_name]
        if isinstance(ratio, float):
            _, dataset_test = split_data_along_timedim(dataset_full, split_ratio=ratio)
        else:
            _, dataset_test = split_data_along_sessiondim(dataset_full, list_test_sessions=ratio)

        data_input = dataset_test.xs

        # === Setup agents ===
        agent_rnn = setup_agent_rnn(class_rnn=class_rnn, path_model=rnn_path)
        agent_spice = setup_agent_spice(class_rnn=class_rnn, path_rnn=rnn_path, path_spice=spice_path) if spice_path else None

        n_actions_rnn = agent_rnn._n_actions
        data_test_rnn = dataset_test.xs[..., :n_actions_rnn]
        data_test_spice = dataset_test.xs[..., :agent_spice._n_actions] if agent_spice else None

        # === Evaluation ===
        nll_rnn_total, nll_spice_total, considered_trials = 0.0, 0.0, 0
        for i in range(len(dataset_test)):
            try:
                pid = dataset_test.xs[i, 0, -1].int().item()
                if pid not in participant_ids:
                    continue

                # --- RNN evaluation ---
                probs_rnn = get_update_dynamics(data_input[i], agent_rnn)[1]
                n_trials = len(probs_rnn)
                data_ys = data_test_rnn[i, :n_trials].cpu().numpy()
                scores_rnn = get_scores(
                    data_ys, probs_rnn,
                    n_parameters=sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad)
                )
                nll_rnn_total += scores_rnn[0]

                # --- SPICE evaluation (matches original script exactly) ---
                if agent_spice:
                    additional_inputs_embedding = data_input[0, agent_spice._n_actions*2:-3]
                    agent_spice.new_sess(participant_id=pid, additional_embedding_inputs=additional_inputs_embedding)
                    probs_spice = get_update_dynamics(data_input[i], agent_spice)[1]
                    data_ys_spice = data_test_spice[i, :len(probs_spice)].cpu().numpy()
                    scores_spice = get_scores(
                        data_ys_spice, probs_spice,
                        n_parameters=agent_spice.count_parameters()[pid]
                    )
                    nll_spice_total += scores_spice[0]

                considered_trials += n_trials

            except Exception as e:
                print(f"Skipping participant {i} due to error: {e}")
                continue

        avg_trial_likelihood_rnn = np.exp(-nll_rnn_total / considered_trials)
        avg_trial_likelihood_spice = (
            np.exp(-nll_spice_total / considered_trials) if agent_spice else np.nan
        )

        results.append({
            "dataset": dataset_name,
            "rnn_model": rnn_path,
            "spice_model": spice_path,
            "avg_trial_likelihood_rnn": avg_trial_likelihood_rnn,
            "avg_trial_likelihood_spice": avg_trial_likelihood_spice
        })

    # === Save results ===
    df = pd.DataFrame(results)
    for dataset_name, df_subset in df.groupby("dataset"):
        csv_path = os.path.join(output_dir, f"{dataset_name}_evaluation.csv")
        df_subset.to_csv(csv_path, index=False)
        print(f"Saved evaluation for {dataset_name} to {csv_path}")

    return df


if __name__ == "__main__":
    model_paths = [
        "tutorials/params/dezfouli2019/iMAML_dezfouli2019_ep8192_metalr-1e-2_in-1e-4_rnn.pkl",
        "tutorials/params/eckstein2022/iMAML_eckstein2022_ep8192_metalr-1e-1_in-1e-4_rnn.pkl",
    ]

    df = evaluate_models(model_paths)
