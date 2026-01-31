"""
Test script for end-to-end differentiable SPICE training.
This runs the full pipeline with SINDy regularization during RNN training.
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from spice import SpiceEstimator, csv_to_dataset, split_data_along_sessiondim, split_data_along_timedim, plot_session, Agent
from spice.precoded import workingmemory_multiitem, workingmemory, workingmemory_rewardbinary, choice
from spice.resources.spice_training import _get_terminal_width

from benchmarking.benchmarking_qlearning import QLearning


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Trains a SPICE-RNN with end-to-end differentiable SINDy on behavioral data.')

    # necessary parameters
    parser.add_argument('--model', type=str, default=None, help='Model name to load from and/or save to parameters of RNN')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset')
    
    # RNN training parameters
    parser.add_argument('--epochs', type=int, default=4000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--rnn_l2_lambda', type=float, default=0., help='L2 Reg of the RNN parameters')
    
    # SINDy training parameters
    parser.add_argument('--sindy_weight', type=float, default=0.1, help='Weight for SINDy regularization during RNN training')
    parser.add_argument('--sindy_l2_lambda', type=float, default=0.0001, help='L2 Reg of the SINDy coefficients')
    parser.add_argument('--sindy_pruning_threshold', type=float, default=0.05, help='Threshold value for cutting off sindy terms (lowered for delta-form coefficients)')
    parser.add_argument('--sindy_pruning_terms', type=int, default=1, help='Number of thresholded terms')
    parser.add_argument('--sindy_pruning_freq', type=int, default=1, help='Number of epochs after which to prune')
    parser.add_argument('--sindy_pruning_patience', type=int, default=100, help='Number of epochs after which to prune')
    parser.add_argument('--sindy_confidence', type=float, default=0.1, help='Threshold used for confidence-based pruning across models (participants x experiments)')
    
    # Data setup parameters
    parser.add_argument('--train_ratio_time', type=float, default=None, help='Ratio of data used for training. Split along time dimension. Not combinable with test_sessions')
    parser.add_argument('--test_sessions', type=str, default=None, help='Comma-separated list of integeres which indicate test sessions. Not combinable with train_ratio_time')
    parser.add_argument('--n_items', type=int, default=None, help='Number of items in dataset; Default None: As many items as actions (automatically detected from dataset);')
    parser.add_argument('--additional_columns', type=str, default=None, help='Comma-separated list of columns which are added to the dataset.')
    parser.add_argument('--timeshift_additional_columns', action='store_true', help='Shifts additional columns (defined by the kwarg "additional_columns") [t]->[t-1]; Necessary for e.g. predictor stimuli which are usually listed in the trial of which SPICE has to predict the action.')

    parser.add_argument('--results', action='store_true', help='Shows the results using a fitted SPICE model. The results are value-dynamics-over-time plot, a parameter distribution histogram, and the corresponding symbolic SPICE model.')
    
    args = parser.parse_args()
    
    # args.sindy_weight = 0
    
    # args.model = "weinhardt2025/params/eckstein2022/spice_eckstein2022.pkl"
    # args.data = "weinhardt2025/data/eckstein2022/eckstein2022.csv"
    # args.train_ratio_time = 0.8
    # include_validation = False
    
    # args.model = "weinhardt2025/params/eckstein2024/spice_eckstein2024.pkl"
    # args.data = "weinhardt2025/data/eckstein2024/eckstein2024.csv"
    # args.test_sessions = "1,3"
    
    # args.results = True
    # args.epochs = 10
    # args.model = "weinhardt2025/params/dezfouli2019/spice_dezfouli2019.pkl"
    # args.data = "weinhardt2025/data/dezfouli2019/dezfouli2019.csv"
    # args.test_sessions = "3,6,9"
    
    # args.data="weinhardt2025/data/sugawara2021/sugawara2021.csv" 
    # args.model="weinhardt2025/params/sugawara2021/spice_sugawara2021.pkl" 
    # args.additional_columns="shown_at_0,shown_at_1,shown_at_0_next,shown_at_1_next"
    # args.n_items=8
    # args.test_sessions="1"
    
    # args.data = "weinhardt2025/data/weber2024/weber2024.csv" 
    # args.model = "weinhardt2025/params/weber2024/spice_weber2024.pkl" 
    # args.additional_columns = None,
    # args.test_sessions = "4,8,12"
    
    # args.epochs = 10
    # args.results = True
    # args.data = "weinhardt2025/data/synthetic/synthetic_256p_0_0.csv"
    # args.model = args.data.replace("data", "params").replace("/synthetic_", "/spice_synthetic_").replace(".csv", "_test.pkl")
    
    example_participant = 2
    plot_coef_dist = False
    
    if args.train_ratio_time and args.test_sessions:
        raise ValueError("kwargs train_ratio_time and test_sessions cannot be assigned at the same time.")
    
    print("\n"+"="*_get_terminal_width())
    print(f"Dataset: {args.data}")
    dataset = csv_to_dataset(
        file=args.data,
        df_participant_id='participant',
        additional_inputs=args.additional_columns.split(',') if args.additional_columns else None,
        timeshift_additional_inputs=args.timeshift_additional_columns,
    )
    
    if args.train_ratio_time:
        args.test_sessions = None
        dataset_train, dataset_test = split_data_along_timedim(dataset, args.train_ratio_time)
    elif args.test_sessions:
        args.test_sessions = [int(item) for item in args.test_sessions.split(',')]
        dataset_train, dataset_test = split_data_along_sessiondim(dataset, args.test_sessions)
    else:
        print("Training/test split: None")
        dataset_train, dataset_test = dataset, dataset
    
    dataset_tuple = dataset_train.xs, dataset_train.ys, dataset_train.xs, dataset_train.ys
    
    n_actions = dataset_train.ys.shape[-1]
    n_participants = len(dataset_train.xs[..., -1].unique())
    n_experiments = len(dataset_train.xs[..., -2].unique())
    n_items = args.n_items if args.n_items else n_actions
    
    if n_items == n_actions:
        # if ((dataset.xs[..., n_actions:n_actions*2].nan_to_num(0) == 1).int() + (dataset.xs[..., n_actions:n_actions*2].nan_to_num(0) == 0).int()).sum() == dataset.xs.shape[0]*dataset.xs.shape[1]*n_actions:
        #     spice_model = workingmemory_rewardbinary
        # else:
        spice_model = workingmemory
    else:
        spice_model = workingmemory_multiitem
    
    # spice_model = choice
    
    class_rnn = spice_model.SpiceModel
    spice_config = spice_model.CONFIG 

    print(f"Dataset size: {n_participants} participants, {n_actions} actions, {n_items} items")
    
    estimator = SpiceEstimator(
        
        # model paramaeters
        rnn_class=class_rnn,
        spice_config=spice_config,
        n_participants=n_participants,
        n_experiments=n_experiments,
        n_actions=n_actions,
        n_items=n_items,
        
        # rnn training parameters
        epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.epochs//4,
        l2_rnn=args.rnn_l2_lambda,
        batch_size=1024,
        bagging=True,
        scheduler=True,
        
        # sindy fitting parameters
        sindy_epochs=args.epochs,
        sindy_weight=args.sindy_weight,
        sindy_l2_lambda=args.sindy_l2_lambda,
        sindy_pruning_threshold=args.sindy_pruning_threshold,
        sindy_pruning_frequency=args.sindy_pruning_freq,
        sindy_pruning_terms=args.sindy_pruning_terms,
        sindy_pruning_patience=args.sindy_pruning_patience,
        sindy_confidence_threshold=args.sindy_confidence,
        sindy_library_polynomial_degree=2,
        sindy_optimizer_reset=None,
        sindy_ensemble_size=1,
        
        # other parameters
        verbose=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_path_spice=args.model,
    )
    
    if args.epochs == 0:
        estimator.load_spice(args.model)

    training_device_str = "CUDA" if estimator.device == torch.device('cuda') else "CPU"
    print("Training device:", training_device_str)
    print("="*_get_terminal_width())
    if estimator.sindy_epochs > 0 or estimator.epochs > 0:
        estimator.fit(*dataset_tuple)
    
    print("\nTraining complete!")
    
    print(f"\nModel saved to: {args.model}")
    
    if args.results:
        agents={
            'rnn': estimator.rnn_agent,
            }
        
        mask_participant = dataset.xs[:, 0, -1] == example_participant
        
        if args.sindy_weight > 0:
            
            agents['spice'] = estimator.spice_agent
            
            # Print example SPICE model for first participant
            print(f"\nExample SPICE model (participant {example_participant}; n_parameters = {int(estimator.spice_agent.count_parameters()[example_participant, 0])}):")
            print("-" * _get_terminal_width())
            estimator.print_spice_model(participant_id=example_participant)
            print("-" * _get_terminal_width())

        if 'synthetic' in args.data:
            rl_parameters = ['beta_reward', 'beta_choice', 'alpha_reward', 'alpha_penalty', 'alpha_choice', 'forget_rate']
            dataset = csv_to_dataset(
                file=args.data,
                additional_inputs=rl_parameters,
            )
            
            agents['groundtruth'] = Agent(QLearning(
                n_actions=2,
                n_participants=n_participants,
                n_experiments=n_experiments,
                beta_reward=dataset.xs[mask_participant][0, 0, n_actions*2+0].item(),
                beta_choice=dataset.xs[mask_participant][0, 0, n_actions*2+1].item(),
                alpha_reward=dataset.xs[mask_participant][0, 0, n_actions*2+2].item(),
                alpha_penalty=dataset.xs[mask_participant][0, 0, n_actions*2+3].item(),
                alpha_choice=dataset.xs[mask_participant][0, 0, n_actions*2+4].item(),
                forget_rate=dataset.xs[mask_participant][0, 0, n_actions*2+5].item(),
            ), use_sindy=True, deterministic=True)

        fig, axs = plot_session(
            agents,
            experiment=dataset.xs[mask_participant][0],
            signals_to_plot=['value_reward', 'value_choice'],
            )

        plt.show()

        # =====================================================================
        # Plotting section: SINDy coefficients across participants
        # =====================================================================
        if args.sindy_weight > 0 and plot_coef_dist:
            print("\nGenerating coefficient variance plots across participants...")

            # Extract coefficients for all participants
            ensemble_idx = 0  # Use first ensemble member
            coeff_data = {}

            # load ground truth model if given in data
            if 'synthetic' in args.data:
                # dataset with ground truth parameters has already been loaded
                # extract parameters for each participant
                rl_parameter_dict = {param: torch.zeros(n_participants, n_experiments) for param in rl_parameters}
                for index_participant in dataset.xs[:, 0, -1].unique().long():
                    mask_participant = dataset.xs[:, 0, -1] == index_participant
                    params_participant = dataset.xs[mask_participant][0, 0, n_actions*2:-3]
                    for index_param, param in enumerate(rl_parameters):
                        rl_parameter_dict[param][index_participant, 0] = params_participant[index_param]
                        
                # initialize the model
                qlearning = QLearning(
                    n_actions=n_actions,
                    n_participants=n_participants,
                    n_experiments=n_experiments,
                    **rl_parameter_dict,
                )
            else:
                qlearning = None
            
            for module_name in estimator.rnn_model.submodules_rnn:
                # Get coefficients: [n_participants, n_ensemble, n_library_terms]
                coeffs = estimator.rnn_model.sindy_coefficients[module_name][:, ensemble_idx, :].detach().cpu().numpy()
                mask = estimator.rnn_model.sindy_coefficients_presence[module_name][:, ensemble_idx, :].cpu().numpy()

                # Apply mask and adjust identity terms
                sparse_coeffs = coeffs * mask

                # Add 1 to the identity term (where term == module_name)
                term_names = estimator.rnn_model.sindy_candidate_terms[module_name]
                for idx, term in enumerate(term_names):
                    if term == module_name:
                        sparse_coeffs[..., idx] += 1

                coeff_data[module_name] = {
                    'coeffs': sparse_coeffs,
                    'terms': term_names
                }

                # Map ground truth coefficients from qlearning to SPICE positions
                if qlearning is not None:
                    gt_coeffs = np.zeros((n_participants, len(term_names)))
                    candidate_terms_qlearning = qlearning.sindy_candidate_terms[module_name]
                    for term in candidate_terms_qlearning:
                        if term not in term_names:
                            raise ValueError(f"Candidate term {term} of the ground truth model was not found among the candidate terms of the fitted model ({term_names}).")
                        idx_spice = term_names.index(term)
                        idx_qlearning = candidate_terms_qlearning.index(term)
                        gt_coeffs[:, idx_spice] = qlearning.sindy_coefficients[module_name][:, 0, 0, idx_qlearning].detach().cpu().numpy()

                    # Add 1 to identity terms (same transformation as SPICE)
                    for idx, term in enumerate(term_names):
                        if term == module_name:
                            gt_coeffs[:, idx] += 1

                    coeff_data[module_name]['gt_coeffs'] = gt_coeffs

            # Collect all coefficients for one figure
            all_terms = []
            all_coeffs_list = []
            all_gt_coeffs_list = []

            for module_name, data in coeff_data.items():
                coeffs = data['coeffs']
                terms = data['terms']
                gt_coeffs = data.get('gt_coeffs', None)

                for idx, term in enumerate(terms):
                    all_terms.append(f"{module_name}: {term}")
                    all_coeffs_list.append(coeffs[..., idx])
                    if gt_coeffs is not None:
                        all_gt_coeffs_list.append(gt_coeffs[..., idx])
                    else:
                        all_gt_coeffs_list.append(None)

            n_total_terms = len(all_terms)

            # Create single figure with all coefficient histograms
            n_cols = min(4, n_total_terms)
            n_rows = (n_total_terms + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            fig.suptitle('Coefficient Distribution Across Participants (Normalized)', fontsize=14, fontweight='bold')

            if n_total_terms == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for idx, (term_name, coeff_values) in enumerate(zip(all_terms, all_coeffs_list)):
                ax = axes[idx]
                gt_coeff_values = all_gt_coeffs_list[idx]

                # Calculate mean and std for normalization (using SPICE values as reference)
                mean_val = coeff_values.mean()
                std_val = coeff_values.std()

                # Normalize SPICE coefficients: (x - mean) / std
                if std_val > 0:
                    normalized_coeffs = (coeff_values - mean_val) / std_val
                else:
                    normalized_coeffs = coeff_values - mean_val

                # Normalize ground truth and plot both with shared bin edges
                if gt_coeff_values is not None:
                    if std_val > 0:
                        normalized_gt = (gt_coeff_values - mean_val) / std_val
                    else:
                        normalized_gt = gt_coeff_values - mean_val
                    # Compute shared bin edges for both distributions
                    all_vals = np.concatenate([normalized_coeffs.flatten(), normalized_gt.flatten()])
                    bin_edges = np.linspace(all_vals.min(), all_vals.max(), 16)
                    ax.hist(normalized_gt, bins=bin_edges, color='tab:blue', alpha=0.7, edgecolor='black', label='Ground Truth')
                    ax.hist(normalized_coeffs, bins=bin_edges, color='tab:orange', alpha=0.7, edgecolor='black', label='SPICE')
                else:
                    # Plot SPICE histogram only (orange)
                    ax.hist(normalized_coeffs, bins=15, color='tab:orange', alpha=0.7, edgecolor='black', label='SPICE')

                # Add vertical line at mean (which should be 0 after normalization)
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean')

                ax.set_title(f'{term_name}', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend(fontsize=8)

            # Hide unused subplots
            for idx in range(n_total_terms, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.show()

            print(f"Coefficient distribution plot generated with {n_total_terms} coefficients from {len(coeff_data)} modules.")
