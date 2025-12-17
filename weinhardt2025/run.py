"""
Test script for end-to-end differentiable SPICE training.
This runs the full pipeline with SINDy regularization during RNN training.
"""
import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd

from spice import SpiceEstimator, convert_dataset, split_data_along_sessiondim, split_data_along_timedim, plot_session
from spice.precoded import choice
from spice.resources.bandits import AgentQ
from spice.precoded import workingmemory_multiitem, workingmemory, choice, rescorlawagner, forgetting


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Trains a SPICE-RNN with end-to-end differentiable SINDy on behavioral data.')

    # necessary parameters
    parser.add_argument('--model', type=str, default=None, help='Model name to load from and/or save to parameters of RNN')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset')
    
    # RNN training parameters
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--l2_rnn', type=float, default=0., help='L2 Reg of the RNN parameters')
    
    # SINDy training parameters
    parser.add_argument('--sindy_weight', type=float, default=0.1, help='Weight for SINDy regularization during RNN training')
    parser.add_argument('--sindy_alpha', type=float, default=0.001, help='L2 Reg of the SINDy coefficients')
    parser.add_argument('--sindy_threshold', type=float, default=0.05, help='Threshold value for cutting off sindy terms')
    parser.add_argument('--sindy_cutoff', type=int, default=1, help='Number of thresholded terms')
    parser.add_argument('--sindy_cutoff_freq', type=int, default=1, help='Number of epochs after which to cutoff')
    parser.add_argument('--sindy_cutoff_patience', type=int, default=100, help='Number of epochs after which to cutoff')
    
    # Data setup parameters
    parser.add_argument('--train_ratio_time', type=float, default=None, help='Ratio of data used for training. Split along time dimension. Not combinable with test_sessions')
    parser.add_argument('--test_sessions', type=str, default=None, help='Comma-separated list of integeres which indicate test sessions. Not combinable with train_ratio_time')
    parser.add_argument('--n_items', type=int, default=None, help='Number of items in dataset; Default None: As many items as actions (automatically detected from dataset);')
    parser.add_argument('--additional_columns', type=str, default=None, help='Comma-separated list of columns which are added to the dataset.')
    parser.add_argument('--timeshift_additional_columns', action='store_true', help='Shifts additional columns (defined by the kwarg "additional_columns") [t]->[t-1]; Necessary for e.g. predictor stimuli which are usually listed in the trial of which SPICE has to predict the action.')
    
    args = parser.parse_args()

    # args.model = "weinhardt2025/params/eckstein2022/spice_eckstein2022.pkl"
    # args.data = "weinhardt2025/data/eckstein2022/eckstein2022.csv"
    # args.train_ratio_time = 0.8
    
    # args.model = "weinhardt2025/params/eckstein2024/spice_eckstein2024.pkl"
    # args.data = "weinhardt2025/data/eckstein2024/eckstein2024.csv"
    # args.test_sessions = "1,3"
    
    # args.model = "weinhardt2025/params/dezfouli2019/spice_dezfouli2019.pkl"
    # args.data = "weinhardt2025/data/dezfouli2019/dezfouli2019.csv"
    # args.test_sessions = "3,6,9"
    
    # args.data="weinhardt2025/data/sugawara2021/sugawara2021.csv" 
    # args.model="weinhardt2025/params/sugawara2021/spice_sugawara2021.pkl" 
    # args.additional_columns="shown_at_0,shown_at_1,shown_at_0_next,shown_at_1_next"
    # args.n_items=8
    # args.test_sessions="1"
    
    args.data = "weinhardt2025/data/synthetic/synthetic_2_256p_0.csv"
    args.model = args.data.replace("data", "params").replace("/synthetic_", "/spice_synthetic_test").replace(".csv", ".pkl")
    
    args.epochs = 100
    args.lr = 0.01
    args.sindy_weight = 0.001
    args.sindy_cutoff_freq = 1
    args.sindy_cutoff = 1
    args.sindy_cutoff_patience = 100
    args.sindy_threshold = 0.05
    args.sindy_alpha = 0.001
    sindy_epochs = 1000
    warmup_steps = 10
    
    example_participant = 1
    plot_coef_dist = True
    
    if args.train_ratio_time and args.test_sessions:
        raise ValueError("kwargs train_ratio_time and test_sessions cannot be assigned at the same time.")
    
    print(f"Loading dataset from {args.data}...")
    dataset = convert_dataset(
        file=args.data,
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
        print("No split into training and test data.")
        dataset_train, dataset_test = dataset, dataset

    n_actions = dataset_train.ys.shape[-1]
    n_participants = len(dataset_train.xs[..., -1].unique())
    n_experiments = len(dataset_train.xs[..., -2].unique())
    n_items = args.n_items if args.n_items else n_actions
    
    if n_items == n_actions:
        spice_model = workingmemory
    else:
        spice_model = workingmemory_multiitem

    spice_model = choice
    
    class_rnn = spice_model.SpiceModel
    spice_config = spice_model.CONFIG

    print(f"Dataset: {n_participants} participants, {n_actions} actions, {n_items} items")
    print(f"Test data: {1-args.train_ratio_time if args.train_ratio_time else args.test_sessions}")

    print(f"\nInitializing SpiceEstimator with end-to-end SINDy training (weight={args.sindy_weight})...")
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
        warmup_steps=warmup_steps,
        l2_rnn=args.l2_rnn,
        learning_rate=args.lr,
        
        # sindy fitting parameters
        sindy_weight=args.sindy_weight,
        sindy_threshold=args.sindy_threshold,
        sindy_threshold_frequency=args.sindy_cutoff_freq,
        sindy_threshold_terms=args.sindy_cutoff,
        sindy_cutoff_patience=args.sindy_cutoff_patience,
        sindy_epochs=sindy_epochs,
        sindy_alpha=args.sindy_alpha,
        sindy_library_polynomial_degree=2,
        sindy_ensemble_size=1,
        
        # additional generalization parameters
        bagging=True,
        # scheduler=True,
        
        # other parameters
        verbose=True,
        # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_path_spice=args.model,
    )
    
    if args.epochs == 0:
        estimator.load_spice(args.model)
    
    print(f"\nStarting training on {estimator.device}...")
    print("=" * 80)
    if estimator.sindy_epochs > 0 or estimator.epochs > 0:
        estimator.fit(dataset_train.xs, dataset_train.ys, data_test=dataset_test.xs, target_test=dataset_test.ys)
    print("=" * 80)
    print("\nTraining complete!")
    
    print(f"\nModel saved to: {args.model}")
    
    agents={
        'rnn': estimator.rnn_agent,
        }
    
    if args.sindy_weight > 0:
        
        agents['spice'] = estimator.spice_agent
        
        # Print example SPICE model for first participant
        print(f"\nExample SPICE model (participant {example_participant}; n_parameters = {estimator.spice_agent.count_parameters()[example_participant]}):")
        print("-" * 80)
        estimator.print_spice_model(participant_id=example_participant)
        print("-" * 80)

    if 'synthetic' in args.data:
        dataset_df = pd.read_csv(args.data)

        # get the parameters for the selected participant and set up the ground truth model
        n_trials = 200
        dataset_df = dataset_df[dataset_df['session'] == example_participant]

        agents['groundtruth'] = AgentQ(
            n_actions=2,
            beta_reward=dataset_df['beta_reward'].values[0],
            alpha_reward=dataset_df['alpha_reward'].values[0],
            alpha_penalty=dataset_df['alpha_penalty'].values[0],
            forget_rate=dataset_df['forget_rate'].values[0],
            beta_choice=dataset_df['beta_choice'].values[0],
            alpha_choice=dataset_df['alpha_choice'].values[0],
        )

    fig, axs = plot_session(
        agents,
        experiment=dataset.xs[example_participant]
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

        # Collect all coefficients for one figure
        all_terms = []
        all_coeffs_list = []

        for module_name, data in coeff_data.items():
            coeffs = data['coeffs']
            terms = data['terms']

            for idx, term in enumerate(terms):
                all_terms.append(f"{module_name}: {term}")
                all_coeffs_list.append(coeffs[..., idx])

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

            # Calculate mean and std for normalization
            mean_val = coeff_values.mean()
            std_val = coeff_values.std()

            # Normalize coefficients: (x - mean) / std
            if std_val > 0:
                normalized_coeffs = (coeff_values - mean_val) / std_val
            else:
                normalized_coeffs = coeff_values - mean_val

            # Create histogram
            ax.hist(normalized_coeffs, bins=15, color='steelblue', alpha=0.7, edgecolor='black')

            # Add vertical line at mean (which should be 0 after normalization)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean')

            # ax.set_xlabel('Normalized Coefficient Value')
            # ax.set_ylabel('Frequency')
            ax.set_title(f'{term_name}', fontsize=9)  # \nμ={mean_val:.3f}, σ={std_val:.3f}
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(n_total_terms, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Coefficient distribution plot generated with {n_total_terms} coefficients from {len(coeff_data)} modules.")
