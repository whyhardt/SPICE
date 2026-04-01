import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path='AAAAsindy_analysis_with_metrics.csv'):
    """
    Load data and organize SINDY modules (same as original script)
    """
    df = pd.read_csv(csv_path)

    # RNN modules
    list_rnn_modules = [
        'x_learning_rate_reward',
        'x_value_reward_not_chosen',
        'x_value_choice_chosen',
        'x_value_choice_not_chosen'
    ]

    all_sindy_cols = [col for col in df.columns if col.startswith('x_')]

    modules = {}
    for module in list_rnn_modules:
        modules[module] = {
            'bias': [],
            'variable': [],
            'control': [],
            'variable_interaction': []
        }

    # Categorize SINDY columns by module and input type
    for col in all_sindy_cols:
        module_found = None
        for module in list_rnn_modules:
            if col.startswith(module):
                module_found = module
                break
        if module_found is None:
            continue

        if col.endswith('_1'):
            modules[module_found]['bias'].append(col)
        elif '_x_' in col and not col.endswith('_1'):
            modules[module_found]['variable'].append(col)
        elif 'c_action' in col or 'c_reward' in col:
            modules[module_found]['control'].append(col)
        elif 'c_value_reward' in col or 'c_value_choice' in col:
            modules[module_found]['variable_interaction'].append(col)
        else:
            if '_x_' in col:
                modules[module_found]['variable'].append(col)

    # Filter out empty modules
    modules = {
        k: v for k, v in modules.items()
        if any(len(input_list) > 0 for input_list in v.values())
    }

    return df, modules


def create_module_correlation_matrices(df, modules, output_dir):
    """
    Create correlation matrices for each SINDY module vs behavioral measures
    in a 2x2 subplot layout
    """
    
    # Define behavioral measures
    behavioral_measures = [
        'stay_after_reward',
        'switch_rate',
        'perseveration',
        'avg_reward',
        'avg_rt'
    ]
    
    # Filter behavioral measures that exist in the dataframe
    available_behaviors = [col for col in behavioral_measures if col in df.columns]
    if not available_behaviors:
        print("No behavioral measures found in the dataframe!")
        return
    
    print(f"Available behavioral measures: {available_behaviors}")
    
    # Get module names and ensure we have exactly 4 modules for 2x2 layout
    module_names = list(modules.keys())
    print(f"Available modules: {module_names}")
    
    if len(module_names) == 0:
        print("No modules found!")
        return
    
    # Create 2x2 subplot figure with larger size and more spacing between subplots
    fig, axes = plt.subplots(2, 2, figsize=(28, 22))
    fig.suptitle('SINDY Module Correlation Matrices with Behavioral Measures', fontsize=24, y=0.98)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Process each module (up to 4 modules for 2x2 layout)
    for idx, module_name in enumerate(module_names[:4]):
        ax = axes_flat[idx]
        
        # Collect all SINDY features for this module in specified order
        module_features = []
        module_feature_names = []
        
        # Define the order: Variable, Variable Interaction, Control, Bias
        input_type_order = ['variable', 'variable_interaction', 'control', 'bias']
        
        # Get clean module name for display
        clean_module_name = module_name.replace('x_', '').replace('_', ' ').title()
        
        for input_type in input_type_order:
            if input_type in modules[module_name] and len(modules[module_name][input_type]) > 0:
                cols = modules[module_name][input_type]
                
                for col in cols:
                    module_features.append(col)
                    
                    # Create better readable names based on input type
                    if input_type == 'bias':
                        module_feature_names.append("Bias")
                    
                    elif input_type == 'variable':
                        # Use module name instead of 'Variable'
                        var_name = col.replace(module_name + '_', '')
                        # Remove trailing numbers and clean up
                        var_name = var_name.split('_')[0] if '_' in var_name else var_name
                        var_name = var_name.replace('x', '').strip('_')
                        if var_name:
                            module_feature_names.append(f"{clean_module_name}: {var_name}")
                        else:
                            module_feature_names.append(clean_module_name)
                    
                    elif input_type == 'variable_interaction':
                        # Extract interaction details - show which OTHER module this one interacts with
                        interaction_part = col.replace(module_name + '_', '')
                        
                        # Debug: print the interaction_part to see what we're working with
                        print(f"DEBUG: Column: {col}")
                        print(f"DEBUG: Module name: {module_name}")
                        print(f"DEBUG: Interaction part: {interaction_part}")
                        
                        # Look for patterns to identify the OTHER module this current module interacts with
                        # Check for the most specific patterns first
                        if 'value_reward_not_chosen' in interaction_part:
                            module_feature_names.append("Interaction with\nValue Reward Not Chosen")
                        elif 'value_choice_not_chosen' in interaction_part:
                            module_feature_names.append("Interaction with\nValue Choice Not Chosen")
                        elif 'value_choice_chosen' in interaction_part:
                            module_feature_names.append("Interaction with\nValue Choice Chosen")
                        elif 'learning_rate_reward' in interaction_part:
                            module_feature_names.append("Interaction with\nLearning Rate Reward")
                        elif 'value_reward' in interaction_part and 'not_chosen' not in interaction_part:
                            module_feature_names.append("Interaction with\nValue Reward")
                        elif 'value_choice' in interaction_part and 'not_chosen' not in interaction_part and 'chosen' not in interaction_part:
                            module_feature_names.append("Interaction with\nValue Choice")
                        elif 'learning_rate' in interaction_part:
                            module_feature_names.append("Interaction with\nLearning Rate")
                        else:
                            # Try to extract the interacting module name from the column
                            # Remove the current module prefix and 'c_' prefix to find the OTHER module
                            remaining = interaction_part.replace('c_', '')
                            if remaining:
                                # Convert to readable format - this is the OTHER module
                                other_module = remaining.replace('_', ' ').title()
                                module_feature_names.append(f"Interaction with\n{other_module}")
                                print(f"DEBUG: Fallback - other_module: {other_module}")
                            else:
                                module_feature_names.append("Interaction")
                    
                    elif input_type == 'control':
                        # Extract control type
                        if 'c_action' in col:
                            module_feature_names.append("Control: action")
                        elif 'c_reward' in col:
                            module_feature_names.append("Control: reward")
                        else:
                            # Generic control
                            control_part = col.replace(module_name + '_', '')
                            control_type = control_part.replace('c_', '').split('_')[0]
                            module_feature_names.append(f"Control: {control_type}")
                    
                    else:
                        # Fallback
                        readable_name = col.replace(module_name + '_', '').replace('_', ' ')
                        module_feature_names.append(f"{input_type}: {readable_name}")
        
        if len(module_features) == 0:
            ax.text(0.5, 0.5, f'No features found\nfor {module_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f"{module_name.replace('x_', '').replace('_', ' ').title()}", fontsize=18)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        print(f"\nProcessing {module_name} with {len(module_features)} features")
        print("Feature names being created:")
        for i, name in enumerate(module_feature_names):
            print(f"  {i}: {name}")
        print("Corresponding columns:")
        for i, col in enumerate(module_features):
            print(f"  {i}: {col}")
        print()
        
        # Create correlation matrix
        correlation_matrix = np.zeros((len(module_features), len(available_behaviors)))
        p_value_matrix = np.zeros((len(module_features), len(available_behaviors)))
        
        # Calculate correlations
        for i, feature in enumerate(module_features):
            for j, behavior in enumerate(available_behaviors):
                # Get valid data pairs
                valid_data = df[[feature, behavior]].dropna()
                
                if len(valid_data) >= 10:  # Minimum sample size
                    r, p = pearsonr(valid_data[feature], valid_data[behavior])
                    correlation_matrix[i, j] = r
                    p_value_matrix[i, j] = p
                else:
                    correlation_matrix[i, j] = np.nan
                    p_value_matrix[i, j] = np.nan
        
        # Create heatmap
        mask = np.isnan(correlation_matrix)
        
        # Use seaborn for better looking heatmap with larger annotations
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            mask=mask,
            ax=ax,
            cbar_kws={'shrink': 0.8},
            square=False,
            linewidths=0.5,
            annot_kws={'fontsize': 14}  # Larger annotation text
        )
        
        # Set labels with larger fonts
        ax.set_xlabel('Behavioral Measures', fontsize=16)
        ax.set_ylabel('SINDY Features', fontsize=16)
        ax.set_title(f"{module_name.replace('x_', '').replace('_', ' ').title()}", 
                    fontsize=18, pad=15)
        
        # Set tick labels - allow for longer names and more rows
        behavior_labels = [behavior.replace('_', ' ').title() for behavior in available_behaviors]
        ax.set_xticklabels(behavior_labels, rotation=45, ha='right', fontsize=14)
        
        # Set y-axis labels with larger font and better spacing
        display_names = []
        for name in module_feature_names:
            # Don't truncate - allow full names to be displayed
            display_names.append(name)
        
        ax.set_yticklabels(display_names, rotation=0, fontsize=14, linespacing=1.5)
        
        # Add significance markers
        for i in range(len(module_features)):
            for j in range(len(available_behaviors)):
                if not np.isnan(p_value_matrix[i, j]):
                    p_val = p_value_matrix[i, j]
                    if p_val < 0.001:
                        marker = '***'
                    elif p_val < 0.01:
                        marker = '**'
                    elif p_val < 0.05:
                        marker = '*'
                    else:
                        marker = ''
                    
                    if marker:
                        ax.text(j + 0.5, i + 0.7, marker, 
                               ha='center', va='center', fontsize=12, 
                               color='white', weight='bold')
    
    # Turn off any unused subplots
    for idx in range(len(module_names), 4):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.12, left=0.12, right=0.96, wspace=0.35, hspace=0.30)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'sindy_module_correlation_matrices.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCorrelation matrix plot saved to: {os.path.join(output_dir, 'sindy_module_correlation_matrices.png')}")


def create_summary_correlation_table(df, modules, output_dir):
    """
    Create a summary table of the strongest correlations for each module
    """
    behavioral_measures = [
        'stay_after_reward',
        'switch_rate', 
        'perseveration',
        'avg_reward',
        'avg_rt'
    ]
    
    available_behaviors = [col for col in behavioral_measures if col in df.columns]
    
    summary_data = []
    
    for module_name, input_types in modules.items():
        # Collect all features for this module
        all_features = []
        for cols in input_types.values():
            all_features.extend(cols)
        
        if len(all_features) == 0:
            continue
            
        # Find strongest correlations for this module
        module_correlations = []
        
        for feature in all_features:
            for behavior in available_behaviors:
                valid_data = df[[feature, behavior]].dropna()
                
                if len(valid_data) >= 10:
                    r, p = pearsonr(valid_data[feature], valid_data[behavior])
                    module_correlations.append({
                        'Module': module_name.replace('x_', '').replace('_', ' ').title(),
                        'Feature': feature,
                        'Behavior': behavior.replace('_', ' ').title(),
                        'Correlation': r,
                        'Abs_Correlation': abs(r),
                        'P_Value': p,
                        'N_Samples': len(valid_data)
                    })
        
        # Get top 3 strongest correlations for this module
        if module_correlations:
            module_df = pd.DataFrame(module_correlations)
            top_correlations = module_df.nlargest(3, 'Abs_Correlation')
            summary_data.extend(top_correlations.to_dict('records'))
    
    # Save summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(['Module', 'Abs_Correlation'], ascending=[True, False])
        summary_df.to_csv(os.path.join(output_dir, 'sindy_module_correlation_summary.csv'), index=False)
        
        print(f"\nTop correlations by module:")
        print("=" * 80)
        for module in summary_df['Module'].unique():
            module_data = summary_df[summary_df['Module'] == module].head(3)
            print(f"\n{module}:")
            for _, row in module_data.iterrows():
                print(f"  {row['Feature']} ↔ {row['Behavior']}: r={row['Correlation']:.3f}, p={row['P_Value']:.3f}")


def main():
    # Set up output directory
    output_dir = '/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots/sindy_correlation_matrices'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load and prepare data
    print("Loading and preparing data...")
    df, modules = load_and_prepare_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of modules found: {len(modules)}")
    
    # Create correlation matrices
    print("\nCreating correlation matrices...")
    create_module_correlation_matrices(df, modules, output_dir)
    
    # Create summary table
    print("\nCreating summary correlation table...")
    create_summary_correlation_table(df, modules, output_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()