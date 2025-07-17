import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
import torch
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import AgentNetwork, AgentSpice
from utils.setup_agents import setup_agent_rnn, setup_agent_spice
from resources.rnn import RLRNN_eckstein2022, RLRNN_dezfouli2019
from resources.sindy_utils import SindyConfig_eckstein2022, SindyConfig_dezfouli2019

# =============================================================================
# CONFIGURATION
# =============================================================================

PARTICIPANT_EMB = 0
SINDY_COEFS = 1

# CHANGE THIS VALUE TO SWITCH FROM PARTICIPANT EMBEDDING TO SINDY COEFS
embedding_type = SINDY_COEFS

# Dataset configuration
path_data = 'data/eckstein2022/eckstein2022_age.csv'
path_rnn = 'params/eckstein2022/rnn_eckstein2022_rldm_l1emb_0_001_l2_0_0005.pkl'
path_spice = 'params/eckstein2022/spice_eckstein2022_rldm_l1emb_0_001_l2_0_0005.pkl'
demo_cols = ['age']
rnn_class = RLRNN_eckstein2022
sindy_config = SindyConfig_eckstein2022

# Alternative dataset (uncomment to use)
# path_data = 'data/dezfouli2019/dezfouli2019.csv'
# path_rnn = 'params/dezfouli2019/rnn_dezfouli2019_rldm_l1emb_0_001_l2_0_0001.pkl'
# path_spice = 'params/dezfouli2019/spice_dezfouli2019_rldm_l1emb_0_001_l2_0_0001.pkl'
# demo_cols = ['diag']
# rnn_class = RLRNN_dezfouli2019
# sindy_config = SindyConfig_dezfouli2019

col_participant_id = 'session'
core_cols = [col_participant_id, 'choice', 'reward']

# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================

def calculate_behavioral_metrics(df):
    """Calculate behavioral metrics for each participant from bandit data"""
    behavioral_metrics = []
    
    for participant_id in df[col_participant_id].unique():
        p_data = df[df[col_participant_id] == participant_id].copy()
        
        # Basic performance metrics
        avg_reward = p_data['reward'].mean()
        total_trials = len(p_data)
        
        # Calculate stay/switch behavior
        p_data['prev_choice'] = p_data['choice'].shift(1)
        p_data['prev_reward'] = p_data['reward'].shift(1)
        p_data['stayed'] = (p_data['choice'] == p_data['prev_choice'])
        
        # Remove first trial of each block (no previous choice)
        valid_trials = p_data.dropna(subset=['prev_choice', 'prev_reward'])
        
        if len(valid_trials) == 0:
            continue
            
        # Stay probabilities
        stay_after_reward = valid_trials[valid_trials['prev_reward'] == 1]['stayed'].mean()
        stay_after_no_reward = valid_trials[valid_trials['prev_reward'] == 0]['stayed'].mean()
        overall_stay_prob = valid_trials['stayed'].mean()
        
        # Win-stay-lose-shift ratios
        win_stay = stay_after_reward if not pd.isna(stay_after_reward) else 0
        lose_shift = 1 - stay_after_no_reward if not pd.isna(stay_after_no_reward) else 0
        
        # Choice entropy (exploration measure)
        choice_probs = p_data['choice'].value_counts(normalize=True)
        if len(choice_probs) == 2:
            choice_entropy = -np.sum(choice_probs * np.log2(choice_probs))
        else:
            choice_entropy = 0  # All choices were the same
        
        # Learning-related metrics
        first_half_reward = p_data.iloc[:len(p_data)//2]['reward'].mean()
        second_half_reward = p_data.iloc[len(p_data)//2:]['reward'].mean()
        learning_improvement = second_half_reward - first_half_reward
        
        # Choice consistency (inverse of switching)
        switch_rate = 1 - overall_stay_prob
        
        behavioral_metrics.append({
            col_participant_id: participant_id,
            'avg_reward': avg_reward,
            'total_trials': total_trials,
            'stay_after_reward': win_stay,
            'stay_after_no_reward': stay_after_no_reward,
            'overall_stay_prob': overall_stay_prob,
            'win_stay_lose_shift': win_stay + lose_shift,
            'choice_entropy': choice_entropy,
            'learning_improvement': learning_improvement,
            'switch_rate': switch_rate
        })
    
    return pd.DataFrame(behavioral_metrics)

def extract_participant_demographics(df, demo_cols):
    """Extract demographic/meta information for each participant"""
    if not demo_cols:
        print("No demographic columns found")
        return pd.DataFrame({col_participant_id: df[col_participant_id].unique()})
    
    # map demographics into linear space
    for demo_col in demo_cols:
        unique_values = df[demo_col].unique()
        if isinstance(unique_values[0], str):
            mapping_values = np.linspace(0, 1, len(unique_values))
            mapping = {val: mapping_values[idx] for idx, val in enumerate(unique_values)}
            df[demo_col] = df[demo_col].map(mapping)
    
    # Get first occurrence of each participant (demographics should be constant)
    demographics = df.groupby(col_participant_id)[demo_cols].first().reset_index()
    
    return demographics

def get_embeddings(agent_rnn: AgentNetwork):
    """Extract participant embeddings from trained RNN"""
    participant_ids = torch.arange(agent_rnn._model.n_participants, dtype=torch.int32)
    embeddings = agent_rnn._model.participant_embedding(participant_ids).detach().numpy()
    return embeddings

def get_coefficients(agent_spice: AgentSpice):
    """Extract SINDy coefficients from trained SPICE model"""
    coefficients = None
    coefficient_names = []
    for pid in range(agent_spice._model.n_participants):
        agent_spice.new_sess(participant_id=pid)
        
        betas = agent_spice.get_betas()
        modules = agent_spice.get_modules()
        
        if coefficients is None:
            n_coefs = 0
            for beta in betas:
                coefficient_names.append('beta ' + beta)
            for module in modules:
                n_coefs += max(modules[module][pid].coefficients().shape)
                coefficient_names += [module + ' ' + feature for feature in modules[module][pid].get_feature_names()]
            coefficients = np.zeros((agent_spice._model.n_participants, n_coefs+len(betas)))
        
        for index_beta, beta in enumerate(betas):
            coefficients[pid, index_beta] = betas[beta]
        
        index_coefs = len(betas)
        for module in modules:
            n_coefs_module = max(modules[module][pid].coefficients().shape)
            coefficients[pid, index_coefs:index_coefs+n_coefs_module] = modules[module][pid].coefficients().reshape(-1)
    
    return coefficients, coefficient_names

# =============================================================================
# QUARTILE ANALYSIS FUNCTIONS
# =============================================================================

def categorize_variables_by_quartiles(behavioral_df, demographic_df, embedding_names=None):
    """
    Categorize variables into quartiles (continuous) or existing categories (categorical)
    
    Returns:
    --------
    quartile_data : dict
        Dictionary with variable names as keys and categorization info as values
    """
    # Merge data
    full_data = behavioral_df.merge(demographic_df, on=col_participant_id, how='inner')
    quartile_data = {}
    
    # Get all variables except participant_id
    variables = [col for col in full_data.columns if col != col_participant_id]
    
    for var in variables:
        values = full_data[var].values
        
        # Skip non-numeric variables
        if not np.issubdtype(values.dtype, np.number):
            print(f"Skipping non-numeric variable: {var}")
            continue
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        valid_participant_ids = full_data[col_participant_id].values[valid_mask]
        
        if len(valid_values) < 4:  # Need at least 4 participants
            print(f"Skipping {var}: insufficient data")
            continue
        
        # Determine if categorical or continuous
        unique_vals = len(np.unique(valid_values))
        
        if unique_vals <= 5:  # Treat as categorical
            categories = {}
            for unique_val in np.unique(valid_values):
                mask = valid_values == unique_val
                participant_subset = valid_participant_ids[mask]
                categories[f"{var}_class_{unique_val}"] = {
                    'participant_ids': participant_subset,
                    'type': 'categorical',
                    'value': unique_val,
                    'n_participants': len(participant_subset)
                }
            quartile_data[var] = categories
            
        else:  # Treat as continuous - use quartiles
            # Calculate quartiles
            q25 = np.percentile(valid_values, 25)
            q75 = np.percentile(valid_values, 75)
            
            # Identify participants in lower and upper quartiles
            lower_quartile_mask = valid_values <= q25
            upper_quartile_mask = valid_values >= q75
            
            quartile_data[var] = {
                f"{var}_lower_quartile": {
                    'participant_ids': valid_participant_ids[lower_quartile_mask],
                    'type': 'continuous_lower',
                    'threshold': q25,
                    'n_participants': np.sum(lower_quartile_mask),
                    'value_range': (np.min(valid_values[lower_quartile_mask]), q25)
                },
                f"{var}_upper_quartile": {
                    'participant_ids': valid_participant_ids[upper_quartile_mask],
                    'type': 'continuous_upper', 
                    'threshold': q75,
                    'n_participants': np.sum(upper_quartile_mask),
                    'value_range': (q75, np.max(valid_values[upper_quartile_mask]))
                }
            }
    
    return quartile_data, full_data

def analyze_quartile_embeddings(quartile_data, embeddings, embedding_names=None, min_participants=5):
    """
    Analyze embedding patterns for each quartile/category
    
    Parameters:
    -----------
    quartile_data : dict
        Output from categorize_variables_by_quartiles
    embeddings : np.array
        Embedding matrix (n_participants x n_dimensions)
    embedding_names : list, optional
        Names of embedding dimensions
    min_participants : int
        Minimum participants required for analysis
    """
    
    results = {}
    
    for variable, categories in quartile_data.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING: {variable}")
        print(f"{'='*60}")
        
        var_results = {}
        
        for category_name, category_info in categories.items():
            if category_info['n_participants'] < min_participants:
                print(f"Skipping {category_name}: only {category_info['n_participants']} participants")
                continue
                
            participant_ids = category_info['participant_ids'].astype(int)
            category_embeddings = embeddings[participant_ids]
            
            # Calculate summary statistics for this category
            mean_embedding = np.mean(category_embeddings, axis=0)
            std_embedding = np.std(category_embeddings, axis=0)
            
            var_results[category_name] = {
                'info': category_info,
                'embeddings': category_embeddings,
                'mean_embedding': mean_embedding,
                'std_embedding': std_embedding,
                'participant_ids': participant_ids
            }
            
            print(f"\n{category_name}:")
            print(f"  Participants: {category_info['n_participants']}")
            if category_info['type'].startswith('continuous'):
                print(f"  Value range: {category_info['value_range']}")
            else:
                print(f"  Category value: {category_info['value']}")
        
        # Compare categories within this variable
        if len(var_results) >= 2:
            var_results['comparisons'] = compare_category_embeddings(
                var_results, variable, embedding_names
            )
        
        results[variable] = var_results
    
    return results

def compare_category_embeddings(var_results, variable_name, embedding_names=None, n_top=5):
    """
    Compare embeddings between categories of the same variable
    """
    print(f"\n--- COMPARISONS FOR {variable_name} ---")
    
    comparisons = {}
    category_names = [name for name in var_results.keys() if name != 'comparisons']
    
    # For continuous variables, compare lower vs upper quartile
    if len(category_names) == 2 and any('lower_quartile' in name for name in category_names):
        lower_name = [name for name in category_names if 'lower_quartile' in name][0]
        upper_name = [name for name in category_names if 'upper_quartile' in name][0]
        
        comparison = compare_two_categories(
            var_results[lower_name], var_results[upper_name], 
            lower_name, upper_name, embedding_names, n_top
        )
        comparisons[f"{lower_name}_vs_{upper_name}"] = comparison
        
    # For categorical variables, compare all pairs
    else:
        for i, cat1 in enumerate(category_names):
            for cat2 in category_names[i+1:]:
                comparison = compare_two_categories(
                    var_results[cat1], var_results[cat2],
                    cat1, cat2, embedding_names, n_top
                )
                comparisons[f"{cat1}_vs_{cat2}"] = comparison
    
    return comparisons

def compare_two_categories(cat1_data, cat2_data, cat1_name, cat2_name, embedding_names=None, n_top=5):
    """
    Statistical comparison between two categories
    """
    emb1 = cat1_data['embeddings']
    emb2 = cat2_data['embeddings']
    mean1 = cat1_data['mean_embedding']
    mean2 = cat2_data['mean_embedding']
    
    n_dims = emb1.shape[1]
    
    # Statistical tests for each dimension
    p_values = []
    effect_sizes = []
    mean_differences = []
    
    for dim in range(n_dims):
        # Mann-Whitney U test (non-parametric)
        statistic, p_val = mannwhitneyu(emb1[:, dim], emb2[:, dim], alternative='two-sided')
        
        # Effect size (Cohen's d approximation)
        pooled_std = np.sqrt(((len(emb1) - 1) * np.var(emb1[:, dim]) + 
                             (len(emb2) - 1) * np.var(emb2[:, dim])) / 
                            (len(emb1) + len(emb2) - 2))
        effect_size = (mean1[dim] - mean2[dim]) / pooled_std if pooled_std > 0 else 0
        
        p_values.append(p_val)
        effect_sizes.append(effect_size)
        mean_differences.append(mean1[dim] - mean2[dim])
    
    # Multiple comparison correction (Bonferroni)
    p_values = np.array(p_values)
    p_values_corrected = p_values * n_dims
    p_values_corrected = np.minimum(p_values_corrected, 1.0)
    
    # Find most significant differences
    significant_dims = np.where(p_values_corrected < 0.05)[0]
    effect_sizes = np.array(effect_sizes)
    
    # Sort by absolute effect size
    sorted_indices = np.argsort(np.abs(effect_sizes))[::-1]
    top_indices = sorted_indices[:n_top]
    
    print(f"\n{cat1_name} (n={len(emb1)}) vs {cat2_name} (n={len(emb2)}):")
    print(f"  Significant dimensions: {len(significant_dims)}/{n_dims}")
    
    print(f"\n  Top {n_top} dimensions by effect size:")
    for i, dim_idx in enumerate(top_indices):
        significance = "***" if p_values_corrected[dim_idx] < 0.001 else \
                     "**" if p_values_corrected[dim_idx] < 0.01 else \
                     "*" if p_values_corrected[dim_idx] < 0.05 else ""
        
        if embedding_names and dim_idx < len(embedding_names):
            dim_name = embedding_names[dim_idx]
        else:
            dim_name = f"Dim_{dim_idx}"
            
        print(f"    {dim_name}: "
              f"Δ={mean_differences[dim_idx]:+.3f}, "
              f"Cohen's d={effect_sizes[dim_idx]:+.3f}, "
              f"p={p_values_corrected[dim_idx]:.3f} {significance}")
    
    return {
        'p_values': p_values,
        'p_values_corrected': p_values_corrected,
        'effect_sizes': effect_sizes,
        'mean_differences': mean_differences,
        'significant_dimensions': significant_dims,
        'top_dimensions': top_indices,
        'n_cat1': len(emb1),
        'n_cat2': len(emb2)
    }

def classify_quartiles_with_embeddings(quartile_data, embeddings, embedding_names=None):
    """
    Build classifiers to predict quartile membership from embeddings
    """
    print(f"\n{'='*60}")
    print("CLASSIFICATION ANALYSIS")
    print(f"{'='*60}")
    
    classification_results = {}
    
    for variable, categories in quartile_data.items():
        category_names = list(categories.keys())
        
        # Skip if too few categories or participants
        if len(category_names) < 2:
            continue
            
        total_participants = sum(cat['n_participants'] for cat in categories.values())
        if total_participants < 20:
            print(f"\nSkipping {variable}: insufficient total participants ({total_participants})")
            continue
        
        print(f"\n--- CLASSIFYING {variable} ---")
        
        # Prepare data for classification
        X_list = []
        y_list = []
        labels = []
        
        for cat_idx, (cat_name, cat_data) in enumerate(categories.items()):
            if cat_data['n_participants'] >= 5:  # Minimum participants per class
                participant_ids = cat_data['participant_ids'].astype(int)
                cat_embeddings = embeddings[participant_ids]
                
                X_list.append(cat_embeddings)
                y_list.extend([cat_idx] * len(cat_embeddings))
                labels.append(cat_name)
        
        if len(X_list) < 2:
            print(f"  Insufficient categories with enough participants")
            continue
            
        # Combine data
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        # Standardize embeddings
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train classifier
        if len(np.unique(y)) == 2:
            # Binary classification - use balanced class weights
            clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        else:
            # Multi-class classification
            clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        
        # Fit and evaluate
        clf.fit(X_scaled, y)
        
        # Predictions and metrics
        y_pred = clf.predict(X_scaled)
        train_accuracy = accuracy_score(y, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
        cv_accuracy = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"  Classes: {labels}")
        print(f"  Sample sizes: {[np.sum(y == i) for i in range(len(labels))]}")
        print(f"  Train accuracy: {train_accuracy:.3f}")
        print(f"  CV accuracy: {cv_accuracy:.3f} ± {cv_std:.3f}")
        
        # Feature importance
        if hasattr(clf, 'coef_'):
            # Linear model coefficients
            if clf.coef_.ndim > 1:
                importances = np.abs(clf.coef_).mean(axis=0)
            else:
                importances = np.abs(clf.coef_)
        elif hasattr(clf, 'feature_importances_'):
            # Tree-based feature importances
            importances = clf.feature_importances_
        else:
            importances = None
        
        if importances is not None:
            top_features = np.argsort(importances)[-5:][::-1]
            print(f"  Top 5 predictive dimensions:")
            for idx in top_features:
                if embedding_names and idx < len(embedding_names):
                    feature_name = embedding_names[idx]
                else:
                    feature_name = f"Dim_{idx}"
                print(f"    {feature_name}: {importances[idx]:.3f}")
        
        classification_results[variable] = {
            'classifier': clf,
            'scaler': scaler,
            'labels': labels,
            'train_accuracy': train_accuracy,
            'cv_accuracy': cv_accuracy,
            'cv_std': cv_std,
            'feature_importances': importances,
            'X': X_scaled,
            'y': y
        }
    
    return classification_results

def plot_quartile_analysis_results(quartile_results, classification_results, embedding_names=None):
    """
    Create comprehensive visualizations of the quartile analysis
    """
    # 1. Classification performance plot
    if classification_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Classification accuracies
        variables = list(classification_results.keys())
        train_accs = [classification_results[var]['train_accuracy'] for var in variables]
        cv_accs = [classification_results[var]['cv_accuracy'] for var in variables]
        cv_stds = [classification_results[var]['cv_std'] for var in variables]
        
        x_pos = np.arange(len(variables))
        axes[0, 0].bar(x_pos - 0.2, train_accs, 0.4, label='Train Accuracy', alpha=0.7)
        axes[0, 0].errorbar(x_pos + 0.2, cv_accs, yerr=cv_stds, fmt='o', 
                           label='CV Accuracy', capsize=5)
        axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
        axes[0, 0].set_xlabel('Variable')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Quartile Classification Performance')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(variables, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Feature importance heatmap (top variables only)
        if len(variables) > 0:
            # Select top 3 variables by CV accuracy
            top_vars = sorted(variables, key=lambda x: classification_results[x]['cv_accuracy'], reverse=True)[:3]
            
            importance_matrix = []
            for var in top_vars:
                importances = classification_results[var]['feature_importances']
                if importances is not None:
                    # Take top 10 features
                    top_features = np.argsort(importances)[-10:][::-1]
                    importance_matrix.append(importances[top_features])
            
            if importance_matrix:
                importance_matrix = np.array(importance_matrix)
                im = axes[0, 1].imshow(importance_matrix, cmap='viridis', aspect='auto')
                axes[0, 1].set_xlabel('Top Embedding Dimensions')
                axes[0, 1].set_ylabel('Variables')
                axes[0, 1].set_title('Feature Importance Heatmap')
                axes[0, 1].set_yticks(range(len(top_vars)))
                axes[0, 1].set_yticklabels(top_vars)
                plt.colorbar(im, ax=axes[0, 1])
        
        # Sample size distribution
        all_sample_sizes = []
        all_variable_names = []
        for var, categories in quartile_results.items():
            for cat_name, cat_data in categories.items():
                if cat_name != 'comparisons':
                    all_sample_sizes.append(cat_data['info']['n_participants'])
                    all_variable_names.append(f"{var}_{cat_data['info']['type']}")
        
        axes[1, 0].bar(range(len(all_sample_sizes)), all_sample_sizes, alpha=0.7)
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Number of Participants')
        axes[1, 0].set_title('Sample Sizes by Category')
        axes[1, 0].set_xticks(range(len(all_variable_names)))
        axes[1, 0].set_xticklabels(all_variable_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Effect size distribution (from comparisons)
        all_effect_sizes = []
        comparison_names = []
        for var, var_data in quartile_results.items():
            if 'comparisons' in var_data:
                for comp_name, comp_data in var_data['comparisons'].items():
                    effect_sizes = comp_data['effect_sizes']
                    # Take absolute values and get max effect size
                    max_effect = np.max(np.abs(effect_sizes))
                    all_effect_sizes.append(max_effect)
                    comparison_names.append(f"{var}")
        
        if all_effect_sizes:
            axes[1, 1].bar(range(len(all_effect_sizes)), all_effect_sizes, alpha=0.7)
            axes[1, 1].set_xlabel('Variable Comparison')
            axes[1, 1].set_ylabel('Max |Effect Size| (Cohen\'s d)')
            axes[1, 1].set_title('Maximum Effect Sizes by Variable')
            axes[1, 1].set_xticks(range(len(comparison_names)))
            axes[1, 1].set_xticklabels(comparison_names, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def create_summary_report(quartile_data, quartile_results, classification_results):
    """
    Create a comprehensive summary report of the analysis
    """
    print("\n" + "="*80)
    print("QUARTILE ANALYSIS SUMMARY REPORT")
    print("="*80)
    
    # Variable overview
    print(f"\n1. VARIABLES ANALYZED:")
    print("-" * 40)
    for var, categories in quartile_data.items():
        n_categories = len(categories)
        total_participants = sum(cat['n_participants'] for cat in categories.values())
        print(f"  {var}: {n_categories} categories/quartiles ({total_participants} total participants)")
    
    # Statistical significance summary
    print(f"\n2. STATISTICAL COMPARISONS:")
    print("-" * 40)
    for var, var_data in quartile_results.items():
        if 'comparisons' in var_data:
            for comp_name, comp_data in var_data['comparisons'].items():
                n_sig = len(comp_data['significant_dimensions'])
                n_total = len(comp_data['effect_sizes'])
                max_effect = np.max(np.abs(comp_data['effect_sizes']))
                print(f"  {comp_name}:")
                print(f"    Significant dimensions: {n_sig}/{n_total}")
                print(f"    Max effect size: {max_effect:.3f}")
    
    # Classification performance summary
    if classification_results:
        print(f"\n3. CLASSIFICATION PERFORMANCE:")
        print("-" * 40)
        for var, results in classification_results.items():
            cv_acc = results['cv_accuracy']
            cv_std = results['cv_std']
            above_chance = cv_acc > 0.6  # Assuming binary classification
            status = "✓" if above_chance else "✗"
            print(f"  {var}: {cv_acc:.3f} ± {cv_std:.3f} {status}")
    
    # Top predictive features
    print(f"\n4. MOST PREDICTIVE EMBEDDING DIMENSIONS:")
    print("-" * 40)
    if classification_results:
        all_importances = {}
        for var, results in classification_results.items():
            if results['feature_importances'] is not None:
                importances = results['feature_importances']
                top_features = np.argsort(importances)[-3:][::-1]
                print(f"  {var}:")
                for idx in top_features:
                    print(f"    Dim {idx}: {importances[idx]:.3f}")

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def main_quartile_analysis(embedding_type: int = PARTICIPANT_EMB):
    """
    Main function to run the quartile-based embedding analysis
    """
    print("="*80)
    print("QUARTILE-BASED EMBEDDING ANALYSIS")
    print("="*80)
    
    # Step 0: Load and prepare data
    print("\nStep 0: Loading and preparing data...")
    df = pd.read_csv(path_data)
    
    # map participant ids onto arange
    unique_values = df[col_participant_id].unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    df[col_participant_id] = df[col_participant_id].map(mapping)
    
    # Calculate behavioral metrics
    print("Calculating behavioral metrics...")
    behavioral_df = calculate_behavioral_metrics(df)
    print(f"Calculated metrics for {len(behavioral_df)} participants")
    
    # Extract demographics
    print("Extracting demographic information...")
    demographic_df = extract_participant_demographics(df, demo_cols=demo_cols)
    print("Demographic variables:", list(demographic_df.columns))
    
    # Get embeddings or coefficients
    if embedding_type == PARTICIPANT_EMB:
        print("Loading participant embeddings...")
        agent_rnn = setup_agent_rnn(
            class_rnn=rnn_class, 
            path_model=path_rnn,
            list_sindy_signals=sindy_config['rnn_modules']+sindy_config['control_parameters'],
        )
        embeddings = get_embeddings(agent_rnn=agent_rnn)
        embedding_names = None
        print(f"Using participant embeddings, shape: {embeddings.shape}")
    else:
        print("Loading SINDy coefficients...")
        agent_spice = setup_agent_spice(
            class_rnn=rnn_class,
            path_rnn=path_rnn,
            path_data=path_data,
            path_spice=path_spice,
            rnn_modules=sindy_config['rnn_modules'],
            control_parameters=sindy_config['control_parameters'],
            sindy_library_setup=sindy_config['library_setup'],
            sindy_filter_setup=sindy_config['filter_setup'],
            sindy_dataprocessing=sindy_config['dataprocessing_setup'],
            sindy_library_polynomial_degree=1,
        )
        embeddings, embedding_names = get_coefficients(agent_spice=agent_spice)
        print(f"Using SINDy coefficients, shape: {embeddings.shape}")
        print(f"Coefficient names: {len(embedding_names)} features")
    
    # Step 1: Categorize variables by quartiles/categories
    print("\nStep 1: Categorizing variables...")
    quartile_data, full_data = categorize_variables_by_quartiles(
        behavioral_df, demographic_df, embedding_names
    )
    
    # Step 2: Analyze embedding patterns for each quartile/category
    print("\nStep 2: Analyzing embedding patterns...")
    quartile_results = analyze_quartile_embeddings(
        quartile_data, embeddings, embedding_names
    )
    
    # Step 3: Build classifiers to predict quartile membership
    print("\nStep 3: Building classifiers...")
    classification_results = classify_quartiles_with_embeddings(
        quartile_data, embeddings, embedding_names
    )
    
    # Step 4: Visualize results
    print("\nStep 4: Creating visualizations...")
    plot_quartile_analysis_results(quartile_results, classification_results, embedding_names)
    
    # Step 5: Create summary report
    create_summary_report(quartile_data, quartile_results, classification_results)
    
    return {
        'data': {
            'behavioral_df': behavioral_df,
            'demographic_df': demographic_df,
            'embeddings': embeddings,
            'embedding_names': embedding_names
        },
        'quartile_data': quartile_data,
        'quartile_results': quartile_results,
        'classification_results': classification_results
    }

# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

def get_participants_in_quartile(quartile_data, variable, quartile_type):
    """
    Get participant IDs for a specific quartile/category
    
    Parameters:
    -----------
    quartile_data : dict
        Output from categorize_variables_by_quartiles
    variable : str
        Variable name (e.g., 'avg_reward')
    quartile_type : str
        'lower', 'upper', or specific category value
    """
    if variable not in quartile_data:
        print(f"Variable {variable} not found")
        return None
    
    categories = quartile_data[variable]
    
    for cat_name, cat_data in categories.items():
        if quartile_type == 'lower' and 'lower_quartile' in cat_name:
            return cat_data['participant_ids']
        elif quartile_type == 'upper' and 'upper_quartile' in cat_name:
            return cat_data['participant_ids']
        elif f"class_{quartile_type}" in cat_name:
            return cat_data['participant_ids']
    
    print(f"Quartile type {quartile_type} not found for variable {variable}")
    return None

def compare_specific_quartiles(quartile_results, var1, quartile1, var2, quartile2, embedding_names=None):
    """
    Compare embeddings between specific quartiles of different variables
    
    Example: Compare high reward earners vs high age participants
    """
    # Get the embedding data for each group
    try:
        if 'comparisons' in quartile_results[var1]:
            # This is a bit complex to extract from the existing structure
            # For now, just print what comparisons are available
            print(f"Available comparisons for {var1}:")
            for comp in quartile_results[var1]['comparisons'].keys():
                print(f"  {comp}")
        
        print("For custom cross-variable comparisons, use get_participants_in_quartile() to get participant IDs")
        print("Then manually extract embeddings and use compare_two_categories()")
        
    except KeyError:
        print(f"Variable {var1} or {var2} not found in results")

def save_results_to_file(results, filename="quartile_analysis_results.pkl"):
    """
    Save analysis results to file
    """
    import pickle
    
    # Remove non-serializable objects (sklearn models)
    results_copy = results.copy()
    if 'classification_results' in results_copy:
        for var in results_copy['classification_results']:
            # Remove the actual model objects but keep metrics
            results_copy['classification_results'][var] = {
                k: v for k, v in results_copy['classification_results'][var].items() 
                if k not in ['classifier', 'scaler']
            }
    
    with open(filename, 'wb') as f:
        pickle.dump(results_copy, f)
    
    print(f"Results saved to {filename}")

# =============================================================================
# EXAMPLE USAGE AND MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the complete quartile analysis
    print("Starting quartile-based embedding analysis...")
    
    # Main analysis
    results = main_quartile_analysis(embedding_type=embedding_type)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\nYou can now use the following:")
    print("1. results['quartile_data'] - Raw quartile categorizations")
    print("2. results['quartile_results'] - Statistical comparisons")
    print("3. results['classification_results'] - Classification performance")
    print("4. results['data'] - Original data and embeddings")
    
    # Example of accessing specific results
    print("\nExample: Getting high vs low reward participants:")
    if 'avg_reward' in results['quartile_data']:
        quartile_data = results['quartile_data']
        high_reward_participants = get_participants_in_quartile(quartile_data, 'avg_reward', 'upper')
        low_reward_participants = get_participants_in_quartile(quartile_data, 'avg_reward', 'lower')
        
        if high_reward_participants is not None and low_reward_participants is not None:
            print(f"  High reward participants (n={len(high_reward_participants)}): {high_reward_participants[:5]}...")
            print(f"  Low reward participants (n={len(low_reward_participants)}): {low_reward_participants[:5]}...")
    
    # Save results
    save_results_to_file(results)
    
    print(f"\nAnalysis complete! Check the plots and printed output above for insights.")