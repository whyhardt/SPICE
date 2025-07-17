import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path='dezfouli_final_df_sindy_analysis_with_metrics.csv'):
    """Load data and extract SINDY coefficients + beta_reward & beta_choice."""
    df = pd.read_csv(csv_path)
    sindy_cols = [c for c in df.columns if c.startswith('x_')]
    for extra in ('beta_reward', 'beta_choice'):
        if extra in df.columns:
            sindy_cols.append(extra)
    return df, sindy_cols


def clean_coefficient_name(col):
    """Clean coefficient names for display."""
    return (col.replace('x_', '').replace('_', ' ').title())[:50]


def get_significance(p):
    """Convert p-value to significance stars."""
    if pd.isna(p):
        return 'na'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def run_pairwise_logistic(presence, is_healthy):
    """Run logistic regression for binary comparison."""
    try:
        scaler = StandardScaler()
        X = scaler.fit_transform(is_healthy.reshape(-1, 1)).flatten()
        
        model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=0)
        model.fit(X.reshape(-1, 1), presence)
        
        # Odds ratio
        or_val = np.exp(model.coef_[0][0])
        
        # Likelihood ratio test
        p_hat = model.predict_proba(X.reshape(-1, 1))[:, 1]
        eps = 1e-15
        ll = np.sum(presence * np.log(np.clip(p_hat, eps, 1 - eps)) + 
                   (1 - presence) * np.log(np.clip(1 - p_hat, eps, 1 - eps)))
        p0 = presence.mean()
        ll0 = np.sum(presence * np.log(p0) + (1 - presence) * np.log(1 - p0))
        lr = -2 * (ll0 - ll)
        p_val = 1 - stats.chi2.cdf(max(0, lr), df=1)
        
        return or_val, p_val
    except:
        return np.nan, np.nan


def perform_pairwise_diagnosis_analysis(df, cols, output_dir):
    """Perform Healthy vs Bipolar and Healthy vs Depression comparisons."""
    
    # Filter and check data
    df_clean = df[df['Diagnosis'].notna()].copy()
    unique_diagnoses = df_clean['Diagnosis'].unique()
    print(f"Using {len(df_clean)} participants; Diagnoses: {unique_diagnoses}")
    
    if 'Healthy' not in unique_diagnoses:
        raise ValueError("'Healthy' group not found in data")
    
    diagnosis_groups = [d for d in unique_diagnoses if d != 'Healthy']
    print(f"Comparing Healthy vs: {diagnosis_groups}")
    
    # Sample sizes
    for diag in unique_diagnoses:
        n = sum(df_clean['Diagnosis'] == diag)
        print(f"  {diag}: n={n}")
    
    results = []
    skipped = []
    
    for col in cols:
        vals = df_clean[col].values
        mask = ~np.isnan(vals)
        
        if mask.sum() < 10:
            skipped.append((col, f"<10 obs"))
            continue
            
        vals_clean = vals[mask]
        diagnosis_clean = df_clean['Diagnosis'].values[mask]
        presence = (vals_clean != 0).astype(int)
        presence_rate = presence.mean()
        
        if presence_rate == 0:
            skipped.append((col, "all zero"))
            continue
        elif presence_rate == 1.0:
            skipped.append((col, "always present"))
            continue
        
        # Initialize result
        result = {
            'coefficient': col,
            'coefficient_clean': clean_coefficient_name(col),
            'n_total': int(len(presence)),
            'presence_rate': presence_rate
        }
        
        # Run pairwise comparisons
        for diag_group in diagnosis_groups:
            # Create binary mask for this comparison
            pairwise_mask = (diagnosis_clean == 'Healthy') | (diagnosis_clean == diag_group)
            
            if pairwise_mask.sum() < 10:
                result[f'healthy_vs_{diag_group.lower()}_OR'] = np.nan
                result[f'healthy_vs_{diag_group.lower()}_p'] = np.nan
                result[f'healthy_vs_{diag_group.lower()}_sig'] = 'insufficient_data'
                continue
            
            # Get data for this comparison
            pair_diagnosis = diagnosis_clean[pairwise_mask]
            pair_presence = presence[pairwise_mask]
            is_healthy = (pair_diagnosis == 'Healthy').astype(int)
            
            # Sample sizes
            n_healthy = sum(is_healthy)
            n_diag = sum(1 - is_healthy)
            result[f'n_healthy_{diag_group.lower()}'] = n_healthy
            result[f'n_{diag_group.lower()}'] = n_diag
            
            # Check variation
            if pair_presence.std() == 0:
                result[f'healthy_vs_{diag_group.lower()}_OR'] = np.nan
                result[f'healthy_vs_{diag_group.lower()}_p'] = np.nan
                result[f'healthy_vs_{diag_group.lower()}_sig'] = 'no_variation'
                continue
            
            # Run logistic regression
            or_val, p_val = run_pairwise_logistic(pair_presence, is_healthy)
            
            result[f'healthy_vs_{diag_group.lower()}_OR'] = or_val
            result[f'healthy_vs_{diag_group.lower()}_p'] = p_val
            result[f'healthy_vs_{diag_group.lower()}_sig'] = get_significance(p_val)
        
        results.append(result)
    
    # Create and save results
    df_results = pd.DataFrame(results)
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'pairwise_analysis_results.csv')
    df_results.to_csv(csv_path, index=False)
    
    # Print summary
    print(f"\nSkipped {len(skipped)} coefficients")
    print(f"Analyzed {len(results)} coefficients")
    
    for diag_group in diagnosis_groups:
        sig_col = f'healthy_vs_{diag_group.lower()}_sig'
        significant = df_results[df_results[sig_col].isin(['*', '**', '***'])]
        print(f"\nHealthy vs {diag_group}: {len(significant)} significant")
        
        for _, row in significant.head(5).iterrows():
            or_val = row[f'healthy_vs_{diag_group.lower()}_OR']
            p_val = row[f'healthy_vs_{diag_group.lower()}_p']
            sig = row[sig_col]
            direction = "↑ healthy" if or_val > 1 else "↓ healthy"
            print(f"  {row['coefficient_clean']}: OR={or_val:.2f}, p={p_val:.4f} {sig} ({direction})")
    
    # Create plots
    create_pairwise_plots(df_results, df_clean, output_dir, diagnosis_groups)
    
    return df_results


def create_pairwise_plots(df_results, df_clean, output_dir, diagnosis_groups):
    """Create odds ratio and presence rate plots."""
    
    color_map = {'***': '#FF0000', '**': '#FFA500', '*': '#FFD700', 'ns': '#999999'}
    
    # 1. Odds ratio plots
    n_plots = len(diagnosis_groups)
    if n_plots == 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 8))
        if n_plots == 1:
            axes = [axes]
    
    for i, diag_group in enumerate(diagnosis_groups):
        or_col = f'healthy_vs_{diag_group.lower()}_OR'
        sig_col = f'healthy_vs_{diag_group.lower()}_sig'
        
        # Get valid data
        valid_data = df_results.dropna(subset=[or_col]).copy()
        valid_data = valid_data[valid_data[sig_col].isin(['***', '**', '*', 'ns'])]
        
        if valid_data.empty:
            axes[i].text(0.5, 0.5, f'No valid data', ha='center', va='center', 
                        transform=axes[i].transAxes)
            continue
        
        # Sort and plot
        valid_data = valid_data.sort_values(or_col)
        colors = valid_data[sig_col].map(color_map)
        
        y_pos = np.arange(len(valid_data))
        axes[i].barh(y_pos, valid_data[or_col], color=colors, edgecolor='black')
        axes[i].axvline(x=1, color='black', linestyle='--', alpha=0.7)
        
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(valid_data['coefficient_clean'], fontsize=8)
        axes[i].set_xlabel('Odds Ratio (OR > 1: more in healthy)')
        axes[i].set_title(f'Healthy vs {diag_group}')
        axes[i].set_xscale('log')
        
        # Add OR values
        for j, (_, row) in enumerate(valid_data.iterrows()):
            axes[i].text(row[or_col], j, f' {row[or_col]:.2f}', va='center', fontsize=7)
    
    # Legend
    if len(diagnosis_groups) > 0:
        handles = [plt.Rectangle((0,0),1,1, facecolor=color_map[sig], edgecolor='black') 
                   for sig in ['***', '**', '*', 'ns']]
        axes[-1].legend(handles, ['p<0.001', 'p<0.01', 'p<0.05', 'ns'], 
                       title='Significance', loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'odds_ratios.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Presence rates for significant coefficients
    sig_coeffs = []
    for _, row in df_results.iterrows():
        for diag_group in diagnosis_groups:
            sig_col = f'healthy_vs_{diag_group.lower()}_sig'
            if row.get(sig_col) in ['*', '**', '***']:
                sig_coeffs.append(row)
                break
    
    if not sig_coeffs:
        sig_coeffs = df_results.nlargest(6, 'presence_rate').to_dict('records')
    else:
        sig_coeffs = sig_coeffs[:6]
    
    if sig_coeffs:
        presence_data = []
        all_groups = ['Healthy'] + diagnosis_groups
        
        for coef in sig_coeffs:
            col = coef['coefficient']
            for diag in all_groups:
                data = df_clean[df_clean['Diagnosis'] == diag][col].dropna()
                if len(data) > 0:
                    presence_data.append({
                        'Coefficient': coef['coefficient_clean'],
                        'Diagnosis': diag,
                        'Presence_Rate': (data != 0).mean(),
                        'N': len(data)
                    })
        
        if presence_data:
            presence_df = pd.DataFrame(presence_data)
            pivot_df = presence_df.pivot(index='Coefficient', columns='Diagnosis', values='Presence_Rate')
            
            # Reorder: Healthy first
            col_order = ['Healthy'] + [d for d in pivot_df.columns if d != 'Healthy']
            pivot_df = pivot_df[col_order]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_df.plot(kind='bar', ax=ax, width=0.8)
            ax.set_ylabel('Presence Rate')
            ax.set_title('Presence Rates for Significant Coefficients')
            ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')
            ax.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'presence_rates.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Plots saved to {output_dir}")


def main(
    csv_path='dezfouli_final_df_sindy_analysis_with_metrics.csv',
    output_dir='/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_dezfouli/plots/pairwise_analysis'
):
    """Run pairwise diagnosis analysis."""
    df, cols = load_and_prepare_data(csv_path)
    results = perform_pairwise_diagnosis_analysis(df, cols, output_dir)
    return results


if __name__ == '__main__':
    main()