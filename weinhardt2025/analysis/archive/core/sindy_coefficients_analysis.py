import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kruskal, norm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import math

DIAGNOSIS_GROUPS = ['Depression', 'Bipolar', 'Healthy']
DIAGNOSIS_LABELS = ['Depression', 'Bipolar', 'Healthy']
EXTRA_COEFFS = ['beta_reward', 'beta_choice']          


def jonckheere_terpstra(groups):
    """Two-sided Jonckheere-Terpstra trend test (returns z, p)."""
    k       = len(groups)
    n_pairs = sum(len(a)*len(b) for i,a in enumerate(groups) for b in groups[i+1:])
    if k < 3 or n_pairs == 0:
        return np.nan, np.nan
    U = sum(sum(y > x for y in b for x in a)
            for i,a in enumerate(groups) for b in groups[i+1:])
    z = (U - n_pairs/2) / np.sqrt(n_pairs/12)
    return z, 2*(1 - norm.cdf(abs(z)))


def describe_groups(values, diagnoses):
    """Return per-group summary list and raw arrays for tests."""
    summaries, raw = [], []
    for idx, label in enumerate(DIAGNOSIS_LABELS):
        grp = values[diagnoses == idx]
        summaries.append(dict(diagnosis_group=idx, label=label, count=len(grp),
                              mean=grp.mean()    if grp.size else np.nan,
                              median=grp.median() if grp.size else np.nan,
                              std=grp.std()      if grp.size else np.nan))
        raw.append(grp.values)
    return summaries, raw


def analyze(df, diagnosis_col='diagnosis_group', out_dir='sindy_magnitude_analysis'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    coeff_cols = [c for c in df.columns if c.startswith('x_')] + \
                 [c for c in EXTRA_COEFFS if c in df.columns]

    keep = [c for c in coeff_cols
            if (nz := df.loc[df[c] != 0, c]).size > 10 and nz.std() > 1e-10]
    if not keep:
        raise ValueError('No variable coefficients found.')

    print('='*80, '\nSINDY / BETA COEFFICIENT MAGNITUDE ANALYSIS')
    print(f'Total participants: {len(df)}')
    print(f'Total candidate columns: {len(coeff_cols)}')
    print(f'Columns with sufficient variation: {len(keep)}\n')

    results = []
    for c in keep:
        nz_mask      = df[c] != 0
        vals, diagnoses = df.loc[nz_mask, c], df.loc[nz_mask, diagnosis_col]
        grp_stats, g = describe_groups(vals, diagnoses)

        rho, p_s     = spearmanr(diagnoses, vals) if vals.size > 10 else (np.nan, np.nan)
        valid_groups = [x for x in g if x.size and x.std() > 1e-10]
        kw,  p_kw    = kruskal(*valid_groups) if len(valid_groups) >= 3 else (np.nan, np.nan)
        z,   p_jt    = jonckheere_terpstra(g)

        trend = ('Increasing' if rho > 0 else 'Decreasing') + ' across diagnosis groups' \
                if not np.isnan(rho) else 'Undetermined'

        results.append(dict(coefficient=c, n_nonzero=vals.size,
                            mean=vals.mean(), std=vals.std(),
                            spearman_rho=rho, spearman_p=p_s,
                            kruskal_stat=kw, kruskal_p=p_kw,
                            jt_z=z, jt_p=p_jt, trend=trend,
                            group_stats=grp_stats))

        print(f'{c:48s}  ρ={rho:5.3f}  p_Spearman={p_s:6.4f} '
              f'p_JT={p_jt:6.4g}  {trend}')

    res = pd.DataFrame(results).sort_values('jt_p')

    for col in ['spearman_p', 'kruskal_p', 'jt_p']:
        mask = res[col].notna()
        res.loc[mask, f'{col}_fdr'] = multipletests(res.loc[mask, col],
                                                    method='fdr_bh')[1]

    _save_csv(res, out/'sindy_magnitude_analysis_results.csv')
    _make_plots(res, df, diagnosis_col, out)
    _print_summary(res, out)
    return res


def _save_csv(res_df, path):
    flat = []
    for _, row in res_df.iterrows():
        rec = {k: row[k] for k in res_df.columns if k != 'group_stats'}
        for g in row.group_stats:
            d = g['diagnosis_group']
            for k in ('count', 'mean', 'median', 'std'):
                rec[f'group_{d}_{k}'] = g[k]
        flat.append(rec)
    pd.DataFrame(flat).to_csv(path, index=False)


def _make_plots(res_df, df, diagnosis_col, out):
    diagnosis_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red for Depression, Teal for Bipolar, Blue for Healthy
    
    # Plot ALL coefficients instead of just top 6
    all_coeffs = res_df.copy()  # Use all coefficients
    n_coeffs = len(all_coeffs)
    
    # Calculate grid dimensions to accommodate all coefficients
    n_cols = 3  # Keep 3 columns for readability
    n_rows = math.ceil(n_coeffs / n_cols)
    
    print(f"Creating plots for {n_coeffs} coefficients in {n_rows}x{n_cols} grid")

    # VIOLIN PLOTS - ALL COEFFICIENTS
    fig_v, axs_v = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    # Handle case where we have only one row
    if n_rows == 1:
        axs_v = axs_v.reshape(1, -1)
    
    for idx, (_, row) in enumerate(all_coeffs.iterrows()):
        ax_row, ax_col = divmod(idx, n_cols)
        ax = axs_v[ax_row, ax_col]
        
        mask = df[row.coefficient] != 0
        vals, diagnoses = df.loc[mask, row.coefficient], df.loc[mask, diagnosis_col]
        data = [vals[diagnoses == g] for g in range(len(DIAGNOSIS_GROUPS))]

        v = ax.violinplot(data, positions=range(len(DIAGNOSIS_GROUPS)),
                          showmeans=False, showmedians=True)

        for i, body in enumerate(v['bodies']):
            body.set_facecolor(diagnosis_colors[i]); body.set_alpha(0.6)
            body.set_edgecolor('black'); body.set_linewidth(1)
        v['cmedians'].set_color('black'); v['cmedians'].set_linewidth(2.1)

        for i, grp in enumerate(data):
            ax.scatter(np.random.normal(i, .04, size=len(grp)),
                       grp, alpha=.2, s=14, color='black', zorder=3)

        # Truncate long coefficient names for better display
        coeff_name = row.coefficient[:30] + '...' if len(row.coefficient) > 30 else row.coefficient
        ax.set_title(f'{coeff_name}\nJT p={row.jt_p:.3g}', fontsize=9)
        ax.set_xticks(range(len(DIAGNOSIS_GROUPS))); ax.set_xticklabels(DIAGNOSIS_LABELS, rotation=45)
        ax.set_xlabel('Diagnosis Group'); ax.set_ylabel('Coefficient Magnitude')
        ax.grid(True, alpha=.8)
    
    # Turn off extra subplots if we have them
    for idx in range(n_coeffs, n_rows * n_cols):
        ax_row, ax_col = divmod(idx, n_cols)
        axs_v[ax_row, ax_col].axis('off')
    
    fig_v.tight_layout()
    fig_v.savefig(out/'sindy_coefficient_diagnosis_trends_all.png', dpi=300, bbox_inches='tight')
    plt.close(fig_v)

    # BOX PLOTS - ALL COEFFICIENTS
    fig_b, axs_b = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    # Handle case where we have only one row
    if n_rows == 1:
        axs_b = axs_b.reshape(1, -1)
    
    for idx, (_, row) in enumerate(all_coeffs.iterrows()):
        ax_row, ax_col = divmod(idx, n_cols)
        ax = axs_b[ax_row, ax_col]
        
        mask = df[row.coefficient] != 0
        vals, diagnoses = df.loc[mask, row.coefficient], df.loc[mask, diagnosis_col]
        data = [vals[diagnoses == g] for g in range(len(DIAGNOSIS_GROUPS))]

        bp = ax.boxplot(data, positions=range(len(DIAGNOSIS_GROUPS)),
                        showmeans=False, patch_artist=True)

        for i, box in enumerate(bp['boxes']):               # colour boxes
            box.set_facecolor(diagnosis_colors[i]); box.set_alpha(0.6)
            box.set_edgecolor('black'); box.set_linewidth(1)

        for med in bp['medians']:                           # bold medians
            med.set_color('black'); med.set_linewidth(2.1)

        for part in ('whiskers', 'caps'):                   # darker whisk/caps
            for line in bp[part]:
                line.set_color('black'); line.set_linewidth(1)

        for i, grp in enumerate(data):                      # jittered points
            ax.scatter(np.random.normal(i, .04, size=len(grp)),
                       grp, alpha=.2, s=14, color='black', zorder=3)

        # Truncate long coefficient names for better display
        coeff_name = row.coefficient[:30] + '...' if len(row.coefficient) > 30 else row.coefficient
        ax.set_title(f'{coeff_name}\nJT p={row.jt_p:.3g}', fontsize=9)
        ax.set_xticks(range(len(DIAGNOSIS_GROUPS))); ax.set_xticklabels(DIAGNOSIS_LABELS, rotation=45)
        ax.set_xlabel('Diagnosis Group'); ax.set_ylabel('Coefficient Magnitude')
        ax.grid(True, alpha=.8)
    
    # Turn off extra subplots if we have them
    for idx in range(n_coeffs, n_rows * n_cols):
        ax_row, ax_col = divmod(idx, n_cols)
        axs_b[ax_row, ax_col].axis('off')
    
    fig_b.tight_layout()
    fig_b.savefig(out/'sindy_coefficient_diagnosis_trends_box_all.png', dpi=300, bbox_inches='tight')
    plt.close(fig_b)

    # HEATMAP - Keep top 15 for readability, or use all if ≤15
    top_for_heatmap = res_df.head(min(15, len(res_df)))
    heat = pd.DataFrame({row.coefficient:
                          [g['mean'] for g in row.group_stats]
                          for _, row in top_for_heatmap.iterrows()}).T
    heat.columns = DIAGNOSIS_LABELS

    plt.figure(figsize=(12, max(8, len(top_for_heatmap) * 0.5)))
    sns.heatmap(heat, annot=True, cmap='RdBu_r', center=0,
                fmt='.3f', cbar_kws={'label': 'Mean Coefficient'})
    heatmap_title = f'SINDy / beta means by diagnosis group (top-{len(top_for_heatmap)} JT)'
    plt.title(heatmap_title)
    plt.tight_layout()
    plt.savefig(out/'sindy_coefficient_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated plots for all {n_coeffs} coefficients")
    print(f"Violin plot saved as: sindy_coefficient_diagnosis_trends_all.png")
    print(f"Box plot saved as: sindy_coefficient_diagnosis_trends_box_all.png")
    print(f"Heatmap saved as: sindy_coefficient_heatmap.png")


def _print_summary(res, out):
    print(f'Total coefficients analysed:            {len(res)}')
    print(f'Significant Spearman (p<0.05):          {(res.spearman_p < 0.05).sum()}')
    print(f'Significant Kruskal-Wallis (p<0.05):    {(res.kruskal_p < 0.05).sum()}')
    print(f'Significant Jonckheere-Terpstra:        {(res.jt_p < 0.05).sum()}')
    print('\nTOP 10 by JT p-value')
    print(res[['coefficient', 'spearman_rho', 'spearman_p',
               'jt_p', 'trend']].head(10).to_string(index=False))


if __name__ == '__main__':
    DATA = '/Users/martynaplomecka/closedloop_rl/dezfouli_final_df_sindy_analysis_with_metrics.csv'
    df   = pd.read_csv(DATA)
    
    diagnosis_mapping = {diag: idx for idx, diag in enumerate(DIAGNOSIS_GROUPS)}
    df['diagnosis_group'] = df['Diagnosis'].map(diagnosis_mapping)
    
    df = df.dropna(subset=['diagnosis_group'])
    
    analyze(df,
            out_dir='/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_dezfouli/plots/sindy_coefficients_analysis')