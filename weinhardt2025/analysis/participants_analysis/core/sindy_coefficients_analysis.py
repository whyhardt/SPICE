import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kruskal, norm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

AGE_BINS   = [0, 10, 13, 15, 17, 24, 100]
AGE_LABELS = ['8-10', '10-13', '13-15', '15-17', '18-24', '25-30']
EXTRA_COEFFS = ['beta_reward', 'beta_choice']          # extra vars to test


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


def describe_groups(values, ages):
    """Return per-group summary list and raw arrays for tests."""
    summaries, raw = [], []
    for idx, label in enumerate(AGE_LABELS, 1):
        grp = values[ages == idx]
        summaries.append(dict(age_group=idx, label=label, count=len(grp),
                              mean=grp.mean()    if grp.size else np.nan,
                              median=grp.median() if grp.size else np.nan,
                              std=grp.std()      if grp.size else np.nan))
        raw.append(grp.values)
    return summaries, raw


def analyze(df, age_col='age_group', out_dir='sindy_magnitude_analysis'):
    # create output directory and any missing parents
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
        vals, ages   = df.loc[nz_mask, c], df.loc[nz_mask, age_col]
        grp_stats, g = describe_groups(vals, ages)

        rho, p_s     = spearmanr(ages, vals) if vals.size > 10 else (np.nan, np.nan)
        valid_groups = [x for x in g if x.size and x.std() > 1e-10]
        kw,  p_kw    = kruskal(*valid_groups) if len(valid_groups) >= 3 else (np.nan, np.nan)
        z,   p_jt    = jonckheere_terpstra(g)

        trend = ('Increasing' if rho > 0 else 'Decreasing') + ' with age' \
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
    _make_plots(res, df, age_col, out)
    _print_summary(res, out)
    return res


def _save_csv(res_df, path):
    flat = []
    for _, row in res_df.iterrows():
        rec = {k: row[k] for k in res_df.columns if k != 'group_stats'}
        for g in row.group_stats:
            a = g['age_group']
            for k in ('count', 'mean', 'median', 'std'):
                rec[f'group_{a}_{k}'] = g[k]
        flat.append(rec)
    pd.DataFrame(flat).to_csv(path, index=False)


def _make_plots(res_df, df, age_col, out):
    blue_colors = ['#E3F2FD', '#B6D7FF', '#7BB3F0F0',
                   '#4A90E2', '#2E5BBA', '#1A237E']
    top6 = res_df.nsmallest(6, 'jt_p')

    fig_v, axs_v = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (_, row) in zip(axs_v.ravel(), top6.iterrows()):
        mask       = df[row.coefficient] != 0
        vals, ages = df.loc[mask, row.coefficient], df.loc[mask, age_col]
        data       = [vals[ages == g] for g in range(1, 7)]

        v = ax.violinplot(data, positions=range(1, 7),
                          showmeans=False, showmedians=True)

        for i, body in enumerate(v['bodies']):
            body.set_facecolor(blue_colors[i]); body.set_alpha(0.6)
            body.set_edgecolor('black'); body.set_linewidth(1)
        v['cmedians'].set_color('black'); v['cmedians'].set_linewidth(2.1)

        for i, grp in enumerate(data):
            ax.scatter(np.random.normal(i+1, .04, size=len(grp)),
                       grp, alpha=.2, s=14, color='black', zorder=3)

        ax.set_title(f'{row.coefficient[:30]}...\nJT p={row.jt_p:.3g}', fontsize=9)
        ax.set_xticks(range(1, 7)); ax.set_xticklabels(AGE_LABELS, rotation=45)
        ax.set_xlabel('Age Group'); ax.set_ylabel('Coefficient Magnitude')
        ax.grid(True, alpha=.8)
    for extra_ax in axs_v.ravel()[len(top6):]:
        extra_ax.axis('off')
    fig_v.tight_layout()
    fig_v.savefig(out/'sindy_coefficient_age_trends.png', dpi=300)
    plt.close(fig_v)

    fig_b, axs_b = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (_, row) in zip(axs_b.ravel(), top6.iterrows()):
        mask       = df[row.coefficient] != 0
        vals, ages = df.loc[mask, row.coefficient], df.loc[mask, age_col]
        data       = [vals[ages == g] for g in range(1, 7)]

        bp = ax.boxplot(data, positions=range(1, 7),
                        showmeans=False, patch_artist=True)

        for i, box in enumerate(bp['boxes']):               # colour boxes
            box.set_facecolor(blue_colors[i]); box.set_alpha(0.6)
            box.set_edgecolor('black'); box.set_linewidth(1)

        for med in bp['medians']:                           # bold medians
            med.set_color('black'); med.set_linewidth(2.1)

        for part in ('whiskers', 'caps'):                   # darker whisk/caps
            for line in bp[part]:
                line.set_color('black'); line.set_linewidth(1)

        for i, grp in enumerate(data):                      # jittered points
            ax.scatter(np.random.normal(i+1, .04, size=len(grp)),
                       grp, alpha=.2, s=14, color='black', zorder=3)

        ax.set_title(f'{row.coefficient[:30]}...\nJT p={row.jt_p:.3g}', fontsize=9)
        ax.set_xticks(range(1, 7)); ax.set_xticklabels(AGE_LABELS, rotation=45)
        ax.set_xlabel('Age Group'); ax.set_ylabel('Coefficient Magnitude')
        ax.grid(True, alpha=.8)
    for extra_ax in axs_b.ravel()[len(top6):]:
        extra_ax.axis('off')
    fig_b.tight_layout()
    fig_b.savefig(out/'sindy_coefficient_age_trends_box.png', dpi=300)
    plt.close(fig_b)

    top15 = res_df.nsmallest(15, 'jt_p')
    heat  = pd.DataFrame({row.coefficient:
                          [g['mean'] for g in row.group_stats]
                          for _, row in top15.iterrows()}).T
    heat.columns = AGE_LABELS

    plt.figure(figsize=(12, 8))
    sns.heatmap(heat, annot=True, cmap='RdBu_r', center=0,
                fmt='.3f', cbar_kws={'label': 'Mean Coefficient'})
    plt.title('SINDy / beta means by age group (top-15 JT)')
    plt.tight_layout()
    plt.savefig(out/'sindy_coefficient_heatmap.png', dpi=300)
    plt.close()


def _print_summary(res, out):
    print(f'Total coefficients analysed:            {len(res)}')
    print(f'Significant Spearman (p<0.05):          {(res.spearman_p < 0.05).sum()}')
    print(f'Significant Kruskal-Wallis (p<0.05):    {(res.kruskal_p < 0.05).sum()}')
    print(f'Significant Jonckheere-Terpstra:        {(res.jt_p < 0.05).sum()}')
    print('\nTOP 10 by JT p-value')
    print(res[['coefficient', 'spearman_rho', 'spearman_p',
               'jt_p', 'trend']].head(10).to_string(index=False))


if __name__ == '__main__':
    DATA = '/Users/martynaplomecka/closedloop_rl/final_df_sindy_analysis_with_metrics.csv'
    df   = pd.read_csv(DATA)
    df['age_group'] = pd.cut(df.Age, bins=AGE_BINS, labels=range(1, 7),
                             include_lowest=True).astype(int)
    analyze(df,
            out_dir='/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis/plots/sindy_coefficients_analysis')
