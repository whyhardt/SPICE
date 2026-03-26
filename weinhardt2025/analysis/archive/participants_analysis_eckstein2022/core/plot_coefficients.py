import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

AGE_BINS     = [0, 10, 13, 15, 17, 24, 100]
AGE_LABELS   = ['8-10', '10-13', '13-15', '15-17', '18-24', '25-30']
EXTRA_COEFFS = ['beta_reward', 'beta_choice']

def plot_sindy_coefficients(df, age_col='age_group', out_dir='sindy_plots', n_top=None):
    # Prepare output directory
    out = Path(out_dir).parent if out_dir.endswith('.csv') else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Identify all coefficient columns
    coeff_cols = [c for c in df.columns if c.startswith('x_')] + \
                 [c for c in EXTRA_COEFFS if c in df.columns]

    # Keep those with ≥10 non-zero entries
    keep = [c for c in coeff_cols if (df[c] != 0).sum() >= 10]
    if not keep:
        raise ValueError('No coefficients with at least 10 non-zero values found.')

    plot_coeffs = keep if n_top is None else keep[:n_top]
    n = len(plot_coeffs)

    # Grid layout for full plots
    n_cols = min(5, n)
    n_rows = int(np.ceil(n / n_cols))

    # Color palette
    blue_colors = ['#E3F2FD', '#B6D7FF', '#7BB3F0', '#4A90E2', '#2E5BBA', '#1A237E']

    # ---- Violin plots (all) ----
    fig_v, axs_v = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axs_v = np.array(axs_v).ravel()

    for i, coeff in enumerate(plot_coeffs):
        ax = axs_v[i]
        mask = df[coeff] != 0
        vals = df.loc[mask, coeff]
        ages = df.loc[mask, age_col]
        data = [vals[ages == g].values for g in range(1, len(AGE_LABELS)+1)]

        v = ax.violinplot(data, positions=range(1, len(data)+1),
                          showmeans=False, showmedians=True)
        for j, body in enumerate(v['bodies']):
            if j < len(blue_colors):
                body.set_facecolor(blue_colors[j])
                body.set_alpha(0.6)
            body.set_edgecolor('black')
            body.set_linewidth(1)
        v['cmedians'].set_color('black')
        v['cmedians'].set_linewidth(2)

        for j, grp in enumerate(data):
            if grp.size:
                ax.scatter(np.random.normal(j+1, 0.04, size=len(grp)),
                           grp, alpha=0.2, s=14, color='black', zorder=3)

        ax.set_title(coeff, fontsize=9)
        ax.set_xticks(range(1, len(data)+1))
        ax.set_xticklabels(AGE_LABELS, rotation=45)
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Coefficient')
        ax.grid(True, alpha=0.3)

    for j in range(n, n_rows*n_cols):
        axs_v[j].axis('off')

    fig_v.tight_layout()
    fig_v.savefig(out / 'sindy_coeffs_violin_all.png', dpi=300, bbox_inches='tight')
    plt.close(fig_v)

    print(f"Saved violin plot to {out / 'sindy_coeffs_violin_all.png'}")

    # ---- Box plots (all) ----
    fig_b, axs_b = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axs_b = np.array(axs_b).ravel()

    for i, coeff in enumerate(plot_coeffs):
        ax = axs_b[i]
        mask = df[coeff] != 0
        vals = df.loc[mask, coeff]
        ages = df.loc[mask, age_col]
        data = [vals[ages == g].values for g in range(1, len(AGE_LABELS)+1)]

        bp = ax.boxplot(data, positions=range(1, len(data)+1),
                        showmeans=False, patch_artist=True)
        for j, box in enumerate(bp['boxes']):
            if j < len(blue_colors):
                box.set_facecolor(blue_colors[j])
                box.set_alpha(0.6)
            box.set_edgecolor('black')
            box.set_linewidth(1)
        for part in ['whiskers', 'caps', 'medians']:
            for line in bp.get(part, []):
                line.set_color('black')
                line.set_linewidth(1.5)

        for j, grp in enumerate(data):
            if grp.size:
                ax.scatter(np.random.normal(j+1, 0.04, size=len(grp)),
                           grp, alpha=0.2, s=14, color='black', zorder=3)

        ax.set_title(coeff, fontsize=9)
        ax.set_xticks(range(1, len(data)+1))
        ax.set_xticklabels(AGE_LABELS, rotation=45)
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Coefficient')
        ax.grid(True, alpha=0.3)

    for j in range(n, n_rows*n_cols):
        axs_b[j].axis('off')

    fig_b.tight_layout()
    fig_b.savefig(out / 'sindy_coeffs_box_all.png', dpi=500, bbox_inches='tight')
    plt.close(fig_b)

    print(f"Saved box plot to   {out / 'sindy_coeffs_box_all.png'}")

    # ---- Box plots (selected indices, 2 cols × 3 rows) ----
    selected_1based = [1, 5, 8, 12, 13]
    sel_idx = [i-1 for i in selected_1based if 1 <= i <= n]
    sel_coeffs = [plot_coeffs[i] for i in sel_idx]

    # Force 3 rows × 2 columns, larger size
    sel_rows, sel_cols = 3, 2
    fig_s, axs_s = plt.subplots(sel_rows, sel_cols, figsize=(10, 12))
    axs_s = np.array(axs_s).ravel()

    for i, coeff in enumerate(sel_coeffs):
        ax = axs_s[i]
        mask = df[coeff] != 0
        vals = df.loc[mask, coeff]
        ages = df.loc[mask, age_col]
        data = [vals[ages == g].values for g in range(1, len(AGE_LABELS)+1)]

        bp = ax.boxplot(data, positions=range(1, len(data)+1),
                        showmeans=False, patch_artist=True)
        for j, box in enumerate(bp['boxes']):
            color = blue_colors[j] if j < len(blue_colors) else 'white'
            box.set_facecolor(color)
            box.set_alpha(0.6)
            box.set_edgecolor('black')
            box.set_linewidth(1)
        for part in ['whiskers', 'caps', 'medians']:
            for line in bp.get(part, []):
                line.set_color('black')
                line.set_linewidth(1.5)

        for j, grp in enumerate(data):
            if grp.size:
                ax.scatter(np.random.normal(j+1, 0.04, size=len(grp)),
                           grp, alpha=0.2, s=14, color='black', zorder=3)

        ax.set_title(coeff, fontsize=9)
        ax.set_xticks(range(1, len(data)+1))
        ax.set_xticklabels(AGE_LABELS, rotation=45)
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Coefficient')
        ax.grid(True, alpha=0.3)

    # Turn off any unused axes (6 slots, 5 plots)
    for j in range(len(sel_coeffs), sel_rows*sel_cols):
        axs_s[j].axis('off')

    fig_s.tight_layout()
    fig_s.savefig(out / 'sindy_coeffs_box_selected.png', dpi=500, bbox_inches='tight')
    plt.close(fig_s)

    print(f"Saved subset box plot to {out / 'sindy_coeffs_box_selected.png'}")

    return plot_coeffs

if __name__ == '__main__':
    DATA = '/Users/martynaplomecka/closedloop_rl/final_df_sindy_analysis_with_metrics.csv'
    df = pd.read_csv(DATA)
    df['age_group'] = pd.cut(
        df['Age'],
        bins=AGE_BINS,
        labels=range(1, len(AGE_LABELS)+1),
        include_lowest=True
    ).astype(int)

    plot_sindy_coefficients(
        df,
        out_dir=(
            '/Users/martynaplomecka/closedloop_rl/'
            'analysis/participants_analysis/plots/'
            'plotted_coefficients/'
        ),
        n_top=None
    )
