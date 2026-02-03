import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
import warnings

# ─── Use MathText (no external LaTeX) ───────────────────────────────────────────
mpl.rcParams['text.usetex']    = False
mpl.rcParams['font.family']    = 'arial'
mpl.rcParams['font.size']      = 16  # Increased base font size

warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path='final_df_sindy_analysis_with_metrics.csv'):
    df = pd.read_csv(csv_path)
    sindy_cols = [c for c in df.columns if c.startswith('x_')]
    for extra in ('beta_reward', 'beta_choice'):
        if extra in df.columns:
            sindy_cols.append(extra)
    return df, sindy_cols


def clean_coefficient_name(col):
    return (col.replace('x_', '').replace('_', ' ').title())[:50]


def perform_logistic_regression_analysis(df, cols, output_dir):
    df_clean = df[df['Age'].notna()].copy()
    if df_clean.empty:
        raise ValueError("No valid ages found.")
    age_min, age_max = df_clean['Age'].min(), df_clean['Age'].max()
    print(f"Using {len(df_clean)} participants; Age range {age_min:.1f}–{age_max:.1f}")

    age_std = StandardScaler().fit_transform(df_clean[['Age']]).flatten()
    results, skipped = [], []

    for col in cols:
        vals = df_clean[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 10:
            skipped.append((col, f"<10 obs ({mask.sum()})"))
            continue

        y = (vals[mask] != 0).astype(int)
        rate = y.mean()
        if rate == 0:
            skipped.append((col, "all zero"))
            continue
        if rate == 1.0:
            results.append(dict(
                coefficient=col,
                beta_age=np.nan,
                p_value=np.nan,
                n_nonzero=int(y.sum()),
                n_total=int(len(y)),
                coefficient_clean=clean_coefficient_name(col),
                significance='ns',
                note='always present'
            ))
            continue

        solver = 'saga' if rate < 0.1 else 'liblinear'
        its    = 2000   if rate < 0.1 else 1000
        model  = LogisticRegression(solver=solver, max_iter=its, random_state=0)
        model.fit(age_std[mask].reshape(-1, 1), y)
        beta_age = model.coef_[0][0]

        p_hat = model.predict_proba(age_std[mask].reshape(-1, 1))[:, 1]
        eps   = 1e-15
        ll    = np.sum(y * np.log(np.clip(p_hat, eps, 1 - eps)) +
                       (1 - y) * np.log(np.clip(1 - p_hat, eps, 1 - eps)))
        p0    = y.mean()
        ll0   = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
        lr    = -2 * (ll0 - ll)
        pval  = 1 - chi2.cdf(max(0, lr), df=1)

        if pval < 1e-3: star = '***'
        elif pval < 1e-2: star = '**'
        elif pval < 5e-2: star = '*'
        else:             star = 'ns'

        results.append(dict(
            coefficient=col,
            beta_age=beta_age,
            p_value=pval,
            n_nonzero=int(y.sum()),
            n_total=int(len(y)),
            coefficient_clean=clean_coefficient_name(col),
            significance=star
        ))

    if skipped:
        print(f"Skipped {len(skipped)} coeffs (e.g. {skipped[:3]})")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        raise ValueError("No valid regressions run.")

    mask_reg = df_res.get('note').isna()
    print("\nLogistic regressions (0 < rate < 1):")
    for _, r in df_res[mask_reg].iterrows():
        print(f" - {r['coefficient_clean']}: p={r['p_value']:.4f}, sig={r['significance']}")

    os.makedirs(output_dir, exist_ok=True)
    df_res.to_csv(os.path.join(output_dir,
                      'sindy_age_logistic_regression_all.csv'), index=False)

    never   = [clean_coefficient_name(c) for c, n in skipped if n == 'all zero']
    always  = df_res.loc[df_res.get('note')=='always present','coefficient_clean'].tolist()
    print("\nNever-present:\n ", "\n  ".join(never))
    print("\nAlways-present:\n ", "\n  ".join(always))

    df_rem = df_res[mask_reg].copy()
    df_rem['abs_beta'] = df_rem['beta_age'].abs()
    df_rem.sort_values('abs_beta', ascending=False, inplace=True)
    df_rem.drop('abs_beta', axis=1, inplace=True)
    df_rem.to_csv(os.path.join(output_dir,
                    'sindy_age_logistic_regression_remaining.csv'),
                  index=False)

    if df_rem.empty:
        print("No remaining to plot.")
    else:
        print(f"\nPlotting {len(df_rem)} remaining (sorted by |β|)…")
        create_beta_bar_plot(df_rem, output_dir)
        create_logistic_regression_plot(df_rem, output_dir, age_min, age_max)

    return df_res


def create_beta_bar_plot(df, output_dir):
    """
    Bar-plot of β sorted by magnitude with two-line x-tick labels
    (description on line 1, SINDy‐term shorthand on line 2) and extra‐large fonts.
    """
    cmap = {'***':'#FF0000', '**':'#FFA500', '*':'#FFD700', 'ns':'#999999'}
    colors = df['significance'].map(cmap)

    fig, ax = plt.subplots(figsize=(20, 12))  # Increased figure size
    ax.bar(np.arange(len(df)), df['beta_age'], color=colors, edgecolor='black')
    ax.axhline(0, linestyle='--', color='black')

    # two-line labels: Description : SINDy‐term shorthand
    labels = [
        "Unchosen choice\n value update:\n"  + r"$Q_{\mathrm{reward}}$",
        "Unchosen reward\n value update:\n"   + r"$Q_{\mathrm{reward}}$",
        "Learning rate\n update:\n"  + r"$\alpha$",
        "Unchosen reward\n value update:\n"     + r"$Q_{\mathrm{choice}}$",
        "Unchosen choice\n value update:\n"          + r"$Q_{\mathrm{choice}}$"
    ]

    # Shift the tick positions slightly to the right for better label alignment
    ax.set_xticks(np.arange(len(labels)) + 0.30)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=37)
    ax.set_ylabel(r'Age effect ($\beta$)', fontsize=42)
    ax.set_title('Age Effect (β) by Sindy Coefficient', fontsize=45)
    ax.tick_params(axis='y', labelsize=37)

    # Legend with larger fonts
    handles = [plt.Rectangle((0,0),1,1, facecolor=cmap[s], edgecolor='black')
               for s in ['***','**','*','ns']]
    legend = ax.legend(handles, ['***','**','*','ns'],
                       title='Significance', loc='best')
    plt.setp(legend.get_title(), fontsize=34)
    plt.setp(legend.get_texts(), fontsize=34)

    plt.tight_layout()
    out = os.path.join(output_dir, 'beta_vs_coefficient_shorthand.png')
    plt.savefig(out, dpi=500, bbox_inches='tight')
    plt.close()
    print(f"  • β-bar plot w/ two-line shorthand labels → {out}")



def create_logistic_regression_plot(df, output_dir, age_min, age_max):
    cmap = {'***':'#FF0000', '**':'#FFA500', '*':'#FFD700', 'ns':'#999999'}
    ages = np.linspace(age_min, age_max, 200)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Increased figure size
    fig.suptitle('Age-Dependent Presence (remaining)', fontsize=32, y=0.98)  # Increased from 24

    for ax, (_, row) in zip(axes.flatten(), df.iterrows()):
        beta_age = row['beta_age']
        age_std  = (ages - ages.mean()) / ages.std()
        p        = 1 / (1 + np.exp(-beta_age * age_std))
        ax.plot(ages, p, color=cmap[row['significance']], linewidth=3)  # Added linewidth
        ax.set_title(row['coefficient_clean'], fontsize=22)  # Increased from 18
        ax.set_ylim(0, 1)
        ax.set_xlabel('Age', fontsize=24)  # Increased from 16
        ax.set_ylabel('Prob.', fontsize=20)  # Increased from 16
        ax.tick_params(axis='both', labelsize=18)  # Increased from 14

    plt.tight_layout()
    out = os.path.join(output_dir, 'logistic_curves_shorthand.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  • logistic curves plot → {out}")


def main(
    csv_path='final_df_sindy_analysis_with_metrics.csv',
    output_dir='/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis/plots/logistic_regression'
):
    df, cols = load_and_prepare_data(csv_path)
    perform_logistic_regression_analysis(df, cols, output_dir)


if __name__ == '__main__':
    main()