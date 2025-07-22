import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path='final_df_sindy_analysis_with_metrics.csv'):
    """
    Load data and extract SINDY coefficients + beta_reward & beta_choice for logistic regression.
    """
    df = pd.read_csv(csv_path)
    # Pick up all x_ columns
    sindy_cols = [c for c in df.columns if c.startswith('x_')]
    # Also include these two if present
    for extra in ('beta_reward', 'beta_choice'):
        if extra in df.columns:
            sindy_cols.append(extra)

    return df, sindy_cols


def clean_coefficient_name(col):
    return (col.replace('x_', '').replace('_', ' ').title())[:50]


def perform_logistic_regression_analysis(df, cols, output_dir):
    """
    Runs logistic regression on each coefficient vs age, prints never/always present,
    then plots only the middle group sorted by |β|.
    """
    # 1) filter missing ages
    df_clean = df[df['Age'].notna()].copy()
    if df_clean.empty:
        raise ValueError("No valid ages found.")
    age_min, age_max = df_clean['Age'].min(), df_clean['Age'].max()
    print(f"Using {len(df_clean)} participants; Age range {age_min:.1f}–{age_max:.1f}")

    # 2) standardize
    scaler = StandardScaler()
    age_std = scaler.fit_transform(df_clean[['Age']]).flatten()

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

        # fit
        solver = 'saga' if rate < 0.1 else 'liblinear'
        its = 2000 if rate < 0.1 else 1000
        model = LogisticRegression(solver=solver, max_iter=its, random_state=0)
        model.fit(age_std[mask].reshape(-1, 1), y)
        beta_age = model.coef_[0][0]

        # LRT
        p_hat = model.predict_proba(age_std[mask].reshape(-1, 1))[:, 1]
        eps = 1e-15
        ll = np.sum(y * np.log(np.clip(p_hat, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p_hat, eps, 1 - eps)))
        p0 = y.mean()
        ll0 = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
        lr = -2 * (ll0 - ll)
        pval = 1 - chi2.cdf(max(0, lr), df=1)

        # stars
        if pval < 1e-3:
            star = '***'
        elif pval < 1e-2:
            star = '**'
        elif pval < 5e-2:
            star = '*'
        else:
            star = 'ns'

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

    # only those with a real regression (note is NaN)
    mask_reg = df_res['note'].isna()
    print("\nLogistic regressions (0 < rate < 1):")
    for _, r in df_res[mask_reg].iterrows():
        print(f" - {r['coefficient_clean']}: p={r['p_value']:.4f}, sig={r['significance']}")

    os.makedirs(output_dir, exist_ok=True)
    all_csv = os.path.join(output_dir, 'sindy_age_logistic_regression_all.csv')
    df_res.to_csv(all_csv, index=False)
    print(f"\nFull results → {all_csv}")

    # never / always
    never = [clean_coefficient_name(c) for c, n in skipped if n == 'all zero']
    always = df_res.loc[df_res['note'] == 'always present', 'coefficient_clean'].tolist()
    print("\nNever-present:\n ", "\n  ".join(never))
    print("\nAlways-present:\n ", "\n  ".join(always))

    # remaining & sort by |β|
    df_rem = df_res[mask_reg].copy()
    df_rem['abs_beta'] = df_rem['beta_age'].abs()
    df_rem.sort_values('abs_beta', ascending=False, inplace=True)
    df_rem.drop('abs_beta', axis=1, inplace=True)

    rem_csv = os.path.join(output_dir, 'sindy_age_logistic_regression_remaining.csv')
    df_rem.to_csv(rem_csv, index=False)
    print(f"\nRemaining results → {rem_csv}")

    # plot remaining
    if df_rem.empty:
        print("No remaining to plot.")
    else:
        print(f"\nPlotting {len(df_rem)} remaining (sorted by |β|)…")
        create_beta_bar_plot(df_rem, output_dir)
        create_logistic_regression_plot(df_rem, output_dir, age_min, age_max)

    return df_res


def create_beta_bar_plot(df, output_dir):
    """
    Bar‑plot of β sorted by magnitude.
    """
    cmap = {'***': '#FF0000', '**': '#FFA500', '*': '#FFD700', 'ns': '#999999'}
    colors = df['significance'].map(cmap)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df['coefficient_clean'], df['beta_age'], color=colors, edgecolor='black')
    ax.axhline(0, linestyle='--', color='black')
    ax.set_ylabel('Age Effect (β)')
    ax.set_title('Age Effect (β) by Coefficient (remaining)')
    ax.set_xticklabels(df['coefficient_clean'], rotation=45, ha='right', fontsize=8)

    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap[s], edgecolor='black') for s in ['***', '**', '*', 'ns']]
    ax.legend(handles, ['***', '**', '*', 'ns'], title='Significance', loc='best')

    plt.tight_layout()
    out = os.path.join(output_dir, 'beta_vs_coefficient_remaining_sorted.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  • β-bar plot → {out}")


def create_logistic_regression_plot(df, output_dir, age_min, age_max):
    """
    Curve‑plot for the remaining sorted coefficients.
    """
    cmap = {'***': '#FF0000', '**': '#FFA500', '*': '#FFD700', 'ns': '#999999'}
    ages = np.linspace(age_min, age_max, 200)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Age-Dependent Presence (remaining)', y=0.98)

    for ax, (_, row) in zip(axes.flatten(), df.iterrows()):
        beta_age = row['beta_age']
        # logistic curve
        age_std = (ages - ages.mean()) / ages.std()
        p = 1 / (1 + np.exp(-beta_age * age_std))
        ax.plot(ages, p, color=cmap[row['significance']])
        ax.set_title(row['coefficient_clean'], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Age')
        ax.set_ylabel('Prob.')

    plt.tight_layout()
    out = os.path.join(output_dir, 'logistic_curves_remaining_sorted.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  • curves plot → {out}")


def main(
    csv_path='final_df_sindy_analysis_with_metrics.csv',
    output_dir='/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis/plots/logistic_regression'
):
    df, cols = load_and_prepare_data(csv_path)
    perform_logistic_regression_analysis(df, cols, output_dir)


if __name__ == '__main__':
    main()