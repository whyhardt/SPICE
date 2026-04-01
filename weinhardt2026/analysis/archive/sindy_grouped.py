import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns
from scipy import stats

def get_all_sindy_coefficients():
    return [
        'x_learning_rate_reward_c_value_reward',
        'x_value_reward_not_chosen_x_value_reward_not_chosen',
        'x_value_reward_not_chosen_c_value_choice',
        'x_learning_rate_reward_x_learning_rate_reward',
        'x_learning_rate_reward_1',
        'x_value_reward_not_chosen_c_reward_chosen',
        'x_learning_rate_reward_c_value_choice',
        'x_value_choice_not_chosen_1',
        'x_value_choice_chosen_c_value_reward',
        'x_value_choice_chosen_1',
        'x_value_choice_not_chosen_c_value_reward',
        'x_learning_rate_reward_c_reward_chosen',
        'x_value_choice_chosen_x_value_choice_chosen',
        'x_value_reward_not_chosen_1',
        'x_value_choice_not_chosen_x_value_choice_not_chosen',
        'beta_reward',
        'beta_choice',
        'params_x_learning_rate_reward',
        'params_x_value_reward_not_chosen',
        'params_x_value_choice_chosen',
        'params_x_value_choice_not_chosen'
    ]


def create_presence_matrix(df, threshold=1e-4):
    coeffs = get_all_sindy_coefficients()
    available = [c for c in coeffs if c in df.columns]
    presence = df[['participant_id', 'Age'] + available].copy()
    for c in available:
        presence[f"{c}_present"] = (df[c].abs() > threshold).astype(int)
    return presence, available


def plot_mechanism_prevalence(df, out_dir):
    _, coeffs = create_presence_matrix(df)
    prevalences = [(c, (df[c].abs() > 1e-4).mean() * 100) for c in coeffs]
    prevalences.sort(key=lambda x: x[1], reverse=True)
    names = [c.replace('x_', '').replace('_c_', ' ← ').replace('_x_', ' persist').replace('_', ' ') for c, _ in prevalences]
    values = [v for _, v in prevalences]

    plt.figure(figsize=(12, len(names) * 0.3))
    y = np.arange(len(names))
    plt.barh(y, values)
    plt.yticks(y, names, fontsize=8)
    plt.xlabel('Percent of Participants')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(out_dir / 'sindy_mechanism_prevalence.png', dpi=300)
    plt.close()


def plot_age_vs_mechanism_presence(df, out_dir):
    # Numeric categories 1,2,3 without ordering assumptions
    df['Age_Category'] = pd.cut(df['Age'], bins=3, labels=False) + 1
    _, coeffs = create_presence_matrix(df)
    age_cats = df['Age_Category'].unique()

    # Prepare data
    data = []
    for age_cat in age_cats:
        sub = df[df['Age_Category'] == age_cat]
        for c in coeffs:
            rate = (sub[c].abs() > 1e-4).mean() * 100
            data.append({'Age_Category': age_cat, 'mechanism': c, 'rate': rate})
    age_df = pd.DataFrame(data)

    x = np.arange(len(coeffs))
    width = 0.8 / len(age_cats)  # adjust bar width based on number of categories

    plt.figure(figsize=(14, 6))
    for i, cat in enumerate(age_cats):
        rates = age_df[age_df['Age_Category'] == cat]['rate'].values
        plt.bar(x + i * width, rates, width, label=str(cat))
    plt.xticks(x + width * (len(age_cats)-1)/2, [c.replace('x_', '').replace('_', ' ') for c in coeffs],
               rotation=45, ha='right', fontsize=8)
    plt.ylabel('% Present')
    plt.legend(title='Age Category')
    plt.tight_layout()
    plt.savefig(out_dir / 'age_vs_mechanism_presence.png', dpi=300)
    plt.close()


def create_age_heatmap(df, out_dir):
    df['Age_Category'] = pd.cut(df['Age'], bins=3, labels=False) + 1
    _, coeffs = create_presence_matrix(df)
    age_cats = df['Age_Category'].unique()

    data = []
    for c in coeffs:
        for age_cat in age_cats:
            sub = df[df['Age_Category'] == age_cat]
            pct = (sub[c].abs() > 1e-4).mean()
            data.append({'mech': c, 'age_cat': age_cat, 'pct': pct})
    heat_df = pd.DataFrame(data)

    # Pivot and preserve original category order
    heat = heat_df.pivot(index='mech', columns='age_cat', values='pct')
    heat = heat[age_cats]

    plt.figure(figsize=(8, len(coeffs) * 0.2 + 2))
    sns.heatmap(heat, annot=True, fmt='.2f')
    plt.tight_layout()
    plt.savefig(out_dir / 'age_mechanism_heatmap.png', dpi=300)
    plt.close()


def analyze_age_correlations(df, out_dir):
    df['Age_Category'] = pd.cut(df['Age'], bins=3, labels=False) + 1
    _, coeffs = create_presence_matrix(df)
    results = []
    for c in coeffs:
        pres = (df[c].abs() > 1e-4).astype(int)
        corr, p = stats.spearmanr(df['Age'], pres)
        results.append({'mechanism': c, 'corr': corr, 'p_value': p})
    pd.DataFrame(results).to_csv(out_dir / 'age_mechanism_correlations.csv', index=False)


def main():
    df = pd.read_csv('AAAAsindy_analysis_with_metrics.csv')
    df = df.dropna(subset=['Age'])
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age'])

    out_dir = Path('/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots/sindy_presence')
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_mechanism_prevalence(df, out_dir)
    plot_age_vs_mechanism_presence(df, out_dir)
    create_age_heatmap(df, out_dir)
    analyze_age_correlations(df, out_dir)

if __name__ == '__main__':
    main()
