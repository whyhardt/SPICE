#  relationship between participant age and model performance metrics.
# It creates scatter plots showing correlations between age and various model evaluation metrics (likelihood, BIC, AIC).
#THIS IS ONLY THE SANITY CHECK TO MAKE SURE THAT AGE DOESNT DRIVE SINDY PERFORMANCE


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path
from scipy import stats

def calculate_correlation(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if sum(mask) < 3:
        return "Insufficient data", 1.0

    r, p = stats.pearsonr(x[mask], y[mask])
    significance = ""
    if p < 0.001:
        significance = "***"
    elif p < 0.01:
        significance = "**"
    elif p < 0.05:
        significance = "*"

    return f"r = {r:.3f}{significance}", p

def main():
    parser = argparse.ArgumentParser(
        description="Sanity check: relationship between participant age and model performance metrics."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/Users/martynaplomecka/closedloop_rl/AAAAsindy_analysis_with_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots/age_vs_model_metrics",
    )
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    model_metrics = [
        'nll_spice',
        'nll_rnn',
        'trial_likelihood_spice',
        'trial_likelihood_rnn',
        'bic_spice',
        'aic_spice'
    ]

    metric_names = {
        'nll_spice': 'Negative Log-Likelihood (SPICE)',
        'nll_rnn': 'Negative Log-Likelihood (RNN)',
        'trial_likelihood_spice': 'Trial Likelihood (SPICE)',
        'trial_likelihood_rnn': 'Trial Likelihood (RNN)',
        'bic_spice': 'BIC (SPICE)',
        'aic_spice': 'AIC (SPICE)'
    }

    # Plot 1: Overview plot with all metrics
    plt.figure(figsize=(18, 12))

    for i, metric in enumerate(model_metrics):
        ax = plt.subplot(2, 3, i + 1)

        valid_data = df.dropna(subset=[metric, 'Age'])

        if len(valid_data) > 2:
            sns.scatterplot(
                x='Age',
                y=metric,
                data=valid_data,
                alpha=0.7,
                edgecolor='w',
                s=80,
                ax=ax
            )

            sns.regplot(
                x='Age',
                y=metric,
                data=valid_data,
                scatter=False,
                ci=95,
                line_kws={'color': 'red'},
                ax=ax
            )

            corr_text, p_value = calculate_correlation(
                valid_data['Age'],
                valid_data[metric]
            )

            ax.text(
                0.05,
                0.95,
                corr_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

            if p_value < 0.05:
                slope, intercept, _, _, _ = stats.linregress(
                    valid_data['Age'].dropna(),
                    valid_data[metric].dropna()
                )
                equation = f"y = {slope:.3f}x + {intercept:.3f}"
                ax.text(
                    0.05,
                    0.87,
                    equation,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                transform=ax.transAxes,
                fontsize=12,
                ha='center'
            )

        ax.set_title(metric_names[metric], fontsize=14)
        ax.set_xlabel('Age (years)', fontsize=12)
        ax.set_ylabel(metric_names[metric], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'age_vs_model_metrics_overview.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # Plot 2: Individuald plots for each metric
    for metric in model_metrics:
        plt.figure(figsize=(8, 6))

        valid_data = df.dropna(subset=[metric, 'Age'])

        if len(valid_data) > 2:
            sns.scatterplot(
                x='Age',
                y=metric,
                data=valid_data,
                alpha=0.7,
                edgecolor='w',
                s=100
            )

            sns.regplot(
                x='Age',
                y=metric,
                data=valid_data,
                scatter=False,
                ci=95,
                line_kws={'color': 'red'}
            )

            corr_text, p_value = calculate_correlation(
                valid_data['Age'],
                valid_data[metric]
            )

            plt.text(
                0.05,
                0.95,
                corr_text,
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

            if p_value < 0.05:
                slope, intercept, _, _, _ = stats.linregress(
                    valid_data['Age'].dropna(),
                    valid_data[metric].dropna()
                )
                equation = f"y = {slope:.3f}x + {intercept:.3f}"
                plt.text(
                    0.05,
                    0.87,
                    equation,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )

            plt.text(
                0.05,
                0.79,
                f"n = {len(valid_data)}",
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
        else:
            plt.text(
                0.5,
                0.5,
                "Insufficient data",
                transform=plt.gca().transAxes,
                fontsize=12,
                ha='center'
            )

        plt.title(f"Age vs {metric_names[metric]}", fontsize=14)
        plt.xlabel('Age (years)', fontsize=12)
        plt.ylabel(metric_names[metric], fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'age_vs_{metric}.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # Correlation summary
    correlation_data = []
    for metric in model_metrics:
        valid_data = df.dropna(subset=[metric, 'Age'])
        if len(valid_data) > 2:
            r, p = stats.pearsonr(
                valid_data['Age'],
                valid_data[metric]
            )
            correlation_data.append({
                'Metric': metric_names[metric],
                'Correlation': r,
                'p-value': p,
                'n': len(valid_data)
            })

    corr_summary = pd.DataFrame(correlation_data)
    print("\nAge vs Model Metrics - Correlation Summary:")
    print(corr_summary.round(3))

    corr_summary.to_csv(
        os.path.join(output_dir, 'age_correlations_summary.csv'),
        index=False
    )

if __name__ == "__main__":
    main()
