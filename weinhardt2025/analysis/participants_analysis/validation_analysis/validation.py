import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")

output_dir = "//Users/martynaplomecka/closedloop_rl/analysis/participants_analysis/validation_analysis/"
os.makedirs(output_dir, exist_ok=True)


try:
    df = pd.read_csv("new_behavioral_metrics_all_simulated.csv")
    print(f"Loaded data with {len(df)} participants")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Columns: {list(df.columns)}")
except FileNotFoundError:
    print("Error: file not found.")
    exit()

# Behavioral metrics to plot
behavioral_metrics = [
    'stay_after_reward',
    'switch_rate',
    'perseveration',
    'stay_after_plus_plus',
    'stay_after_plus_minus',
    'stay_after_minus_plus',
    'stay_after_minus_minus',
    'avg_reward'
]

# Order and colors for datasets
ordered_datasets = ['True', 'ApBr', 'Benchmark', 'RNN', 'SPICE']
dataset_colors = {
    'True': '#FDE725',
    'RNN': '#B5DE2B',
    'SPICE': '#6ECE58',
    'ApBr': '#1F9E89',
    'Benchmark': '#26828E'
}


fig, axes = plt.subplots(4, 2, figsize=(10, 8))
axes = axes.flatten()

for i, metric in enumerate(behavioral_metrics):
    ax = axes[i]
    plot_data, plot_labels, plot_colors = [], [], []

    for dataset in ordered_datasets:
        data = df[df['dataset'] == dataset][metric].dropna()
        if len(data) > 0:
            plot_data.append(data.values)
            plot_labels.append(dataset)
            plot_colors.append(dataset_colors.get(dataset))

    if plot_data:
        parts = ax.violinplot(plot_data,
                             positions=range(len(plot_data)),
                             showmeans=False,
                             showmedians=True)
        # Style violins
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[j])
            pc.set_alpha(0.27)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        for key in ['cmedians', 'cmaxes', 'cmins', 'cbars']:
            parts[key].set_color('grey')
            parts[key].set_linewidth(1)

        # Jittered scatter
        for j, data in enumerate(plot_data):
            x_positions = np.random.normal(j, 0.04, size=len(data))
            ax.scatter(x_positions, data,
                       color=plot_colors[j],
                       alpha=0.5,
                       s=4,
                       edgecolors='black',
                       linewidths=0.2,
                       zorder=4)

        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, rotation=45)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
    else:
        ax.set_title(metric.replace("_", " ").title() + ' - No Data', fontsize=10)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_ylim(-0.1, 1.1)


plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.96])
plt.suptitle('Behavioral Metrics Across Datasets', fontsize=12, fontweight='bold', y=0.98)

save_path = os.path.join(output_dir, 'behavioral_metrics_across_datasets.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path}")

plt.show()

for metric in behavioral_metrics:
    print(f"\n{metric.replace('_', ' ').upper()}:")
    print("-" * 50)
    for dataset in ordered_datasets:
        data = df[df['dataset'] == dataset][metric].dropna()
        if len(data) > 0:
            print(f"{dataset:>10}: Mean={data.mean():.4f}, SEM={data.sem():.4f}, SD={data.std():.4f}, n={len(data)}")
