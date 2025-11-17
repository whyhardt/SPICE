import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import spearmanr

# List of raw SINDy coefficient names
SINDY_COEFFS = [
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
    'params_x_learning_rate_reward',
    'params_x_value_reward_not_chosen',
    'params_x_value_choice_chosen',
    'params_x_value_choice_not_chosen'
]

# Behavioral and demographic metrics
METRICS = ['Age', 'switch_rate', 'stay_after_reward', 'perseveration', 'avg_reward']


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=METRICS)
    available = [c for c in SINDY_COEFFS if c in df.columns]
    return df, available


def scale_matrix(matrix):
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)


def cluster_features(df, available_coeffs, n_clusters=3, method='kmeans'):
    X = df[available_coeffs].fillna(0).values
    Xs = scale_matrix(X)
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=0)
        labels = model.fit_predict(Xs)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(Xs)
    return labels


def plot_dendrogram(df, available_coeffs, out_dir):
    Xs = scale_matrix(df[available_coeffs].fillna(0).values)
    Z = linkage(Xs, method='ward')
    plt.figure(figsize=(10,6))
    dendrogram(Z, labels=df['participant_id'].astype(str).tolist(), leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.tight_layout()
    plt.savefig(out_dir/'dendrogram.png', dpi=300)
    plt.close()


def summarize_clusters(df, out_dir):
    # Boxplots of Age and behavioral metrics by cluster
    for col in METRICS:
        plt.figure(figsize=(6,4))
        sns.boxplot(x='cluster_k', y=col, data=df)
        plt.title(f'{col} by K-Means Cluster')
        plt.tight_layout()
        plt.savefig(out_dir/f'{col}_by_kmeans_cluster.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(6,4))
        sns.boxplot(x='cluster_h', y=col, data=df)
        plt.title(f'{col} by Agglomerative Cluster')
        plt.tight_layout()
        plt.savefig(out_dir/f'{col}_by_agg_cluster.png', dpi=300)
        plt.close()


def plot_cluster_feature_correlations(df, out_dir, cluster_col='cluster_k'):
    features = METRICS[1:]  # exclude Age if desired or include Age
    corrs = []
    for feat in features:
        rho, p = spearmanr(df[cluster_col], df[feat])
        corrs.append({'Feature': feat, 'Correlation': rho, 'p-value': p, 'abs_corr': abs(rho)})
    corr_df = pd.DataFrame(corrs).sort_values('abs_corr', ascending=False)

    plt.figure(figsize=(8,6))
    colors = ['blue' if x>=0 else 'red' for x in corr_df['Correlation']]
    bars = plt.barh(corr_df['Feature'], corr_df['Correlation'], color=colors)
    for i, (p, corr) in enumerate(zip(corr_df['p-value'], corr_df['Correlation'])):
        if p<0.001: marker='***'
        elif p<0.01: marker='**'
        elif p<0.05: marker='*'
        else: marker=''
        if marker:
            x_pos = corr + (0.05 if corr>=0 else -0.05)
            plt.text(x_pos, i, marker, ha='center', va='center', fontsize=12, fontweight='bold')
    plt.axvline(0, color='gray', linestyle='-', alpha=0.7)
    plt.xlabel(f'Spearman Correlation with {cluster_col}')
    plt.title(f'Feature Correlations with {cluster_col}')
    strongest = corr_df.iloc[0]
    plt.text(0.02,0.95,f"Strongest: {strongest['Feature']} (r={strongest['Correlation']:.3f}, p={strongest['p-value']:.3f})",
             transform=plt.gca().transAxes, fontsize=9, bbox=dict(boxstyle='round',facecolor='white',alpha=0.8), va='top')
    plt.tight_layout()
    plt.savefig(out_dir/f'{cluster_col}_feature_correlations.png', dpi=300)
    plt.close()


def plot_cluster_profiles_radar(df, out_dir, cluster_col='cluster_k'):
    metrics = METRICS[1:]
    clusters = sorted(df[cluster_col].unique())
    means = df.groupby(cluster_col)[metrics].mean()
    overall_mean = df[metrics].mean()
    overall_std = df[metrics].std()

    # Prepare radar
    angles = np.linspace(0,2*np.pi,len(metrics),endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
    colors = plt.cm.viridis(np.linspace(0,1,len(clusters)))
    for i, cl in enumerate(clusters):
        vals = ((means.loc[cl] - overall_mean)/overall_std).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=f'Cluster {cl} (n={len(df[df[cluster_col]==cl])})', color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_',' ').title() for m in metrics])
    ax.set_title(f'Cluster Profiles (Z-scores) for {cluster_col}', pad=20)
    ax.grid(True)
    ax.axhline(0,color='gray',alpha=0.5)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
    plt.tight_layout()
    plt.savefig(out_dir/f'{cluster_col}_profiles_radar.png', dpi=300)
    plt.close()


def main():
    df, coeffs = load_data('AAAAsindy_analysis_with_metrics.csv')
    out_dir = Path('/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots/sindy_clusters')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dendrogram
    plot_dendrogram(df, coeffs, out_dir)

    # Assign clusters
    df['cluster_k'] = cluster_features(df, coeffs, n_clusters=3, method='kmeans')
    df['cluster_h'] = cluster_features(df, coeffs, n_clusters=3, method='agglomerative')

    # Summaries
    summarize_clusters(df, out_dir)
    plot_cluster_feature_correlations(df, out_dir, 'cluster_k')
    plot_cluster_feature_correlations(df, out_dir, 'cluster_h')
    plot_cluster_profiles_radar(df, out_dir, 'cluster_k')
    plot_cluster_profiles_radar(df, out_dir, 'cluster_h')

    # Save assignments
    df[['participant_id'] + coeffs + ['cluster_k','cluster_h']].to_csv(out_dir/'cluster_assignments.csv', index=False)

if __name__=='__main__':
    main()
