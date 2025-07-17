import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Paths
csv_path = '/Users/martynaplomecka/closedloop_rl/dezfouli_final_df_sindy_analysis_with_metrics.csv'
output_dir = '/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_dezfouli/plots/embeddings_clusters'

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# Remove rows with missing diagnosis
df = df.dropna(subset=['Diagnosis'])

embedding_cols = [f'embedding_{i}' for i in range(32)]
X = df[embedding_cols].values

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# KMeans clustering with 3 clusters (for 3 diagnosis categories)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_tsne)

df['tsne_1'] = X_tsne[:, 0]
df['tsne_2'] = X_tsne[:, 1]
df['cluster'] = clusters

diagnosis_stats = df.groupby(['cluster', 'Diagnosis']).size().unstack(fill_value=0)
print("Diagnosis distribution by cluster:")
print(diagnosis_stats)
print()

diagnosis_percentages = df.groupby(['cluster', 'Diagnosis']).size().unstack(fill_value=0)
diagnosis_percentages = diagnosis_percentages.div(diagnosis_percentages.sum(axis=1), axis=0) * 100
print("Diagnosis percentages by cluster:")
print(diagnosis_percentages.round(1))
print()

diagnosis_colors = {'Depression': '#FF6B6B', 'Bipolar': '#4ECDC4', 'Healthy': '#45B7D1'}
diagnosis_mapping = {'Depression': 0, 'Bipolar': 1, 'Healthy': 2}


df['color'] = df['Diagnosis'].map(diagnosis_colors)

plt.figure(figsize=(12, 8))

# scatter plot colored by diagnosis
for diagnosis, color in diagnosis_colors.items():
    mask = df['Diagnosis'] == diagnosis
    plt.scatter(df.loc[mask, 'tsne_1'], df.loc[mask, 'tsne_2'], 
               c=color, label=diagnosis, alpha=0.7, s=50)

# Add cluster center labels
for i in range(3):
    cx = df.loc[df['cluster'] == i, 'tsne_1'].mean()
    cy = df.loc[df['cluster'] == i, 'tsne_2'].mean()
    plt.text(cx, cy, f'C{i}', fontsize=14, fontweight='bold', ha='center', va='center', 
             color='white', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

plt.title('t-SNE of Participant Embeddings with KMeans Clusters (3)\nColored by Diagnosis', fontsize=14)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Diagnosis')
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = os.path.join(output_dir, 'embeddings_clusters_diagnosis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Plot saved to {output_path}')
plt.show()

plt.figure(figsize=(12, 8))

# Color by cluster instead of diagnosis
cluster_colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green for clusters 0, 1, 2
for i in range(3):
    mask = df['cluster'] == i
    plt.scatter(df.loc[mask, 'tsne_1'], df.loc[mask, 'tsne_2'], 
               c=cluster_colors[i], label=f'Cluster {i}', alpha=0.7, s=50)

# Add cluster center labels
for i in range(3):
    cx = df.loc[df['cluster'] == i, 'tsne_1'].mean()
    cy = df.loc[df['cluster'] == i, 'tsne_2'].mean()
    plt.text(cx, cy, f'C{i}', fontsize=14, fontweight='bold', ha='center', va='center', 
             color='white', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

plt.title('t-SNE of Participant Embeddings Colored by KMeans Clusters', fontsize=14)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save cluster plot
cluster_output_path = os.path.join(output_dir, 'embeddings_clusters_only.png')
plt.savefig(cluster_output_path, dpi=300, bbox_inches='tight')
print(f'Cluster plot saved to {cluster_output_path}')
plt.show()

# Save the updated dataframe with cluster assignments
df_output_path = os.path.join(output_dir, 'embeddings_with_clusters.csv')
df.to_csv(df_output_path, index=False)
print(f'Data with cluster assignments saved to {df_output_path}')