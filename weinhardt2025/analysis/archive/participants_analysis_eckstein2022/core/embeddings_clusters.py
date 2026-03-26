import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Paths
csv_path = '/Users/martynaplomecka/closedloop_rl/final_df_sindy_analysis_with_metrics.csv'
output_dir = '/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis/plots/embeddings_clusters'
# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(csv_path)

# Prepare embeddings
embedding_cols = [f'embedding_{i}' for i in range(32)]
X = df[embedding_cols].values

# t-SNE reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(X_tsne)

# Add tsne and cluster to df
df['tsne_1'] = X_tsne[:, 0]
df['tsne_2'] = X_tsne[:, 1]
df['cluster'] = clusters

# Age stats
age_stats = df.groupby('cluster')['Age'].agg(['mean', 'var']).rename(columns={'mean': 'age_mean', 'var': 'age_variance'})
print(age_stats)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['tsne_1'], df['tsne_2'], c=df['Age'], cmap='viridis', alpha=0.8)
plt.colorbar(scatter, label='Age')

for i in range(6):
    cx = df.loc[df['cluster'] == i, 'tsne_1'].mean()
    cy = df.loc[df['cluster'] == i, 'tsne_2'].mean()
    plt.text(cx, cy, str(i), fontsize=12, fontweight='bold', ha='center', va='center', color='white',
             bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

plt.title('t-SNE of Participant Embeddings with KMeans Clusters (6)\nand Age-Colored Scatter')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.tight_layout()

# Save plot
output_path = os.path.join(output_dir, 'embeddings_clusters.png')
plt.savefig(output_path)
print(f'Plot saved to {output_path}')
plt.show()