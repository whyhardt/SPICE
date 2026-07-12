"""Behavioral clustering analysis.

Clusters participants by behavioral metrics, then tests whether
equation structure (SINDy term presence and coefficients) differs
significantly between behavioral clusters.

Usage:
    from weinhardt2026.analysis.analysis_behavioral_clustering import analysis_behavioral_clustering

    results = analysis_behavioral_clustering(
        spice_model=estimator,
        path_behavioral_metrics='results/behavioral_metrics_real.csv',
        n_clusters=3,
        output_dir='results',
    )
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from scipy.stats import kruskal, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

from spice import SpiceEstimator


def _extract_equation_features(estimator: SpiceEstimator):
    """Extract per-participant SINDy coefficient and presence matrices.

    Returns:
        coeff_df: DataFrame (n_participants, n_terms) with coefficient values
        presence_df: DataFrame (n_participants, n_terms) with binary presence
        term_names: list of term name strings
    """
    coefficients = estimator.get_sindy_coefficients(aggregate=True)
    candidate_terms = estimator.get_candidate_terms()
    modules = estimator.get_modules()

    rows_coeff = []
    rows_presence = []
    term_names = []

    # Build flat term name list across all modules
    for mod_name in modules:
        terms = candidate_terms[mod_name]
        for term in terms:
            term_names.append(f"{mod_name}:{term}")

    # coefficients shape per module: (P, X, C) — squeeze experiments, convert to numpy
    coeff_np = {}
    for mod_name in modules:
        c = coefficients[mod_name]
        if hasattr(c, 'detach'):
            c = c.detach().cpu()
        c = np.array(c).squeeze(1)  # (P, C)
        coeff_np[mod_name] = c

    n_participants = coeff_np[modules[0]].shape[0]

    for pid in range(n_participants):
        row_c = {}
        row_p = {}
        for mod_name in modules:
            terms = candidate_terms[mod_name]
            coefs = coeff_np[mod_name][pid]  # (C,)
            for i, term in enumerate(terms):
                col = f"{mod_name}:{term}"
                row_c[col] = float(coefs[i])
                row_p[col] = 1 if abs(coefs[i]) > 1e-8 else 0
        rows_coeff.append(row_c)
        rows_presence.append(row_p)

    coeff_df = pd.DataFrame(rows_coeff)
    presence_df = pd.DataFrame(rows_presence)

    return coeff_df, presence_df, term_names


def _cluster_participants(df_metrics, n_clusters=3, method='ward'):
    """Cluster participants on standardized behavioral metrics.

    Args:
        df_metrics: DataFrame with one row per participant, metric columns.
        n_clusters: Number of clusters.
        method: Linkage method for hierarchical clustering.

    Returns:
        labels: (n_participants,) cluster labels (1-indexed).
        linkage_matrix: Linkage matrix for dendrogram plotting.
        centroids: DataFrame of cluster centroids in original scale.
        nearest_to_centroid: dict mapping cluster label -> participant index
            closest to centroid in standardized space.
    """
    metric_cols = [c for c in df_metrics.columns if c != 'participant_id']
    X = df_metrics[metric_cols].values

    # Impute NaNs with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    dist = pdist(X_std, metric='euclidean')
    Z = linkage(dist, method=method)
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    # Centroids in standardized space
    centroids_std = np.zeros((n_clusters, X_std.shape[1]))
    for k in range(1, n_clusters + 1):
        centroids_std[k - 1] = X_std[labels == k].mean(axis=0)

    # Centroids in original scale
    centroids_orig = scaler.inverse_transform(centroids_std)
    centroids = pd.DataFrame(centroids_orig, columns=metric_cols,
                             index=[f'Cluster {k}' for k in range(1, n_clusters + 1)])

    # Nearest participant to centroid
    nearest = {}
    for k in range(1, n_clusters + 1):
        members = np.where(labels == k)[0]
        dists = np.linalg.norm(X_std[members] - centroids_std[k - 1], axis=1)
        nearest[k] = members[np.argmin(dists)]

    return labels, Z, centroids, nearest


def _test_equation_differences(labels, coeff_df, presence_df):
    """Test whether equation terms differ significantly between clusters.

    Uses Kruskal-Wallis for coefficient values and chi-square-like
    proportion tests for term presence.

    Returns:
        df_results: DataFrame with term, test statistic, p-value, and
            per-cluster means/presence rates.
    """
    unique_labels = np.sort(np.unique(labels))
    n_clusters = len(unique_labels)
    results = []

    for col in coeff_df.columns:
        row = {'term': col}

        # Coefficient differences (Kruskal-Wallis)
        groups = [coeff_df[col].values[labels == k] for k in unique_labels]
        non_empty = [g for g in groups if len(g) > 0 and not np.all(np.isnan(g))]
        if len(non_empty) >= 2:
            try:
                stat, p = kruskal(*non_empty)
                row['kw_stat'] = stat
                row['kw_p'] = p
            except ValueError:
                row['kw_stat'] = np.nan
                row['kw_p'] = np.nan
        else:
            row['kw_stat'] = np.nan
            row['kw_p'] = np.nan

        # Per-cluster stats
        for k in unique_labels:
            mask = labels == k
            row[f'mean_c{k}'] = coeff_df[col].values[mask].mean()
            row[f'presence_c{k}'] = presence_df[col].values[mask].mean()

        results.append(row)

    df_results = pd.DataFrame(results)

    # Multiple comparison correction (Bonferroni)
    n_tests = df_results['kw_p'].notna().sum()
    if n_tests > 0:
        df_results['kw_p_corrected'] = df_results['kw_p'] * n_tests
        df_results['kw_p_corrected'] = df_results['kw_p_corrected'].clip(upper=1.0)
    else:
        df_results['kw_p_corrected'] = np.nan

    return df_results


def analysis_behavioral_clustering(
    spice_model: SpiceEstimator,
    path_behavioral_metrics: str,
    n_clusters: int = 3,
    output_dir: str = 'results',
):
    """Run behavioral clustering analysis.

    1. Load per-participant behavioral metrics.
    2. Cluster participants by behavioral metrics.
    3. Extract equation features (coefficients + presence).
    4. Test alignment between behavioral clusters and equation structure.
    5. Save results and diagnostic plots.

    Args:
        spice_model: Fitted SpiceEstimator.
        path_behavioral_metrics: Path to CSV with per-participant metrics
            (from analysis_generative_behavior).
        n_clusters: Number of behavioral clusters.
        output_dir: Output directory.

    Returns:
        dict with keys: 'labels', 'linkage', 'centroids', 'nearest',
        'equation_tests', 'alignment_ari', 'df_metrics'.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load behavioral metrics
    df_metrics = pd.read_csv(path_behavioral_metrics)
    print(f"Loaded {len(df_metrics)} participants from {path_behavioral_metrics}")

    # Extract equation features
    coeff_df, presence_df, term_names = _extract_equation_features(spice_model)
    print(f"Extracted {len(term_names)} SINDy terms across {len(coeff_df)} participants")

    # Cluster on behavioral metrics
    labels, Z, centroids, nearest = _cluster_participants(df_metrics, n_clusters)
    print(f"\nCluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"Cluster centroids:\n{centroids.round(3)}")
    print(f"Nearest to centroid: {nearest}")

    # Test equation differences between clusters
    df_eq_tests = _test_equation_differences(labels, coeff_df, presence_df)
    sig = df_eq_tests[df_eq_tests['kw_p_corrected'] < 0.05]
    print(f"\nSignificant equation terms (p_corrected < 0.05): {len(sig)} / {len(df_eq_tests)}")
    if len(sig) > 0:
        print(sig[['term', 'kw_stat', 'kw_p', 'kw_p_corrected']].to_string(index=False))

    # Alignment: cluster on equation presence and compare with behavioral clusters
    from scipy.cluster.hierarchy import fcluster as fcluster_
    presence_vals = presence_df.values
    if presence_vals.shape[1] > 0 and np.any(presence_vals.std(axis=0) > 0):
        # Remove zero-variance columns
        keep = presence_vals.std(axis=0) > 0
        presence_for_cluster = presence_vals[:, keep]
        if presence_for_cluster.shape[1] > 0:
            dist_eq = pdist(presence_for_cluster, metric='jaccard')
            # Replace NaN distances (identical rows) with 0
            dist_eq = np.nan_to_num(dist_eq, nan=0.0)
            Z_eq = linkage(dist_eq, method='ward')
            labels_eq = fcluster_(Z_eq, t=n_clusters, criterion='maxclust')
            ari = adjusted_rand_score(labels, labels_eq)
        else:
            ari = np.nan
            labels_eq = np.ones(len(labels), dtype=int)
    else:
        ari = np.nan
        labels_eq = np.ones(len(labels), dtype=int)

    print(f"\nBehavior ↔ Equation structure alignment (ARI): {ari:.3f}")

    # Save results
    df_metrics_out = df_metrics.copy()
    df_metrics_out['behavioral_cluster'] = labels
    df_metrics_out.to_csv(os.path.join(output_dir, 'behavioral_clusters.csv'), index=False)

    df_eq_tests.to_csv(os.path.join(output_dir, 'equation_cluster_tests.csv'), index=False)

    # Save cluster assignments with equation data
    eq_out = coeff_df.copy()
    eq_out.insert(0, 'participant_id', df_metrics['participant_id'])
    eq_out.insert(1, 'behavioral_cluster', labels)
    eq_out.to_csv(os.path.join(output_dir, 'participant_equations_by_cluster.csv'), index=False)

    return {
        'labels': labels,
        'linkage': Z,
        'centroids': centroids,
        'nearest': nearest,
        'equation_tests': df_eq_tests,
        'alignment_ari': ari,
        'df_metrics': df_metrics_out,
        'coeff_df': coeff_df,
        'presence_df': presence_df,
    }
