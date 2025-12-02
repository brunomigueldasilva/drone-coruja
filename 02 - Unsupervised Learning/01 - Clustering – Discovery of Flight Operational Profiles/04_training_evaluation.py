#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
MODEL TRAINING AND EVALUATION - CLUSTERING ALGORITHMS
==============================================================================

Purpose: Train and evaluate multiple clustering algorithms on preprocessed flight telemetry data.

This script:
1. Loads preprocessed and scaled feature matrix
2. Trains K-Means clustering with optimal k
3. Trains Agglomerative Clustering with Ward linkage
4. Tunes and trains DBSCAN with grid search
5. Generates dendrogram visualization
6. Computes internal validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
7. Creates comparison table and detailed analysis notes

Expected that BEST_K is determined from elbow method (03_elbow_method.py).

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

# Configuration Constants
SEED = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore')


class Config:
    """Training and evaluation configuration parameters."""
    # Directories
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'
    DATA_PROCESSED_DIR = OUTPUT_DIR / 'data_processed'
    MODELS_DIR = OUTPUT_DIR / 'models'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'

    # Input file
    PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / 'processed_X.npy'

    # Clustering configuration
    BEST_K = 3  # ⚠️ IMPORTANT: Update this based on elbow method results

    # K-Means parameters
    KMEANS_N_INIT = 20
    KMEANS_MAX_ITER = 500

    # Agglomerative parameters
    AGGLOMERATIVE_LINKAGE = 'ward'

    # DBSCAN parameter grid for tuning
    DBSCAN_EPS_RANGE = [0.5, 1.0, 1.5, 2.0]
    DBSCAN_MIN_SAMPLES_RANGE = [5, 10, 15]

    # Visualization settings
    DPI = 300
    FIGSIZE_LARGE = (16, 10)
    FIGSIZE_MEDIUM = (14, 8)


# Visualization configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = Config.FIGSIZE_MEDIUM


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """
    Print formatted section header.

    Args:
        title: Section title text
        char: Character to use for separator line
    """
    print("\n" + char * 120)
    print(title)
    print(char * 120)


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_processed_data(path: Path, best_k: int) -> np.ndarray:
    """
    Load the preprocessed and scaled feature matrix.

    Args:
        path: Path to the .npy file
        best_k: Best k value from elbow method

    Returns:
        Numpy array with scaled features

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {
                path.absolute()}\n" "Please run '02_preprocessing.py' first.")

    X = np.load(path)
    print("✓ Data loaded successfully!")
    print(f"  Shape: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Data type: {X.dtype}")
    print(f"  Value range: [{X.min():.4f}, {X.max():.4f}]")

    print(f"\n⚠️  REMINDER: Current BEST_K = {best_k}")
    print("   Update BEST_K in Config class based on elbow method results")

    return X


# ==============================================================================
# SECTION 4: MODEL TRAINING FUNCTIONS
# ==============================================================================


def train_kmeans(X: np.ndarray,
                 k: int,
                 n_init: int,
                 max_iter: int,
                 random_state: int,
                 models_dir: Path) -> Tuple[KMeans,
                                            np.ndarray,
                                            Dict[str,
                                                 Any]]:
    """
    Train K-Means clustering model.

    Args:
        X: Feature matrix
        k: Number of clusters
        n_init: Number of initializations
        max_iter: Maximum iterations
        random_state: Random seed
        models_dir: Directory to save model

    Returns:
        Tuple of (model, labels, metrics_dict)
    """
    print_section("TRAINING K-MEANS", "-")
    print("Parameters:")
    print(f"  • n_clusters: {k}")
    print(f"  • n_init: {n_init}")
    print(f"  • max_iter: {max_iter}")
    print(f"  • random_state: {random_state}")

    print("\nFitting K-Means...")
    kmeans = KMeans(
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state)
    labels = kmeans.fit_predict(X)

    print("✓ K-Means trained successfully!")
    print(f"  Iterations: {kmeans.n_iter_}")
    print(f"  Inertia (WCSS): {kmeans.inertia_:.2f}")

    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster distribution:")
    for cluster_id, count in zip(unique, counts):
        pct = (count / len(labels)) * 100
        print(f"  • Cluster {cluster_id}: {count} samples ({pct:.2f}%)")

    print("\nComputing evaluation metrics...")
    metrics = {
        'silhouette': silhouette_score(X, labels, random_state=random_state),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'n_clusters': k,
        'n_noise': 0
    }

    print(f"  • Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"  • Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
    print(f"  • Calinski-Harabasz Index: {metrics['calinski_harabasz']:.2f}")

    model_path = models_dir / f"kmeans_k{k}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"\n✓ Model saved to: {model_path}")

    return kmeans, labels, metrics


def train_agglomerative(X: np.ndarray,
                        k: int,
                        linkage_method: str,
                        models_dir: Path) -> Tuple[AgglomerativeClustering,
                                                   np.ndarray,
                                                   Dict[str,
                                                        Any]]:
    """
    Train Agglomerative Clustering model.

    Args:
        X: Feature matrix
        k: Number of clusters
        linkage_method: Linkage criterion ('ward', 'complete', 'average', 'single')
        models_dir: Directory to save model

    Returns:
        Tuple of (model, labels, metrics_dict)
    """
    print_section("TRAINING AGGLOMERATIVE CLUSTERING", "-")
    print("Parameters:")
    print(f"  • n_clusters: {k}")
    print(f"  • linkage: {linkage_method}")
    print("  • metric: euclidean (required for Ward linkage)")

    print("\nFitting Agglomerative Clustering...")
    agglomerative = AgglomerativeClustering(
        n_clusters=k, linkage=linkage_method)
    labels = agglomerative.fit_predict(X)

    print("✓ Agglomerative Clustering trained successfully!")
    print(f"  Number of leaves: {agglomerative.n_leaves_}")
    print(
        f"  Number of connected components: {
            agglomerative.n_connected_components_}")

    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster distribution:")
    for cluster_id, count in zip(unique, counts):
        pct = (count / len(labels)) * 100
        print(f"  • Cluster {cluster_id}: {count} samples ({pct:.2f}%)")

    print("\nComputing evaluation metrics...")
    metrics = {
        'silhouette': silhouette_score(X, labels, random_state=SEED),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'n_clusters': k,
        'n_noise': 0
    }

    print(f"  • Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"  • Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
    print(f"  • Calinski-Harabasz Index: {metrics['calinski_harabasz']:.2f}")

    model_path = models_dir / f"agglomerative_k{k}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(agglomerative, f)
    print(f"\n✓ Model saved to: {model_path}")

    return agglomerative, labels, metrics


def tune_and_train_dbscan(X: np.ndarray,
                          eps_range: List[float],
                          min_samples_range: List[int],
                          models_dir: Path) -> Tuple[DBSCAN,
                                                     np.ndarray,
                                                     Dict[str,
                                                          Any],
                                                     Dict[str,
                                                          Any]]:
    """
    Tune DBSCAN hyperparameters and train the best model.

    Args:
        X: Feature matrix
        eps_range: List of eps values to try
        min_samples_range: List of min_samples values to try
        models_dir: Directory to save model

    Returns:
        Tuple of (best_model, labels, metrics_dict, best_params)
    """
    print_section("TUNING AND TRAINING DBSCAN", "-")
    print("DBSCAN is a density-based clustering algorithm:")
    print("  • eps: Defines neighborhood radius (larger = more points in neighborhood)")
    print("  • min_samples: Minimum points needed to form a dense region (core point)")
    print("  • Points that don't meet criteria are labeled as noise (label = -1)")

    print("\nParameter grid:")
    print(f"  • eps: {eps_range}")
    print(f"  • min_samples: {min_samples_range}")

    best_score = -1
    best_params = None
    best_model = None
    best_labels = None

    print("\nTuning DBSCAN (grid search)...")
    print(
        f"{
            'eps':<8} {
            'min_samples':<15} {
                'n_clusters':<15} {
                    'n_noise':<12} {
                        'Silhouette':<12}")
    print("-" * 120)

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters >= 2 and n_noise < len(labels):
                mask = labels != -1
                if mask.sum() > 0:
                    try:
                        score = silhouette_score(X[mask], labels[mask], random_state=SEED)
                    except BaseException:
                        score = -1
                else:
                    score = -1
            else:
                score = -1

            print(
                f"{
                    eps:<8} {
                    min_samples:<15} {
                    n_clusters:<15} {
                    n_noise:<12} {
                        score:<12.4f}")

            if score > best_score and n_clusters >= 2:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
                best_model = dbscan
                best_labels = labels

    if best_model is None:
        print("\n⚠ WARNING: No valid DBSCAN configuration found. Using default eps=1.0, min_samples=5")
        best_params = {'eps': 1.0, 'min_samples': 5}
        best_model = DBSCAN(eps=1.0, min_samples=5)
        best_labels = best_model.fit_predict(X)

    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    n_noise = list(best_labels).count(-1)

    print("\n✓ Best DBSCAN parameters:")
    print(f"  • eps: {best_params['eps']}")
    print(f"  • min_samples: {best_params['min_samples']}")
    print(f"  • n_clusters: {n_clusters}")
    print(
        f"  • n_noise: {n_noise} ({(n_noise / len(best_labels) * 100):.2f}%)")

    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': None,
        'davies_bouldin': None,
        'calinski_harabasz': None}

    if n_clusters >= 2 and n_noise < len(best_labels):
        mask = best_labels != -1
        try:
            metrics['silhouette'] = silhouette_score(
                X[mask], best_labels[mask], random_state=SEED)
            metrics['davies_bouldin'] = davies_bouldin_score(
                X[mask], best_labels[mask])
            metrics['calinski_harabasz'] = calinski_harabasz_score(
                X[mask], best_labels[mask])
            print("\nEvaluation metrics (excluding noise):")
            print(f"  • Silhouette Score: {metrics['silhouette']:.4f}")
            print(f"  • Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
            print(
                f"  • Calinski-Harabasz Index: {metrics['calinski_harabasz']:.2f}")
        except BaseException:
            print("\n⚠ Could not compute all metrics")

    model_path = models_dir / 'dbscan.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n✓ Model saved to: {model_path}")

    return best_model, best_labels, metrics, best_params


# ==============================================================================
# SECTION 5: VISUALIZATION FUNCTIONS
# ==============================================================================


def plot_dendrogram(
        X: np.ndarray,
        best_k: int,
        png_path: Path,
        pdf_path: Path) -> None:
    """
    Compute linkage and plot dendrogram for hierarchical clustering.

    Args:
        X: Feature matrix
        best_k: Target number of clusters (for color threshold)
        png_path: Path to save PNG version
        pdf_path: Path to save PDF version
    """
    print_section("CREATING DENDROGRAM", "-")
    print("Computing hierarchical linkage matrix...")
    print("  • Method: Ward")
    print("  • Metric: Euclidean")

    linkage_matrix = linkage(X, method='ward')
    print("✓ Linkage computed successfully!")
    print(f"  Shape: {linkage_matrix.shape}")

    print("\nPlotting dendrogram...")
    fig, ax = plt.subplots(figsize=Config.FIGSIZE_LARGE)

    truncate_mode = 'lastp' if X.shape[0] > 100 else None
    p_value = 50 if X.shape[0] > 100 else None

    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode=truncate_mode,
        p=p_value,
        leaf_font_size=10,
        color_threshold=None)

    if X.shape[0] >= best_k:
        threshold_height = linkage_matrix[-(best_k - 1), 2]
        ax.axhline(y=threshold_height, color='red', linestyle='--',
                   linewidth=2, label=f'Cut for k={best_k} clusters')
        ax.legend(fontsize=12)

    ax.set_xlabel(
        'Sample Index or Cluster Size',
        fontsize=13,
        fontweight='bold')
    ax.set_ylabel('Distance (Ward Linkage)', fontsize=13, fontweight='bold')
    ax.set_title('Hierarchical Clustering Dendrogram',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')

    note_text = (f"Dendrogram shows hierarchical cluster structure.\n"
                 f"Height indicates dissimilarity between merged clusters.\n"
                 f"Red dashed line suggests cut for {best_k} clusters.")
    ax.text(
        0.02,
        0.98,
        note_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='wheat',
            alpha=0.6))

    plt.tight_layout()
    plt.savefig(png_path, dpi=Config.DPI, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Dendrogram saved to: {png_path}")
    print(f"✓ Dendrogram saved to: {pdf_path}")
    plt.close()


# ==============================================================================
# SECTION 6: EXPORT FUNCTIONS
# ==============================================================================


def create_comparison_table(kmeans_metrics: Dict[str,
                                                 Any],
                            agglomerative_metrics: Dict[str,
                                                        Any],
                            dbscan_metrics: Dict[str,
                                                 Any],
                            dbscan_params: Dict[str,
                            Any],
                            output_dir: Path,
                            best_k: int) -> None:
    """
    Create comparison table of clustering algorithms.

    Args:
        kmeans_metrics: Metrics from K-Means
        agglomerative_metrics: Metrics from Agglomerative
        dbscan_metrics: Metrics from DBSCAN
        dbscan_params: DBSCAN parameters
        output_dir: Directory to save outputs
        best_k: Best k value
    """
    print_section("CREATING MODEL COMPARISON TABLE", "-")

    comparison_data = []

    comparison_data.append({
        'Model': 'K-Means',
        'n_clusters': kmeans_metrics['n_clusters'],
        'Silhouette': f"{kmeans_metrics['silhouette']:.4f}",
        'Davies-Bouldin': f"{kmeans_metrics['davies_bouldin']:.4f}",
        'Calinski-Harabasz': f"{kmeans_metrics['calinski_harabasz']:.2f}",
        'n_noise': kmeans_metrics['n_noise']
    })

    comparison_data.append({
        'Model': 'Agglomerative',
        'n_clusters': agglomerative_metrics['n_clusters'],
        'Silhouette': f"{agglomerative_metrics['silhouette']:.4f}",
        'Davies-Bouldin': f"{agglomerative_metrics['davies_bouldin']:.4f}",
        'Calinski-Harabasz': f"{agglomerative_metrics['calinski_harabasz']:.2f}",
        'n_noise': agglomerative_metrics['n_noise']
    })

    sil_str = f"{
        dbscan_metrics['silhouette']:.4f}" if dbscan_metrics['silhouette'] is not None else "N/A"
    dbi_str = f"{
        dbscan_metrics['davies_bouldin']:.4f}" if dbscan_metrics['davies_bouldin'] is not None else "N/A"
    chi_str = (f"{dbscan_metrics['calinski_harabasz']:.2f}" if dbscan_metrics['calinski_harabasz'] is not None
               else "N/A")

    comparison_data.append({
        'Model': f"DBSCAN (eps={dbscan_params['eps']}, min={dbscan_params['min_samples']})",
        'n_clusters': dbscan_metrics['n_clusters'],
        'Silhouette': sil_str,
        'Davies-Bouldin': dbi_str,
        'Calinski-Harabasz': chi_str,
        'n_noise': dbscan_metrics['n_noise']
    })

    comparison_df = pd.DataFrame(comparison_data)

    csv_path = output_dir / 'model_comparison.csv'
    comparison_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✓ Saved CSV to: {csv_path}")

    md_path = output_dir / 'model_comparison.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Clustering Model Comparison\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n## Metric Interpretations\n\n")
        f.write("- **Silhouette Score**: [-1, 1], higher is better\n")
        f.write("- **Davies-Bouldin Index**: [0, ∞), lower is better\n")
        f.write("- **Calinski-Harabasz Index**: [0, ∞), higher is better\n")

    print(f"✓ Saved Markdown to: {md_path}")


def create_train_eval_notes(kmeans_metrics: Dict[str,
                                                 Any],
                            agglomerative_metrics: Dict[str,
                                                        Any],
                            dbscan_metrics: Dict[str,
                                                 Any],
                            dbscan_params: Dict[str,
                            Any],
                            best_k: int,
                            output_dir: Path) -> None:
    """
    Generate detailed training and evaluation notes.

    Args:
        kmeans_metrics: Metrics from K-Means
        agglomerative_metrics: Metrics from Agglomerative
        dbscan_metrics: Metrics from DBSCAN
        dbscan_params: DBSCAN parameters
        best_k: Best k value
        output_dir: Directory to save notes
    """
    print_section("GENERATING TRAINING AND EVALUATION NOTES", "-")

    notes_path = output_dir / 'train_eval_notes.md'

    with open(notes_path, 'w', encoding='utf-8') as f:
        f.write("# Model Training and Evaluation Notes\n\n")

        f.write("## 1. Algorithms Overview\n\n")

        f.write("### K-Means\n")
        f.write("**Strengths:**\n")
        f.write("- Fast and scalable to large datasets\n")
        f.write("- Simple to understand and implement\n")
        f.write("- Works well with spherical, similarly-sized clusters\n\n")
        f.write("**Weaknesses:**\n")
        f.write("- Must specify k in advance\n")
        f.write("- Sensitive to initial centroid placement\n")
        f.write("- Sensitive to outliers\n")
        f.write("- Assumes clusters are spherical and similar size\n\n")

        f.write("### Agglomerative Clustering\n")
        f.write("**Strengths:**\n")
        f.write("- Hierarchical structure provides insights\n")
        f.write("- Deterministic (no random initialization)\n")
        f.write("- Can capture non-spherical clusters\n\n")
        f.write("**Weaknesses:**\n")
        f.write("- Computationally expensive O(n²) or O(n³)\n")
        f.write("- Not scalable to very large datasets\n")
        f.write("- Still requires specifying k\n\n")

        f.write("### DBSCAN\n")
        f.write("**Strengths:**\n")
        f.write("- Robust to outliers (labels them as noise)\n")
        f.write("- Can find arbitrarily shaped clusters\n")
        f.write("- No need to specify k in advance\n\n")
        f.write("**Weaknesses:**\n")
        f.write("- Sensitive to eps and min_samples parameters\n")
        f.write("- Struggles with varying density clusters\n")
        f.write("- Can label many points as noise if parameters not tuned well\n\n")

        f.write("## 2. Evaluation Metrics Explained\n\n")

        f.write("### Silhouette Score\n")
        f.write("**Range:** [-1, 1]\n")
        f.write("**Interpretation:**\n")
        f.write(
            "- **+1**: Perfect clustering (points very close to their own cluster, far from others)\n")
        f.write("- **0**: Clusters overlap or point is on boundary\n")
        f.write("- **-1**: Point assigned to wrong cluster\n\n")

        f.write("### Davies-Bouldin Index\n")
        f.write("**Range:** [0, ∞)\n")
        f.write("**Interpretation:**\n")
        f.write("- **Lower is better**\n")
        f.write("- Measures average similarity between clusters\n")
        f.write("- 0 = perfect separation\n\n")

        f.write("### Calinski-Harabasz Index\n")
        f.write("**Range:** [0, ∞)\n")
        f.write("**Interpretation:**\n")
        f.write("- **Higher is better**\n")
        f.write("- Ratio of between-cluster to within-cluster dispersion\n")
        f.write("- Higher values indicate better-defined clusters\n\n")

        f.write("## 3. Results Summary\n\n")

        f.write(f"### K-Means (k={best_k})\n")
        f.write(f"- Silhouette: {kmeans_metrics['silhouette']:.4f}\n")
        f.write(f"- Davies-Bouldin: {kmeans_metrics['davies_bouldin']:.4f}\n")
        f.write(
            f"- Calinski-Harabasz: {kmeans_metrics['calinski_harabasz']:.2f}\n\n")

        f.write(f"### Agglomerative (k={best_k})\n")
        f.write(f"- Silhouette: {agglomerative_metrics['silhouette']:.4f}\n")
        f.write(
            f"- Davies-Bouldin: {agglomerative_metrics['davies_bouldin']:.4f}\n")
        f.write(
            f"- Calinski-Harabasz: {agglomerative_metrics['calinski_harabasz']:.2f}\n\n")

        f.write("### DBSCAN\n")
        f.write(
            f"- Parameters: eps={
                dbscan_params['eps']}, min_samples={
                dbscan_params['min_samples']}\n")
        f.write(f"- n_clusters: {dbscan_metrics['n_clusters']}\n")
        f.write(f"- n_noise: {dbscan_metrics['n_noise']}\n")
        if dbscan_metrics['silhouette'] is not None:
            f.write(f"- Silhouette: {dbscan_metrics['silhouette']:.4f}\n")
            f.write(
                f"- Davies-Bouldin: {dbscan_metrics['davies_bouldin']:.4f}\n")
            f.write(
                f"- Calinski-Harabasz: {dbscan_metrics['calinski_harabasz']:.2f}\n\n")

        f.write("## 4. Next Steps\n\n")
        f.write("1. **Visualize clusters** using PCA or t-SNE\n")
        f.write("2. **Analyze cluster profiles** by examining feature statistics\n")
        f.write("3. **Validate with domain experts**\n")
        f.write("4. **Test stability** with multiple runs\n")
        f.write("5. **Consider external validation** if labels available\n\n")

        f.write("---\n\n")
        f.write("**End of Training and Evaluation Notes**\n")

    print(f"✓ Training and evaluation notes saved to: {notes_path}")


# ==============================================================================
# SECTION 7: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the entire training and evaluation workflow."""
    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    Config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("MODEL TRAINING AND EVALUATION - CLUSTERING ALGORITHMS")
    print("=" * 120)
    print(f"\nRandom seed: {SEED}")
    print(f"Best k: {Config.BEST_K}")

    try:
        # 1. Load processed data
        print_section("STEP 1: LOAD PROCESSED DATA")
        X = load_processed_data(Config.PROCESSED_DATA_PATH, Config.BEST_K)

        # 2. Train K-Means
        print_section("STEP 2: TRAIN K-MEANS")
        kmeans_model, kmeans_labels, kmeans_metrics = train_kmeans(
            X, Config.BEST_K, Config.KMEANS_N_INIT, Config.KMEANS_MAX_ITER, SEED, Config.MODELS_DIR)

        # 3. Train Agglomerative Clustering
        print_section("STEP 3: TRAIN AGGLOMERATIVE CLUSTERING")
        agglomerative_model, agglomerative_labels, agglomerative_metrics = train_agglomerative(
            X, Config.BEST_K, Config.AGGLOMERATIVE_LINKAGE, Config.MODELS_DIR)

        # 4. Create dendrogram
        print_section("STEP 4: CREATE DENDROGRAM")
        plot_dendrogram(X, Config.BEST_K,
                        Config.GRAPHICS_DIR / "dendrogram.png",
                        Config.GRAPHICS_DIR / "dendrogram.pdf")

        # 5. Tune and train DBSCAN
        print_section("STEP 5: TUNE AND TRAIN DBSCAN")
        dbscan_model, dbscan_labels, dbscan_metrics, dbscan_params = tune_and_train_dbscan(
            X, Config.DBSCAN_EPS_RANGE, Config.DBSCAN_MIN_SAMPLES_RANGE, Config.MODELS_DIR
        )

        # 6. Create comparison table
        print_section("STEP 6: CREATE COMPARISON TABLE")
        create_comparison_table(
            kmeans_metrics,
            agglomerative_metrics,
            dbscan_metrics,
            dbscan_params,
            Config.DATA_PROCESSED_DIR,
            Config.BEST_K)

        # 7. Create training and evaluation notes
        print_section("STEP 7: CREATE TRAINING NOTES")
        create_train_eval_notes(
            kmeans_metrics,
            agglomerative_metrics,
            dbscan_metrics,
            dbscan_params,
            Config.BEST_K,
            Config.OUTPUT_DIR)

        # Final summary
        print_section("MODEL TRAINING AND EVALUATION COMPLETED")
        print("\nOUTPUTS GENERATED:")
        print("\nMODELS:")
        print(f"  • {Config.MODELS_DIR / f'kmeans_k{Config.BEST_K}.pkl'}")
        print(
            f"  • {
                Config.MODELS_DIR /
                f'agglomerative_k{
                    Config.BEST_K}.pkl'}")
        print(f"  • {Config.MODELS_DIR / 'dbscan.pkl'}")
        print("\nGRAPHICS:")
        print(f"  • {Config.GRAPHICS_DIR / 'dendrogram.png'}")
        print(f"  • {Config.GRAPHICS_DIR / 'dendrogram.pdf'}")
        print("\nDATA PROCESSED:")
        print(f"  • {Config.DATA_PROCESSED_DIR / 'model_comparison.csv'}")
        print(f"  • {Config.DATA_PROCESSED_DIR / 'model_comparison.md'}")
        print("\nNOTES:")
        print(f"  • {Config.OUTPUT_DIR / 'train_eval_notes.md'}")

        print("\n" + "=" * 120)
        print("✓ All models trained and evaluated successfully!")
        print("=" * 120)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
