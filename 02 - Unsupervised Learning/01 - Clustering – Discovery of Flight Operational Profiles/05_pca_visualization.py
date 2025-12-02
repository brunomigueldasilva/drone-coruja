#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
PCA VISUALIZATION - CLUSTER ANALYSIS IN 2D
==============================================================================

Purpose: Visualize clustering results in 2D space using Principal Component Analysis.

This script:
1. Loads preprocessed feature matrix and model comparison results
2. Selects best clustering model based on evaluation metrics
3. Applies PCA to reduce dimensionality to 2D
4. Generates scatter plots showing cluster assignments in principal component space
5. Creates visualizations for K-Means and Agglomerative clustering
6. Exports explained variance ratios and analysis notes

PCA projects high-dimensional data onto 2 principal components that capture
maximum variance, enabling visual assessment of cluster quality and separation.

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# Configuration Constants
SEED = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore')


class Config:
    """PCA visualization configuration parameters."""
    # Directories
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'
    DATA_PROCESSED_DIR = OUTPUT_DIR / 'data_processed'
    MODELS_DIR = OUTPUT_DIR / 'models'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'

    # Input files
    PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / 'processed_X.npy'
    MODEL_COMPARISON_PATH = DATA_PROCESSED_DIR / 'model_comparison.csv'

    # PCA configuration
    N_COMPONENTS = 2

    # Visualization settings
    DPI = 300
    FIGSIZE = (14, 10)
    CMAP = 'tab10'


# Visualization configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = Config.FIGSIZE


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
# SECTION 3: DATA LOADING AND MODEL SELECTION
# ==============================================================================


def load_processed_data(path: Path) -> np.ndarray:
    """
    Load the preprocessed and scaled feature matrix.

    Args:
        path: Path to the .npy file

    Returns:
        Numpy array with scaled features

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {
                path.absolute()}")

    X = np.load(path)
    print("✓ Data loaded successfully!")
    print(f"  Shape: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Data type: {X.dtype}")

    return X


def select_best_model(comparison_path: Path,
                      models_dir: Path) -> Tuple[str, int]:
    """
    Select the best clustering model based on evaluation metrics.

    Args:
        comparison_path: Path to model comparison CSV
        models_dir: Directory containing model files

    Returns:
        Tuple of (best_model_name, best_k)
    """
    print_section("SELECTING BEST MODEL", "-")
    print(f"Reading model comparison from: {comparison_path}")

    if not comparison_path.exists():
        print("\n⚠ WARNING: Model comparison not found, defaulting to K-Means k=3")
        return 'K-Means', 3

    df = pd.read_csv(comparison_path)
    print("\n✓ Model comparison loaded!")
    print("\nAvailable models:")
    print(df[['Model', 'Silhouette', 'Davies-Bouldin',
          'n_clusters']].to_string(index=False))

    df_valid = df[(df['Silhouette'] != 'N/A') &
                  (~df['Model'].str.contains('DBSCAN'))].copy()

    if df_valid.empty:
        kmeans_files = list(models_dir.glob("kmeans_k*.pkl"))
        if kmeans_files:
            k = int(kmeans_files[0].stem.split('_k')[1])
            print(f"\n✓ Using {kmeans_files[0].name}, k={k}")
            return 'K-Means', k
        return 'K-Means', 3

    df_valid['Silhouette'] = pd.to_numeric(df_valid['Silhouette'])
    df_valid['Davies-Bouldin'] = pd.to_numeric(df_valid['Davies-Bouldin'])

    df_sorted = df_valid.sort_values(
        ['Silhouette', 'Davies-Bouldin'], ascending=[False, True])

    best_row = df_sorted.iloc[0]
    best_model = best_row['Model']
    best_k = int(best_row['n_clusters'])

    print(f"\n✓ Best model selected: {best_model}")
    print(f"  • Silhouette Score: {best_row['Silhouette']:.4f}")
    print(f"  • Davies-Bouldin Index: {best_row['Davies-Bouldin']:.4f}")
    print(f"  • Number of clusters: {best_k}")

    return best_model, best_k


# ==============================================================================
# SECTION 4: PCA FITTING
# ==============================================================================


def fit_pca(X: np.ndarray, n_components: int,
            output_dir: Path) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    Fit PCA to reduce dimensionality for visualization.

    Args:
        X: Feature matrix
        n_components: Number of components to keep
        output_dir: Directory to save variance explained

    Returns:
        Tuple of (pca_model, X_pca, explained_variance_ratio)
    """
    print_section("FITTING PCA", "-")
    print("Parameters:")
    print(f"  • n_components: {n_components}")
    print(f"  • Original dimensions: {X.shape[1]}")

    print("\nFitting PCA...")
    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca = pca.fit_transform(X)

    print("✓ PCA fitted successfully!")
    print(f"  Transformed shape: {X_pca.shape}")

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print("\nExplained variance by component:")
    for i, (var, cum_var) in enumerate(
            zip(explained_variance_ratio, cumulative_variance), 1):
        print(
            f"  • PC{i}: {
                var *
                100:.2f}% (cumulative: {
                cum_var *
                100:.2f}%)")

    variance_df = pd.DataFrame({
        'Component': [f'PC{i + 1}' for i in range(n_components)],
        'Explained_Variance_Ratio': explained_variance_ratio,
        'Cumulative_Variance_Ratio': cumulative_variance
    })

    variance_path = output_dir / 'pca_variance_explained.csv'
    variance_df.to_csv(variance_path, index=False, encoding='utf-8')
    print(f"\n✓ Variance explained saved to: {variance_path}")

    return pca, X_pca, explained_variance_ratio


# ==============================================================================
# SECTION 5: CLUSTER LABEL LOADING
# ==============================================================================


def load_cluster_labels(
        model_name: str,
        best_k: int,
        models_dir: Path,
        X: np.ndarray) -> np.ndarray:
    """
    Load cluster labels from saved model.

    Args:
        model_name: Name of the clustering model
        best_k: Number of clusters
        models_dir: Directory containing models
        X: Feature matrix

    Returns:
        Cluster labels array
    """
    print_section("LOADING CLUSTER LABELS", "-")

    if model_name == 'K-Means':
        model_path = models_dir / f"kmeans_k{best_k}.pkl"
    elif model_name == 'Agglomerative':
        model_path = models_dir / f"agglomerative_k{best_k}.pkl"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if model_name == 'K-Means':
        labels = model.predict(X)
    else:
        labels = model.fit_predict(X)

    print(f"✓ Loaded labels from: {model_path.name}")
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster distribution:")
    for cluster_id, count in zip(unique, counts):
        print(
            f"  • Cluster {cluster_id}: {count} samples ({
                count /
                len(labels) *
                100:.2f}%)")

    return labels


# ==============================================================================
# SECTION 6: VISUALIZATION FUNCTIONS
# ==============================================================================


def plot_pca_clusters(
        X_pca: np.ndarray,
        labels: np.ndarray,
        model_name: str,
        explained_variance: np.ndarray,
        png_path: Path,
        pdf_path: Path) -> None:
    """
    Create PCA scatter plot with cluster colors.

    Args:
        X_pca: PCA-transformed data
        labels: Cluster labels
        model_name: Name of clustering model
        explained_variance: Explained variance ratios
        png_path: Path to save PNG
        pdf_path: Path to save PDF
    """
    print_section(f"PLOTTING PCA CLUSTERS - {model_name}", "-")

    fig, ax = plt.subplots(figsize=Config.FIGSIZE)

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=Config.CMAP,
                         alpha=0.6, edgecolors='k', linewidth=0.5, s=50)

    ax.set_xlabel(
        f'PC1 ({
            explained_variance[0] *
            100:.2f}% variance)',
        fontsize=12,
        fontweight='bold')
    ax.set_ylabel(
        f'PC2 ({
            explained_variance[1] *
            100:.2f}% variance)',
        fontsize=12,
        fontweight='bold')
    ax.set_title(
        f'PCA Visualization - {model_name} Clusters',
        fontsize=14,
        fontweight='bold',
        pad=15)

    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=Config.DPI, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Saved: {png_path.name}")
    print(f"✓ Saved: {pdf_path.name}")
    plt.close()


# ==============================================================================
# SECTION 7: EXPORT FUNCTIONS
# ==============================================================================


def create_pca_notes(
        explained_variance: np.ndarray,
        best_model: str,
        best_k: int,
        output_dir: Path) -> None:
    """
    Generate detailed PCA visualization notes.

    Args:
        explained_variance: Explained variance ratios
        best_model: Best model name
        best_k: Best k value
        output_dir: Directory to save notes
    """
    print_section("GENERATING PCA NOTES", "-")

    notes_path = output_dir / 'pca_notes.md'

    with open(notes_path, 'w', encoding='utf-8') as f:
        f.write("# PCA Visualization Notes\n\n")

        f.write("## 1. What is PCA?\n\n")
        f.write(
            "**Principal Component Analysis (PCA)** is a dimensionality reduction technique that:\n")
        f.write(
            "- Identifies directions (principal components) of maximum variance in data\n")
        f.write("- Projects high-dimensional data onto lower-dimensional space\n")
        f.write("- Preserves as much information (variance) as possible\n")
        f.write("- Creates uncorrelated (orthogonal) components\n\n")

        f.write("## 2. Explained Variance\n\n")
        f.write(
            f"**PC1:** {explained_variance[0] * 100:.2f}% of total variance\n")
        f.write(
            f"**PC2:** {explained_variance[1] * 100:.2f}% of total variance\n")
        f.write(
            f"**Total:** {(explained_variance[0] + explained_variance[1]) * 100:.2f}% captured in 2D\n\n")

        f.write("### Interpretation:\n")
        total_var = (explained_variance[0] + explained_variance[1]) * 100
        if total_var > 80:
            f.write(
                "- **Excellent**: >80% variance captured, 2D visualization is highly representative\n")
        elif total_var > 60:
            f.write(
                "- **Good**: 60-80% variance captured, 2D shows main structure but loses some detail\n")
        elif total_var > 40:
            f.write(
                "- **Moderate**: 40-60% variance captured, important information may be in higher dimensions\n")
        else:
            f.write(
                "- **Limited**: <40% variance captured, data is highly multi-dimensional\n")
        f.write("\n")

        f.write("## 3. Clustering Results\n\n")
        f.write(f"**Best model:** {best_model}\n")
        f.write(f"**Number of clusters:** {best_k}\n\n")

        f.write("### What to Look For:\n")
        f.write("**Well-separated clusters:**\n")
        f.write("- Clear gaps between cluster groups\n")
        f.write("- Minimal overlap\n")
        f.write("- Indicates good clustering quality\n\n")

        f.write("**Overlapping clusters:**\n")
        f.write("- Clusters blend together\n")
        f.write(
            "- May indicate: (1) natural overlap in data, (2) wrong k value, or (3) non-spherical shapes\n\n")

        f.write("**Outliers:**\n")
        f.write("- Points far from any cluster\n")
        f.write("- May need investigation or removal\n\n")

        f.write("## 4. Limitations\n\n")
        f.write("- **Information loss:** Only 2 dimensions shown, rest is compressed\n")
        f.write("- **Linear projection:** PCA assumes linear relationships\n")
        f.write(
            "- **Visual bias:** Clusters may appear better/worse than they actually are\n")
        f.write(
            "- **Scale sensitivity:** PCA affected by feature scales (hence StandardScaler needed)\n\n")

        f.write("## 5. Next Steps\n\n")
        f.write(
            "1. **Examine cluster profiles:** Analyze original feature values per cluster\n")
        f.write("2. **Validate results:** Check if clusters make domain sense\n")
        f.write("3. **Consider 3D PCA:** If variance capture is low, try PC3\n")
        f.write("4. **Try t-SNE/UMAP:** Non-linear alternatives for complex data\n\n")

        f.write("---\n\n")
        f.write("**End of PCA Visualization Notes**\n")

    print(f"✓ PCA notes saved to: {notes_path}")


# ==============================================================================
# SECTION 8: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the entire PCA visualization workflow."""
    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    Config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("PCA VISUALIZATION - CLUSTER ANALYSIS IN 2D")
    print("=" * 120)
    print(f"\nRandom seed: {SEED}")
    print(f"PCA components: {Config.N_COMPONENTS}")

    try:
        # 1. Load processed data
        print_section("STEP 1: LOAD PROCESSED DATA")
        X = load_processed_data(Config.PROCESSED_DATA_PATH)

        # 2. Select best model
        print_section("STEP 2: SELECT BEST MODEL")
        best_model, best_k = select_best_model(
            Config.MODEL_COMPARISON_PATH, Config.MODELS_DIR)

        # 3. Fit PCA
        print_section("STEP 3: FIT PCA")
        pca, X_pca, explained_variance = fit_pca(
            X, Config.N_COMPONENTS, Config.DATA_PROCESSED_DIR)

        # 4. Load K-Means labels
        print_section("STEP 4: LOAD CLUSTER LABELS")
        kmeans_labels = load_cluster_labels(
            'K-Means', best_k, Config.MODELS_DIR, X)

        # 5. Plot K-Means clusters
        print_section("STEP 5: PLOT K-MEANS CLUSTERS")
        plot_pca_clusters(X_pca, kmeans_labels, 'K-Means', explained_variance,
                          Config.GRAPHICS_DIR / 'pca_kmeans.png',
                          Config.GRAPHICS_DIR / 'pca_kmeans.pdf')

        # 6. Try Agglomerative if available
        agg_path = Config.MODELS_DIR / f'agglomerative_k{best_k}.pkl'
        if agg_path.exists():
            print_section("STEP 6: PLOT AGGLOMERATIVE CLUSTERS")
            agg_labels = load_cluster_labels(
                'Agglomerative', best_k, Config.MODELS_DIR, X)
            plot_pca_clusters(
                X_pca,
                agg_labels,
                'Agglomerative',
                explained_variance,
                Config.GRAPHICS_DIR /
                'pca_agglomerative.png',
                Config.GRAPHICS_DIR /
                'pca_agglomerative.pdf')

        # 7. Create notes
        print_section("STEP 7: CREATE PCA NOTES")
        create_pca_notes(
            explained_variance,
            best_model,
            best_k,
            Config.OUTPUT_DIR)

        # Final summary
        print_section("PCA VISUALIZATION COMPLETED")
        print("\nOUTPUTS GENERATED:")
        print("\nGRAPHICS:")
        print(f"  • {Config.GRAPHICS_DIR / 'pca_kmeans.png'}")
        print(f"  • {Config.GRAPHICS_DIR / 'pca_kmeans.pdf'}")
        if agg_path.exists():
            print(f"  • {Config.GRAPHICS_DIR / 'pca_agglomerative.png'}")
            print(f"  • {Config.GRAPHICS_DIR / 'pca_agglomerative.pdf'}")
        print("\nDATA PROCESSED:")
        print(
            f"  • {
                Config.DATA_PROCESSED_DIR /
                'pca_variance_explained.csv'}")
        print("\nNOTES:")
        print(f"  • {Config.OUTPUT_DIR / 'pca_notes.md'}")

        print("\n" + "=" * 120)
        print("✓ PCA visualization complete!")
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
