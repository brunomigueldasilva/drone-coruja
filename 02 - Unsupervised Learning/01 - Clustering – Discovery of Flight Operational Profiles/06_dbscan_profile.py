#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
DBSCAN VISUALIZATION AND CLUSTER PROFILING
==============================================================================

Purpose: Visualize DBSCAN results and create detailed cluster profiles.

This script:
1. Loads preprocessed data and original raw data
2. Visualizes DBSCAN clustering results in PCA space, highlighting outliers
3. Loads K-Means model for cluster profiling
4. Generates cluster profiles using original feature values
5. Creates interpretable statistical summaries per cluster
6. Exports visualizations, profile tables, and analysis notes

DBSCAN identifies outliers (noise points) while K-Means profiles help
understand the business meaning of each cluster.

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
    """DBSCAN and profiling configuration parameters."""
    # Directories
    INPUT_DIR = Path('inputs')
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'
    DATA_PROCESSED_DIR = OUTPUT_DIR / 'data_processed'
    MODELS_DIR = OUTPUT_DIR / 'models'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'

    # Input files
    ORIGINAL_DATA_PATH = INPUT_DIR / 'voos_telemetria_completa.csv'
    PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / 'processed_X.npy'
    DBSCAN_MODEL_PATH = MODELS_DIR / 'dbscan.pkl'

    # PCA configuration
    N_COMPONENTS = 2

    # Visualization settings
    DPI = 300
    FIGSIZE = (14, 10)


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
# SECTION 3: DATA LOADING
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

    return X


def load_original_data(path: Path) -> pd.DataFrame:
    """
    Load original (unscaled) dataset for profiling.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with original feature values

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Original data not found at {
                path.absolute()}")

    df = pd.read_csv(path, encoding='utf-8')
    print("✓ Original data loaded successfully!")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


# ==============================================================================
# SECTION 4: DBSCAN VISUALIZATION
# ==============================================================================


def visualize_dbscan_clusters(X: np.ndarray, dbscan_model_path: Path,
                              png_path: Path, pdf_path: Path) -> np.ndarray:
    """
    Visualize DBSCAN clustering results in PCA space with outliers highlighted.

    Args:
        X: Scaled feature matrix
        dbscan_model_path: Path to saved DBSCAN model
        png_path: Path to save PNG version
        pdf_path: Path to save PDF version

    Returns:
        Array of DBSCAN labels
    """
    print_section("VISUALIZING DBSCAN CLUSTERS", "-")
    print(f"Loading DBSCAN model from: {dbscan_model_path}")

    if not dbscan_model_path.exists():
        raise FileNotFoundError(f"DBSCAN model not found: {dbscan_model_path}")

    with open(dbscan_model_path, 'rb') as f:
        dbscan = pickle.load(f)

    print("\n✓ DBSCAN model loaded!")

    labels = dbscan.labels_ if hasattr(
        dbscan, 'labels_') else dbscan.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("\nDBSCAN Results:")
    print(f"  • Number of clusters: {n_clusters}")
    print(
        f"  • Noise points (outliers): {n_noise} ({(n_noise / len(labels) * 100):.2f}%)")

    if n_clusters > 0:
        print("\nCluster distribution (excluding noise):")
        unique_clusters = sorted([la for la in set(labels) if la != -1])
        for cluster_id in unique_clusters:
            count = list(labels).count(cluster_id)
            pct = (count / len(labels)) * 100
            print(f"  • Cluster {cluster_id}: {count} samples ({pct:.2f}%)")

    print("\nFitting PCA for visualization...")
    pca = PCA(n_components=Config.N_COMPONENTS, random_state=SEED)
    X_pca = pca.fit_transform(X)

    explained_variance = pca.explained_variance_ratio_
    total_variance = sum(explained_variance) * 100
    print("✓ PCA fitted!")
    print(
        f"  PC1: {explained_variance[0] * 100:.2f}%, PC2: {explained_variance[1] * 100:.2f}%")
    print(f"  Total variance: {total_variance:.2f}%")

    print("\nCreating DBSCAN visualization...")
    fig, ax = plt.subplots(figsize=Config.FIGSIZE)

    unique_labels = sorted(set(labels))
    n_colors = max(len(unique_labels) - 1, 1)
    colors = sns.color_palette("husl", n_colors)

    color_idx = 0
    for label in unique_labels:
        if label == -1:
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c='red', marker='x',
                       s=50, alpha=0.5, label='Noise (Outliers)')
        else:
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[color_idx]],
                       alpha=0.6, edgecolors='k', linewidth=0.5, s=50,
                       label=f'Cluster {label}')
            color_idx += 1

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
    ax.set_title('DBSCAN Clustering - Outlier Detection',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=Config.DPI, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Saved: {png_path.name}")
    print(f"✓ Saved: {pdf_path.name}")
    plt.close()

    return labels


# ==============================================================================
# SECTION 5: CLUSTER PROFILING
# ==============================================================================


def find_best_kmeans_model(models_dir: Path) -> Tuple[Path, int]:
    """
    Find the best K-Means model in the models directory.

    Args:
        models_dir: Directory containing models

    Returns:
        Tuple of (model_path, k_value)
    """
    kmeans_files = list(models_dir.glob("kmeans_k*.pkl"))

    if not kmeans_files:
        raise FileNotFoundError("No K-Means models found for profiling")

    kmeans_files.sort()
    model_path = kmeans_files[0]
    k = int(model_path.stem.split('_k')[1])

    return model_path, k


def create_cluster_profiles(
        df_original: pd.DataFrame,
        X: np.ndarray,
        kmeans_model_path: Path,
        output_dir: Path) -> pd.DataFrame:
    """
    Create detailed cluster profiles using original feature values.

    Args:
        df_original: Original unscaled data
        X: Scaled feature matrix
        kmeans_model_path: Path to K-Means model
        output_dir: Directory to save profiles

    Returns:
        DataFrame with cluster profiles
    """
    print_section("CREATING CLUSTER PROFILES", "-")
    print(f"Loading K-Means model from: {kmeans_model_path}")

    with open(kmeans_model_path, 'rb') as f:
        kmeans = pickle.load(f)

    labels = kmeans.predict(X)

    print("\n✓ K-Means model loaded and labels predicted!")
    print(f"  Number of clusters: {kmeans.n_clusters}")

    df_with_clusters = df_original.copy()
    df_with_clusters['cluster'] = labels

    print("\nCluster distribution:")
    cluster_counts = df_with_clusters['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        pct = (count / len(df_with_clusters)) * 100
        print(f"  • Cluster {cluster_id}: {count} samples ({pct:.2f}%)")

    numeric_cols = df_original.select_dtypes(
        include=[np.number]).columns.tolist()

    print(
        f"\n✓ Computing cluster profiles for {
            len(numeric_cols)} numeric features...")

    profile_data = []
    for cluster_id in sorted(df_with_clusters['cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]

        profile = {'cluster': cluster_id, 'n_samples': len(cluster_data)}

        for col in numeric_cols:
            profile[f'{col}_mean'] = cluster_data[col].mean()
            profile[f'{col}_median'] = cluster_data[col].median()
            profile[f'{col}_std'] = cluster_data[col].std()

        profile_data.append(profile)

    profile_df = pd.DataFrame(profile_data)

    csv_path = output_dir / 'cluster_profile_means.csv'
    profile_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n✓ Cluster profiles saved to: {csv_path}")

    md_path = output_dir / 'cluster_profile_means.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Cluster Profile Summary\n\n")
        f.write("Mean values per cluster:\n\n")

        mean_cols = [
            col for col in profile_df.columns if col.endswith('_mean')]
        display_df = profile_df[['cluster', 'n_samples'] + mean_cols].copy()

        f.write(display_df.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("*Note: Full statistics (mean, median, std) available in CSV file.*\n")

    print(f"✓ Cluster profiles saved to: {md_path}")

    return profile_df


# ==============================================================================
# SECTION 6: EXPORT FUNCTIONS
# ==============================================================================


def create_profile_notes(
        profile_df: pd.DataFrame,
        n_clusters: int,
        output_dir: Path) -> None:
    """
    Generate detailed profiling notes.

    Args:
        profile_df: DataFrame with cluster profiles
        n_clusters: Number of clusters
        output_dir: Directory to save notes
    """
    print_section("GENERATING PROFILE NOTES", "-")

    notes_path = output_dir / 'profile_notes.md'

    with open(notes_path, 'w', encoding='utf-8') as f:
        f.write("# Cluster Profiling Notes\n\n")

        f.write("## 1. What is Cluster Profiling?\n\n")
        f.write(
            "**Cluster profiling** translates abstract cluster assignments into business insights by:\n")
        f.write("- Analyzing original feature values per cluster\n")
        f.write("- Computing descriptive statistics (mean, median, std)\n")
        f.write("- Identifying characteristic patterns\n")
        f.write("- Enabling business interpretation and actionability\n\n")

        f.write("## 2. Cluster Summary\n\n")
        f.write(f"**Total clusters identified:** {n_clusters}\n\n")

        for _, row in profile_df.iterrows():
            cluster_id = int(row['cluster'])
            n_samples = int(row['n_samples'])
            pct = (n_samples / profile_df['n_samples'].sum()) * 100

            f.write(f"### Cluster {cluster_id}\n")
            f.write(f"- **Size:** {n_samples} samples ({pct:.2f}% of data)\n")
            f.write(
                "- **Profile:** See detailed statistics in cluster_profile_means.csv\n\n")

        f.write("## 3. How to Interpret Profiles\n\n")

        f.write("### Step 1: Identify Distinctive Features\n")
        f.write("- Compare mean values across clusters\n")
        f.write("- Look for features with large differences\n")
        f.write("- These are the defining characteristics\n\n")

        f.write("### Step 2: Assign Business Meaning\n")
        f.write("- Translate feature patterns into domain concepts\n")
        f.write("- Example: High altitude + Long duration = Long-haul flights\n")
        f.write("- Create descriptive labels for each cluster\n\n")

        f.write("### Step 3: Validate with Experts\n")
        f.write("- Share profiles with domain experts\n")
        f.write("- Verify that clusters align with known categories\n")
        f.write("- Refine interpretations based on feedback\n\n")

        f.write("### Step 4: Actionable Insights\n")
        f.write("- Develop strategies specific to each cluster\n")
        f.write("- Allocate resources based on cluster characteristics\n")
        f.write("- Monitor cluster evolution over time\n\n")

        f.write("## 4. DBSCAN Outlier Analysis\n\n")
        f.write("**Outliers (noise points)** identified by DBSCAN should be:\n")
        f.write("- Investigated individually for anomalies\n")
        f.write("- Checked for data quality issues\n")
        f.write("- Analyzed for rare but legitimate patterns\n")
        f.write("- Considered for special handling or exclusion\n\n")

        f.write("## 5. Next Steps\n\n")
        f.write("1. **Deep dive analysis:** Examine individual samples per cluster\n")
        f.write(
            "2. **Feature importance:** Identify which features best separate clusters\n")
        f.write("3. **Business rules:** Create decision rules for cluster assignment\n")
        f.write("4. **Deployment:** Use profiles for production systems\n")
        f.write("5. **Monitoring:** Track cluster distributions over time\n\n")

        f.write("---\n\n")
        f.write("**End of Profiling Notes**\n")

    print(f"✓ Profile notes saved to: {notes_path}")


# ==============================================================================
# SECTION 7: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the entire DBSCAN visualization and profiling workflow."""
    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    Config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("DBSCAN VISUALIZATION AND CLUSTER PROFILING")
    print("=" * 120)
    print(f"\nRandom seed: {SEED}")

    try:
        # 1. Load processed data
        print_section("STEP 1: LOAD PROCESSED DATA")
        X = load_processed_data(Config.PROCESSED_DATA_PATH)

        # 2. Visualize DBSCAN clusters
        print_section("STEP 2: VISUALIZE DBSCAN CLUSTERS")
        visualize_dbscan_clusters(
            X,
            Config.DBSCAN_MODEL_PATH,
            Config.GRAPHICS_DIR /
            'pca_dbscan.png',
            Config.GRAPHICS_DIR /
            'pca_dbscan.pdf')

        # 3. Load original data
        print_section("STEP 3: LOAD ORIGINAL DATA")
        df_original = load_original_data(Config.ORIGINAL_DATA_PATH)

        # 4. Find best K-Means model
        print_section("STEP 4: FIND BEST K-MEANS MODEL")
        kmeans_model_path, k_value = find_best_kmeans_model(Config.MODELS_DIR)
        print(f"✓ Found K-Means model: {kmeans_model_path.name} (k={k_value})")

        # 5. Create cluster profiles
        print_section("STEP 5: CREATE CLUSTER PROFILES")
        profile_df = create_cluster_profiles(
            df_original, X, kmeans_model_path, Config.DATA_PROCESSED_DIR)

        # 6. Create profiling notes
        print_section("STEP 6: CREATE PROFILING NOTES")
        create_profile_notes(profile_df, k_value, Config.OUTPUT_DIR)

        # Final summary
        print_section("DBSCAN VISUALIZATION AND PROFILING COMPLETED")
        print("\nOUTPUTS GENERATED:")
        print("\nGRAPHICS:")
        print(f"  • {Config.GRAPHICS_DIR / 'pca_dbscan.png'}")
        print(f"  • {Config.GRAPHICS_DIR / 'pca_dbscan.pdf'}")
        print("\nDATA PROCESSED:")
        print(f"  • {Config.DATA_PROCESSED_DIR / 'cluster_profile_means.csv'}")
        print(f"  • {Config.DATA_PROCESSED_DIR / 'cluster_profile_means.md'}")
        print("\nNOTES:")
        print(f"  • {Config.OUTPUT_DIR / 'profile_notes.md'}")

        print("\n" + "=" * 120)
        print("✓ DBSCAN visualization and cluster profiling complete!")
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
