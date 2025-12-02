#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ELBOW METHOD - OPTIMAL K SELECTION FOR K-MEANS
==============================================================================

Purpose: Determine candidate k values for K-Means clustering using the Elbow Method.

This script:
1. Loads preprocessed and scaled feature matrix
2. Computes Within-Cluster Sum of Squares (WCSS/inertia) for k values from 2 to 10
3. Calculates percentage drops in WCSS between consecutive k values
4. Identifies elbow candidates using heuristic analysis
5. Generates elbow curve visualization with candidate markers
6. Exports results and detailed analysis notes

The elbow point indicates where adding more clusters yields diminishing returns
in variance reduction, helping to select an optimal k value.

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

# Configuration Constants
SEED = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore')


class Config:
    """Elbow method configuration parameters."""
    # Directories
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'
    DATA_PROCESSED_DIR = OUTPUT_DIR / 'data_processed'
    MODELS_DIR = OUTPUT_DIR / 'models'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'

    # Input file
    PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / 'processed_X.npy'

    # K-Means parameters
    K_RANGE = range(2, 11)
    N_INIT = 20
    MAX_ITER = 500
    TOLERANCE = 1e-4

    # Visualization settings
    DPI = 300
    FIGSIZE = (12, 6)


# Visualization configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = Config.FIGSIZE


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def save_plot(filepath: Path, dpi: int = 300) -> None:
    """
    Save current plot to file.

    Args:
        filepath: Path to save the plot
        dpi: Resolution in dots per inch
    """
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {filepath.name}")
    plt.close()


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
                path.absolute()}\n" "Please run '02_preprocessing.py' first.")

    X = np.load(path)
    print("✓ Data loaded successfully!")
    print(f"  Shape: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Data type: {X.dtype}")
    print(f"  Value range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  Mean: {X.mean():.4f}, Std: {X.std():.4f}")

    return X


# ==============================================================================
# SECTION 4: ELBOW METHOD COMPUTATION
# ==============================================================================


def compute_elbow_curve(X: np.ndarray,
                        k_range: range,
                        n_init: int,
                        random_state: int,
                        max_iter: int,
                        tol: float) -> Dict[int,
                                            float]:
    """
    Compute WCSS (inertia) for different k values.

    Args:
        X: Feature matrix
        k_range: Range of k values to test
        n_init: Number of initializations
        random_state: Random seed
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Dictionary with k: inertia pairs
    """
    print_section("COMPUTING ELBOW CURVE (WCSS)", "-")
    print(f"Testing k values: {list(k_range)}")
    print("K-Means parameters:")
    print(f"  • n_init: {n_init}")
    print(f"  • random_state: {random_state}")
    print(f"  • max_iter: {max_iter}")
    print(f"  • tol: {tol}")

    inertia_values = {}

    print("\nFitting K-Means for each k value:")
    print(f"{'k':<5} {'Inertia (WCSS)':<20} {'Time':<15}")
    print("-" * 120)

    for k in k_range:
        start_time = time.time()

        kmeans = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol
        )
        kmeans.fit(X)

        inertia = kmeans.inertia_
        inertia_values[k] = inertia

        elapsed = time.time() - start_time
        print(f"{k:<5} {inertia:<20.2f} {elapsed:.2f}s")

    print(f"\n✓ Computed WCSS for {len(inertia_values)} k values")
    return inertia_values


def calculate_percentage_drops(
        inertia_values: Dict[int, float]) -> Dict[int, float]:
    """
    Calculate percentage drops in WCSS between consecutive k values.

    Args:
        inertia_values: Dictionary with k: inertia pairs

    Returns:
        Dictionary with k: percentage_drop pairs
    """
    k_sorted = sorted(inertia_values.keys())
    percentage_drops = {}

    for i in range(1, len(k_sorted)):
        k_prev = k_sorted[i - 1]
        k_curr = k_sorted[i]

        drop = inertia_values[k_prev] - inertia_values[k_curr]
        pct_drop = (drop / inertia_values[k_prev]) * 100
        percentage_drops[k_curr] = pct_drop

    return percentage_drops


def identify_elbow_candidates(inertia_values: Dict[int, float],
                              percentage_drops: Dict[int, float]) -> List[int]:
    """
    Identify candidate k values based on elbow heuristics.

    Args:
        inertia_values: Dictionary with k: inertia pairs
        percentage_drops: Dictionary with k: percentage_drop pairs

    Returns:
        List of recommended k values
    """
    print_section("IDENTIFYING ELBOW CANDIDATES", "-")

    if not percentage_drops:
        print("⚠ WARNING: No percentage drops to analyze")
        return []

    avg_drop = np.mean(list(percentage_drops.values()))
    max_drop = max(percentage_drops.values())
    max_drop_k = [k for k, drop in percentage_drops.items() if drop ==
                  max_drop][0]

    print("\nDrop analysis:")
    print(f"  • Average drop: {avg_drop:.2f}%")
    print(f"  • Maximum drop: {max_drop:.2f}% at k={max_drop_k}")

    print(f"\n{'k':<5} {'WCSS':<15} {'Drop %':<15} {'Status':<30}")
    print("-" * 120)

    candidates = []
    k_sorted = sorted(inertia_values.keys())

    for k in k_sorted:
        wcss = inertia_values[k]
        drop = percentage_drops.get(k, 0.0)

        status = ""
        if k == max_drop_k:
            status = "★ Largest drop"
            candidates.append(k)
        elif drop < avg_drop and k > min(k_sorted):
            prev_k = k_sorted[k_sorted.index(k) - 1]
            if percentage_drops.get(prev_k, 0) >= avg_drop:
                status = "◆ Elbow candidate"
                if k not in candidates:
                    candidates.append(k)

        print(f"{k:<5} {wcss:<15.2f} {drop:<15.2f} {status:<30}")

    candidates = sorted(list(set(candidates)))
    print(f"\n✓ Identified {len(candidates)} elbow candidate(s): {candidates}")

    return candidates


# ==============================================================================
# SECTION 5: VISUALIZATION FUNCTIONS
# ==============================================================================


def plot_elbow_curve(inertia_values: Dict[int, float], candidates: List[int],
                     png_path: Path, pdf_path: Path) -> None:
    """
    Create elbow curve visualization.

    Args:
        inertia_values: Dictionary with k: inertia pairs
        candidates: List of candidate k values
        png_path: Path to save PNG file
        pdf_path: Path to save PDF file
    """
    print_section("GENERATING ELBOW CURVE PLOT", "-")

    k_values = sorted(inertia_values.keys())
    wcss_values = [inertia_values[k] for k in k_values]

    fig, ax = plt.subplots(figsize=Config.FIGSIZE)

    ax.plot(
        k_values,
        wcss_values,
        'bo-',
        linewidth=2,
        markersize=8,
        label='WCSS')

    for k in candidates:
        ax.plot(
            k,
            inertia_values[k],
            'r*',
            markersize=15,
            label=f'Candidate k={k}' if k == candidates[0] else '')

    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel(
        'Within-Cluster Sum of Squares (WCSS)',
        fontsize=12,
        fontweight='bold')
    ax.set_title('Elbow Method - Optimal K Selection',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    ax.set_xticks(k_values)

    save_plot(png_path, dpi=Config.DPI)

    fig, ax = plt.subplots(figsize=Config.FIGSIZE)
    ax.plot(
        k_values,
        wcss_values,
        'bo-',
        linewidth=2,
        markersize=8,
        label='WCSS')
    for k in candidates:
        ax.plot(
            k,
            inertia_values[k],
            'r*',
            markersize=15,
            label=f'Candidate k={k}' if k == candidates[0] else '')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel(
        'Within-Cluster Sum of Squares (WCSS)',
        fontsize=12,
        fontweight='bold')
    ax.set_title('Elbow Method - Optimal K Selection',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(k_values)
    save_plot(pdf_path)


# ==============================================================================
# SECTION 6: EXPORT FUNCTIONS
# ==============================================================================


def save_elbow_results(
        inertia_values: Dict[int, float], output_path: Path) -> None:
    """
    Save elbow results to CSV file.

    Args:
        inertia_values: Dictionary with k: inertia pairs
        output_path: Path to save CSV file
    """
    print_section("SAVING ELBOW RESULTS", "-")

    results_df = pd.DataFrame({
        'k': sorted(inertia_values.keys()),
        'wcss': [inertia_values[k] for k in sorted(inertia_values.keys())]
    })

    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ Saved results to: {output_path}")


def create_elbow_notes(inertia_values: Dict[int,
                                            float],
                       percentage_drops: Dict[int,
                                              float],
                       candidates: List[int],
                       output_dir: Path) -> None:
    """
    Generate detailed elbow method notes in Markdown format.

    Args:
        inertia_values: Dictionary with k: inertia pairs
        percentage_drops: Dictionary with percentage drops
        candidates: List of candidate k values
        output_dir: Directory to save the notes
    """
    print_section("GENERATING ELBOW METHOD NOTES", "-")

    notes_path = output_dir / 'elbow_notes.md'

    with open(notes_path, 'w', encoding='utf-8') as f:
        f.write("# Elbow Method Analysis Notes\n\n")

        f.write("## 1. What is WCSS (Within-Cluster Sum of Squares)?\n\n")
        f.write(
            "**Definition:** WCSS measures cluster compactness by computing the sum of squared distances "
            "from each point to its cluster centroid, then sums across all clusters.\n\n")

        f.write("**Formula:**\n")
        f.write("```\n")
        f.write("WCSS = Σ(i=1 to k) Σ(x in Cluster_i) ||x - centroid_i||²\n")
        f.write("```\n\n")

        f.write("Where:\n")
        f.write("- `k` = number of clusters\n")
        f.write("- `x` = data point\n")
        f.write("- `centroid_i` = center of cluster i\n")
        f.write("- `||x - centroid_i||²` = squared Euclidean distance\n\n")

        f.write("### Why Lower WCSS is Better:\n\n")
        f.write(
            "- **Lower WCSS** → Points are closer to their centroids → **Tighter, more cohesive clusters**\n")
        f.write(
            "- **Higher WCSS** → Points are spread out → **Loose, less meaningful clusters**\n\n")

        f.write("### The Trade-off:\n\n")
        f.write("- **More clusters (higher k)** → Always decreases WCSS\n")
        f.write(
            "- **Extreme case:** k = n (each point is its own cluster) → WCSS = 0\n")
        f.write(
            "- **But:** Too many clusters → Overfitting, no meaningful patterns\n\n")

        f.write(
            "**Key insight:** We want the **smallest k** that provides **good cluster compactness** "
            "without overfitting.\n\n")

        f.write("## 2. How to Spot the Elbow\n\n")
        f.write("### Visual Inspection:\n\n")
        f.write(
            "The elbow curve plots WCSS vs. k. The shape typically resembles an arm:\n\n")
        f.write("```\n")
        f.write("WCSS\n")
        f.write("  |\n")
        f.write("  |●  (k=2: Large drop)\n")
        f.write("  | \\\n")
        f.write("  |  ●  (k=3: Moderate drop)\n")
        f.write("  |   \\\n")
        f.write("  |    ●___  (k=4: ELBOW - diminishing returns begin)\n")
        f.write("  |        ●___\n")
        f.write("  |            ●___\n")
        f.write("  |________________k\n")
        f.write("```\n\n")

        f.write("### Mathematical Detection:\n\n")
        f.write("1. **First Derivative (Rate of Decrease):**\n")
        f.write(
            "   - Calculate percentage drop: `(WCSS[k-1] - WCSS[k]) / WCSS[k-1] × 100%`\n")
        f.write("   - Elbow occurs where drops become smaller\n\n")

        f.write("2. **Second Derivative (Curvature):**\n")
        f.write("   - Measures change in the rate of decrease\n")
        f.write(
            "   - Maximum absolute curvature indicates the sharpest bend (elbow)\n\n")

        f.write("3. **Heuristic Thresholds:**\n")
        f.write("   - Find k where drop falls below average\n")
        f.write("   - Identify largest single drop\n\n")

        f.write("## 3. Results for This Dataset\n\n")

        f.write("### WCSS Values:\n\n")
        f.write("| k | WCSS (Inertia) | Drop from Previous | Drop % |\n")
        f.write("|---|----------------|-------------------|--------|\n")

        k_sorted = sorted(inertia_values.keys())
        for i, k in enumerate(k_sorted):
            wcss = inertia_values[k]
            if i == 0:
                f.write(f"| {k} | {wcss:.2f} | - | - |\n")
            else:
                k_prev = k_sorted[i - 1]
                drop = inertia_values[k_prev] - wcss
                drop_pct = percentage_drops[k]
                f.write(
                    f"| {k} | {
                        wcss:.2f} | {
                        drop:.2f} | {
                        drop_pct:.2f}% |\n")

        f.write("\n")

        f.write("### Recommended K Values:\n\n")
        f.write(f"**Candidates identified: {candidates}**\n\n")

        for k in candidates:
            f.write(f"**k = {k}:**\n")
            if k in percentage_drops:
                f.write(f"- WCSS: {inertia_values[k]:.2f}\n")
                f.write(f"- Drop from k={k - 1}: {percentage_drops[k]:.2f}%\n")
            f.write(
                "- Rationale: Detected as elbow candidate based on drop rate analysis\n\n")

        f.write("## 4. Caveats and Limitations\n\n")

        f.write("### The Elbow May Be Ambiguous:\n\n")
        f.write(
            "- **Smooth curves:** Some datasets show gradual decline without clear elbow\n")
        f.write("- **Multiple elbows:** May indicate hierarchical structure\n")
        f.write(
            "- **Subjective:** Different analysts may identify different elbows\n\n")

        f.write("### WCSS Has Limitations:\n\n")
        f.write("1. **Always decreases:** Adding clusters can never increase WCSS\n")
        f.write(
            "2. **Scale-dependent:** Sensitive to feature scales (hence need for standardization)\n")
        f.write(
            "3. **Assumes spherical clusters:** Works best for convex, well-separated clusters\n")
        f.write("4. **No statistical test:** No p-value or confidence interval\n\n")

        f.write("### Best Practices:\n\n")
        f.write("**Don't rely solely on the elbow method!** Combine with:\n\n")

        f.write("1. **Silhouette Analysis:**\n")
        f.write(
            "   - Measures how similar points are to their own cluster vs. other clusters\n")
        f.write("   - Score ranges from -1 (wrong cluster) to +1 (perfect cluster)\n")
        f.write("   - Provides per-sample and per-cluster quality metrics\n\n")

        f.write("2. **Gap Statistic:**\n")
        f.write(
            "   - Compares WCSS to expected WCSS under null (random) distribution\n")
        f.write("   - Statistical test for optimal k\n")
        f.write("   - More rigorous than visual inspection\n\n")

        f.write("3. **Domain Knowledge:**\n")
        f.write("   - Does k align with business/scientific expectations?\n")
        f.write("   - Are clusters interpretable and actionable?\n")
        f.write(
            "   - Example: Airlines may prefer 3-5 flight types (short/medium/long-haul)\n\n")

        f.write("4. **Cluster Stability:**\n")
        f.write("   - Run K-Means multiple times with different initializations\n")
        f.write("   - Check if cluster assignments are consistent\n")
        f.write("   - Use metrics like Adjusted Rand Index (ARI)\n\n")

        f.write("5. **Visual Inspection:**\n")
        f.write("   - Plot clusters in 2D/3D using PCA or t-SNE\n")
        f.write("   - Verify that clusters look meaningful and separated\n\n")

        f.write("## 5. Next Steps\n\n")
        f.write("Based on this elbow analysis, proceed with:\n\n")

        f.write(
            "1. **Silhouette analysis** for candidate k values to validate cluster quality\n")
        f.write(
            "2. **Fit K-Means** with recommended k values and examine cluster characteristics\n")
        f.write(
            "3. **Compare with other algorithms** (DBSCAN, Hierarchical) for robustness\n")
        f.write("4. **Visualize clusters** in reduced dimensions (PCA/t-SNE)\n")
        f.write(
            "5. **Interpret clusters** based on original feature values and domain context\n\n")

        f.write("---\n\n")
        f.write("**End of Elbow Method Notes**\n")

    print(f"✓ Elbow method notes saved to: {notes_path}")


# ==============================================================================
# SECTION 7: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the entire elbow method analysis."""
    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    Config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("ELBOW METHOD - OPTIMAL K SELECTION FOR K-MEANS")
    print("=" * 120)
    print(f"\nRandom seed: {SEED}")
    print(f"K range: {list(Config.K_RANGE)}")

    try:
        # 1. Load processed data
        print_section("STEP 1: LOAD PROCESSED DATA")
        X = load_processed_data(Config.PROCESSED_DATA_PATH)

        # 2. Compute elbow curve
        print_section("STEP 2: COMPUTE ELBOW CURVE")
        inertia_values = compute_elbow_curve(
            X,
            Config.K_RANGE,
            Config.N_INIT,
            SEED,
            Config.MAX_ITER,
            Config.TOLERANCE)

        # 3. Calculate percentage drops
        percentage_drops = calculate_percentage_drops(inertia_values)

        # 4. Identify elbow candidates
        print_section("STEP 3: IDENTIFY ELBOW CANDIDATES")
        candidates = identify_elbow_candidates(
            inertia_values, percentage_drops)

        # 5. Plot elbow curve
        print_section("STEP 4: GENERATE VISUALIZATIONS")
        plot_elbow_curve(inertia_values, candidates,
                         Config.GRAPHICS_DIR / "elbow_wcss.png",
                         Config.GRAPHICS_DIR / "elbow_wcss.pdf")

        # 6. Save results
        print_section("STEP 5: EXPORT RESULTS")
        save_elbow_results(
            inertia_values,
            Config.DATA_PROCESSED_DIR /
            "elbow_wcss.csv")
        create_elbow_notes(
            inertia_values,
            percentage_drops,
            candidates,
            Config.OUTPUT_DIR)

        # Final summary
        print_section("ELBOW METHOD ANALYSIS COMPLETED")
        print("\nOUTPUTS GENERATED:")
        print("\nGRAPHICS:")
        print(f"  • {Config.GRAPHICS_DIR / 'elbow_wcss.png'}")
        print(f"  • {Config.GRAPHICS_DIR / 'elbow_wcss.pdf'}")
        print("\nDATA PROCESSED:")
        print(f"  • {Config.DATA_PROCESSED_DIR / 'elbow_wcss.csv'}")
        print("\nNOTES:")
        print(f"  • {Config.OUTPUT_DIR / 'elbow_notes.md'}")

        print(f"\nRECOMMENDED K VALUES: {candidates}")
        print("\nNext step: Run silhouette analysis to validate these candidates!")

        print("\n" + "=" * 120)
        print("✓ Elbow method analysis complete!")
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
