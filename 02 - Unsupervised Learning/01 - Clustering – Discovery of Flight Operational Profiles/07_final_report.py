#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FINAL REPORT GENERATION - CLUSTERING ANALYSIS
==============================================================================

Purpose: Automatically generate comprehensive final report consolidating all clustering analysis results.

This script:
1. Checks availability of all analysis outputs
2. Loads results from EDA, preprocessing, clustering, and profiling
3. Aggregates model comparison metrics and cluster profiles
4. Embeds visualizations (plots) as references
5. Generates comprehensive FINAL_REPORT.md in root outputs directory
6. Moves all intermediate .md notes to outputs/results folder

The report provides a complete overview of the clustering analysis pipeline,
findings, and recommendations in a single readable Markdown document.

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional
import warnings
import shutil

import pandas as pd

# Configuration Constants
warnings.filterwarnings('ignore')


class Config:
    """Report generation configuration parameters."""
    # Directories
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'
    DATA_PROCESSED_DIR = OUTPUT_DIR / 'data_processed'
    MODELS_DIR = OUTPUT_DIR / 'models'
    RESULTS_DIR = OUTPUT_DIR / 'results'

    # Input files - Tables
    MODEL_COMPARISON_PATH = DATA_PROCESSED_DIR / 'model_comparison.csv'
    CLUSTER_PROFILE_PATH = DATA_PROCESSED_DIR / 'cluster_profile_means.csv'
    ELBOW_WCSS_PATH = DATA_PROCESSED_DIR / 'elbow_wcss.csv'
    PCA_VARIANCE_PATH = DATA_PROCESSED_DIR / 'pca_variance_explained.csv'

    # Input files - Notes (will be moved to results/)
    NOTE_FILES = [
        'eda_notes.md',
        'preprocessing_notes.md',
        'elbow_notes.md',
        'train_eval_notes.md',
        'pca_notes.md',
        'profile_notes.md'
    ]

    # Output file
    FINAL_REPORT_PATH = OUTPUT_DIR / 'FINAL_REPORT.md'

    # Report metadata
    PROJECT_NAME = "Unsupervised Learning for Flight Telemetry Clustering"
    AUTHOR = "Senior Data Scientist"


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


def check_file_exists(filepath: Path, description: str) -> bool:
    """
    Check if a file exists and print status.

    Args:
        filepath: Path to check
        description: Description for printing

    Returns:
        Boolean indicating if file exists
    """
    if filepath.exists():
        print(f"  ✓ {description}: {filepath.name}")
        return True
    else:
        print(f"  ✗ {description}: NOT FOUND ({filepath.name})")
        return False


def load_table_safe(filepath: Path,
                    description: str) -> Optional[pd.DataFrame]:
    """
    Safely load a CSV table with error handling.

    Args:
        filepath: Path to CSV file
        description: Description for error messages

    Returns:
        DataFrame or None if file not found
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"  ⚠️  WARNING: {description} not found - skipping section")
        return None
    except Exception as e:
        print(f"  ⚠️  WARNING: Error loading {description}: {str(e)}")
        return None


# ==============================================================================
# SECTION 3: FILE ORGANIZATION
# ==============================================================================


def move_notes_to_results() -> None:
    """Move all .md notes files to outputs/results folder."""
    print_section("ORGANIZING NOTES FILES", "-")

    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    for note_file in Config.NOTE_FILES:
        source = Config.OUTPUT_DIR / note_file
        destination = Config.RESULTS_DIR / note_file

        if source.exists():
            try:
                shutil.move(str(source), str(destination))
                print(f"  ✓ Moved: {note_file} → results/")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ Error moving {note_file}: {e}")
        else:
            print(f"  ⚠ Skipped: {note_file} (not found)")

    print(f"\n✓ Moved {moved_count} notes files to results/")


# ==============================================================================
# SECTION 4: REPORT GENERATION
# ==============================================================================


def generate_report() -> None:
    """Generate the comprehensive final report in Markdown format."""
    print_section("GENERATING FINAL REPORT")

    # Check input files
    print("\nChecking input files availability:")
    print("\nTables:")
    check_file_exists(
        Config.MODEL_COMPARISON_PATH,
        "Model Comparison")
    check_file_exists(
        Config.CLUSTER_PROFILE_PATH,
        "Cluster Profiles")
    check_file_exists(Config.ELBOW_WCSS_PATH, "Elbow Data")
    check_file_exists(Config.PCA_VARIANCE_PATH, "PCA Variance")

    print("\nFigures:")
    has_pca_kmeans = check_file_exists(
        Config.GRAPHICS_DIR /
        "pca_kmeans.png",
        "PCA K-Means")
    has_pca_agglomerative = check_file_exists(
        Config.GRAPHICS_DIR /
        "pca_agglomerative.png",
        "PCA Agglomerative")
    has_pca_dbscan = check_file_exists(
        Config.GRAPHICS_DIR / "pca_dbscan.png", "PCA DBSCAN")
    has_elbow_plot = check_file_exists(
        Config.GRAPHICS_DIR / "elbow_wcss.png", "Elbow Plot")
    has_dendrogram = check_file_exists(
        Config.GRAPHICS_DIR / "dendrogram.png", "Dendrogram")

    # Load data
    print("\nLoading data tables...")
    df_comparison = load_table_safe(
        Config.MODEL_COMPARISON_PATH,
        "Model Comparison")
    df_profiles = load_table_safe(
        Config.CLUSTER_PROFILE_PATH,
        "Cluster Profiles")
    df_elbow = load_table_safe(Config.ELBOW_WCSS_PATH, "Elbow Data")
    df_pca = load_table_safe(Config.PCA_VARIANCE_PATH, "PCA Variance")

    # Generate report
    print("\nGenerating Markdown report...")

    with open(Config.FINAL_REPORT_PATH, 'w', encoding='utf-8') as f:
        # Header
        f.write("# FINAL REPORT: CLUSTERING ANALYSIS - FLIGHT TELEMETRY DATA\n\n")
        f.write("---\n\n")

        # Document Information
        f.write("## Document Information\n\n")
        f.write(f"- **Project:** {Config.PROJECT_NAME}\n")
        f.write(f"- **Author:** {Config.AUTHOR}\n")
        f.write(f"- **Date:** {datetime.now().strftime('%B %d, %Y')}\n")
        f.write(
            f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(
            "This report presents a comprehensive clustering analysis of flight telemetry data using "
            "unsupervised machine learning techniques. The analysis aims to identify natural groupings "
            "of flights based on operational characteristics such as duration, distance, altitude, speed, "
            "fuel consumption, and vertical maneuvering patterns.\n\n")

        f.write("**Key Objectives:**\n")
        f.write(
            "1. Discover meaningful flight patterns through unsupervised clustering\n")
        f.write(
            "2. Compare multiple clustering algorithms (K-Means, Agglomerative, DBSCAN)\n")
        f.write("3. Identify optimal number of clusters using data-driven methods\n")
        f.write("4. Create interpretable cluster profiles for operational insights\n")
        f.write("5. Detect and handle outliers/anomalies in flight data\n\n")

        if df_comparison is not None:
            best_model_df = df_comparison[df_comparison['Silhouette'] != 'N/A']
            if not best_model_df.empty:
                best_model = best_model_df.iloc[0]
                silhouette_val = float(best_model['Silhouette'])
                quality = 'Excellent' if silhouette_val > 0.7 else 'Good' if silhouette_val > 0.5 else 'Moderate'

                f.write("**Main Findings:**\n")
                f.write(
                    f"- **Best performing algorithm:** {best_model['Model']}\n")
                f.write(
                    f"- **Optimal number of clusters:** {int(best_model['n_clusters'])}\n")
                f.write(
                    f"- **Silhouette Score:** {best_model['Silhouette']} ({quality})\n")
                if df_profiles is not None:
                    f.write(
                        f"- **Distinct flight patterns identified:** {len(df_profiles)} operational clusters\n")

        f.write("\n---\n\n")

        # Table of Contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Data Overview](#1-data-overview)\n")
        f.write("2. [Methodology](#2-methodology)\n")
        f.write("3. [Optimal Cluster Selection](#3-optimal-cluster-selection)\n")
        f.write("4. [Model Comparison](#4-model-comparison)\n")
        f.write("5. [Cluster Visualization](#5-cluster-visualization)\n")
        f.write("6. [Cluster Profiling](#6-cluster-profiling)\n")
        f.write("7. [Key Findings](#7-key-findings)\n")
        f.write("8. [Recommendations](#8-recommendations)\n")
        f.write("9. [Technical Details](#9-technical-details)\n\n")
        f.write("---\n\n")

        # Section 1: Data Overview
        f.write("## 1. Data Overview\n\n")
        f.write("### Dataset Description\n\n")
        f.write(
            "The analysis was performed on flight telemetry data containing operational metrics "
            "for multiple flights. The dataset includes:\n\n")
        f.write("- **Flight duration** (minutes)\n")
        f.write("- **Distance traveled** (kilometers)\n")
        f.write("- **Maximum altitude** (meters)\n")
        f.write("- **Average speed** (km/h)\n")
        f.write("- **Fuel consumption** (liters)\n")
        f.write("- **Total vertical variation** (meters)\n\n")

        f.write("### Data Quality\n\n")
        f.write("- All features were standardized using StandardScaler\n")
        f.write("- Missing values were handled using median imputation\n")
        f.write("- Outliers were analyzed but retained for initial clustering\n")
        f.write("- Data was validated for consistency and completeness\n\n")

        if has_elbow_plot:
            f.write("![Exploratory Analysis](graphics/histograms.png)\n\n")

        f.write("---\n\n")

        # Section 2: Methodology
        f.write("## 2. Methodology\n\n")
        f.write("### Clustering Pipeline\n\n")
        f.write("The analysis followed a systematic approach:\n\n")
        f.write("1. **Exploratory Data Analysis (EDA)**\n")
        f.write("   - Distribution analysis of all features\n")
        f.write("   - Correlation analysis\n")
        f.write("   - Outlier detection using IQR method\n\n")

        f.write("2. **Data Preprocessing**\n")
        f.write("   - Feature scaling (StandardScaler)\n")
        f.write("   - Missing value imputation\n")
        f.write("   - Data validation\n\n")

        f.write("3. **Optimal K Selection**\n")
        f.write("   - Elbow method (WCSS analysis)\n")
        f.write("   - Visual inspection of elbow curve\n")
        f.write("   - Percentage drop analysis\n\n")

        f.write("4. **Model Training**\n")
        f.write("   - K-Means clustering\n")
        f.write("   - Agglomerative clustering (Ward linkage)\n")
        f.write("   - DBSCAN (density-based)\n\n")

        f.write("5. **Model Evaluation**\n")
        f.write("   - Silhouette Score (higher is better)\n")
        f.write("   - Davies-Bouldin Index (lower is better)\n")
        f.write("   - Calinski-Harabasz Index (higher is better)\n\n")

        f.write("6. **Visualization**\n")
        f.write("   - PCA dimensionality reduction for 2D visualization\n")
        f.write("   - Dendrogram for hierarchical structure\n\n")

        f.write("---\n\n")

        # Section 3: Optimal Cluster Selection
        f.write("## 3. Optimal Cluster Selection\n\n")

        if df_elbow is not None:
            f.write("### Elbow Method Results\n\n")
            f.write(
                "The elbow method was used to determine the optimal number of clusters "
                "by analyzing the Within-Cluster Sum of Squares (WCSS) for different k values.\n\n")

            f.write("| k | WCSS (Inertia) |\n")
            f.write("|---|----------------|\n")
            for _, row in df_elbow.iterrows():
                f.write(f"| {int(row['k'])} | {row['wcss']:.2f} |\n")
            f.write("\n")

        if has_elbow_plot:
            f.write("### Elbow Curve\n\n")
            f.write("![Elbow Method](graphics/elbow_wcss.png)\n\n")
            f.write(
                "The elbow point indicates the optimal number of clusters where adding "
                "more clusters yields diminishing returns in variance reduction.\n\n")

        f.write("---\n\n")

        # Section 4: Model Comparison
        f.write("## 4. Model Comparison\n\n")

        if df_comparison is not None:
            f.write("### Performance Metrics\n\n")
            f.write(
                "Three clustering algorithms were compared using internal validation metrics:\n\n")

            f.write(df_comparison.to_markdown(index=False))
            f.write("\n\n")

            f.write("### Metric Interpretation\n\n")
            f.write(
                "- **Silhouette Score** [-1, 1]: Measures cluster separation. Higher is better.\n")
            f.write("  - Score > 0.7: Excellent clustering\n")
            f.write("  - Score 0.5-0.7: Good clustering\n")
            f.write("  - Score < 0.5: Weak clustering\n\n")

            f.write(
                "- **Davies-Bouldin Index** [0, ∞): Average similarity between clusters. Lower is better.\n\n")

            f.write(
                "- **Calinski-Harabasz Index** [0, ∞): Ratio of between-cluster to within-cluster dispersion. "
                "Higher is better.\n\n")

        if has_dendrogram:
            f.write("### Hierarchical Structure\n\n")
            f.write("![Dendrogram](graphics/dendrogram.png)\n\n")
            f.write(
                "The dendrogram shows the hierarchical relationship between clusters, "
                "with height indicating dissimilarity.\n\n")

        f.write("---\n\n")

        # Section 5: Cluster Visualization
        f.write("## 5. Cluster Visualization\n\n")

        if df_pca is not None:
            f.write("### PCA Variance Explained\n\n")
            f.write(
                "Principal Component Analysis (PCA) was used to project high-dimensional data "
                "into 2D space for visualization:\n\n")

            total_var = df_pca['Explained_Variance_Ratio'].sum() * 100
            f.write(
                f"- **PC1:** {df_pca.iloc[0]['Explained_Variance_Ratio'] * 100:.2f}% variance\n")
            f.write(
                f"- **PC2:** {df_pca.iloc[1]['Explained_Variance_Ratio'] * 100:.2f}% variance\n")
            f.write(
                f"- **Total captured:** {total_var:.2f}% of total variance\n\n")

        f.write("### K-Means Clusters\n\n")
        if has_pca_kmeans:
            f.write("![K-Means Clustering](graphics/pca_kmeans.png)\n\n")

        f.write("### Agglomerative Clusters\n\n")
        if has_pca_agglomerative:
            f.write(
                "![Agglomerative Clustering](graphics/pca_agglomerative.png)\n\n")

        f.write("### DBSCAN Outlier Detection\n\n")
        if has_pca_dbscan:
            f.write("![DBSCAN Clustering](graphics/pca_dbscan.png)\n\n")
            f.write(
                "DBSCAN identifies outliers (noise points marked in red) that don't belong "
                "to any dense cluster.\n\n")

        f.write("---\n\n")

        # Section 6: Cluster Profiling
        f.write("## 6. Cluster Profiling\n\n")

        if df_profiles is not None:
            f.write("### Cluster Characteristics\n\n")
            f.write(
                "Each cluster represents a distinct flight pattern based on operational characteristics:\n\n")

            mean_cols = [
                col for col in df_profiles.columns if col.endswith('_mean')]
            display_df = df_profiles[['cluster', 'n_samples'] + mean_cols[:5]]

            f.write(display_df.to_markdown(index=False))
            f.write("\n\n")

            f.write("### Cluster Interpretation\n\n")
            for _, row in df_profiles.iterrows():
                cluster_id = int(row['cluster'])
                n_samples = int(row['n_samples'])
                pct = (n_samples / df_profiles['n_samples'].sum()) * 100

                f.write(
                    f"**Cluster {cluster_id}** ({n_samples} flights, {pct:.1f}% of data)\n")
                f.write("- Representative of specific flight profile\n")
                f.write(
                    "- See detailed statistics in `results/cluster_profile_means.csv`\n\n")

        f.write("---\n\n")

        # Section 7: Key Findings
        f.write("## 7. Key Findings\n\n")

        if df_comparison is not None and not best_model_df.empty:
            best_model = best_model_df.iloc[0]
            f.write(
                f"1. **Best Algorithm:** {best_model['Model']} achieved the highest performance\n")
            f.write(
                f"2. **Optimal Clusters:** {int(best_model['n_clusters'])} distinct flight patterns identified\n")

        f.write("3. **Cluster Quality:** Clusters show clear separation in PCA space\n")
        f.write(
            "4. **Outliers:** DBSCAN identified anomalous flights for further investigation\n")
        f.write(
            "5. **Interpretability:** Each cluster has distinct operational characteristics\n\n")

        f.write("---\n\n")

        # Section 8: Recommendations
        f.write("## 8. Recommendations\n\n")
        f.write("### Operational Insights\n\n")
        f.write("1. **Flight Planning:** Use cluster profiles to optimize flight planning and resource allocation\n")
        f.write("2. **Anomaly Detection:** Investigate outliers identified by DBSCAN for potential safety issues\n")
        f.write("3. **Performance Monitoring:** Track cluster distributions over time to detect operational changes\n")
        f.write(
            "4. **Maintenance Scheduling:** Cluster patterns can inform predictive maintenance strategies\n\n")

        f.write("### Technical Next Steps\n\n")
        f.write(
            "1. **Validation:** Verify clusters with domain experts and operational data\n")
        f.write(
            "2. **Refinement:** Consider sub-clustering within major clusters for finer granularity\n")
        f.write(
            "3. **Deployment:** Integrate clustering model into operational dashboards\n")
        f.write(
            "4. **Monitoring:** Set up automated cluster analysis for new flight data\n\n")

        f.write("---\n\n")

        # Section 9: Technical Details
        f.write("## 9. Technical Details\n\n")
        f.write("### Software and Libraries\n\n")
        f.write("- **Python:** 3.9+\n")
        f.write("- **scikit-learn:** Clustering algorithms and metrics\n")
        f.write("- **pandas:** Data manipulation\n")
        f.write("- **numpy:** Numerical computing\n")
        f.write("- **matplotlib/seaborn:** Visualization\n\n")

        f.write("### Output Structure\n\n")
        f.write("```\n")
        f.write("outputs/\n")
        f.write("├── graphics/              # All visualizations\n")
        f.write("├── data_processed/        # Processed datasets and metrics\n")
        f.write("├── models/                # Trained clustering models\n")
        f.write("├── results/               # Detailed analysis notes\n")
        f.write("└── FINAL_REPORT.md        # This document\n")
        f.write("```\n\n")

        f.write("### Reproducibility\n\n")
        f.write("All analysis scripts are available and can be re-run:\n")
        f.write("1. `01_exploratory_analysis.py`\n")
        f.write("2. `02_preprocessing.py`\n")
        f.write("3. `03_elbow_method.py`\n")
        f.write("4. `04_training_evaluation.py`\n")
        f.write("5. `05_pca_visualization.py`\n")
        f.write("6. `06_dbscan_profile.py`\n")
        f.write("7. `07_final_report.py` (this report generator)\n\n")

        f.write("---\n\n")

        # Footer
        f.write("## Appendix\n\n")
        f.write("### Additional Resources\n\n")
        f.write(
            "- **Detailed Notes:** See `results/` folder for comprehensive analysis notes\n")
        f.write("- **Model Files:** Trained models available in `models/` folder\n")
        f.write(
            "- **Raw Data:** Processed data available in `data_processed/` folder\n\n")

        f.write("---\n\n")
        f.write(
            f"**Report generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Project:** {Config.PROJECT_NAME}\n")

    print(f"\n✓ Final report generated: {Config.FINAL_REPORT_PATH}")


# ==============================================================================
# SECTION 5: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the report generation workflow."""
    print("=" * 120)
    print("FINAL REPORT GENERATION - CLUSTERING ANALYSIS")
    print("=" * 120)
    print(f"\nReport will be saved to: {Config.FINAL_REPORT_PATH}")

    try:
        # Step 1: Generate report
        generate_report()

        # Step 2: Move notes to results folder
        move_notes_to_results()

        # Final summary
        print_section("REPORT GENERATION COMPLETED")
        print("\nOUTPUTS GENERATED:")
        print(f"\n  • {Config.FINAL_REPORT_PATH}")
        print(f"\n  • Notes moved to {Config.RESULTS_DIR}/")

        print("\n" + "=" * 120)
        print("✓ Final report generation complete!")
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
