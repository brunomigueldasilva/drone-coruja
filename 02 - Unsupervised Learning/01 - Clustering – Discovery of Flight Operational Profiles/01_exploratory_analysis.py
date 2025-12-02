#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
EXPLORATORY DATA ANALYSIS - FLIGHT TELEMETRY DATA
==============================================================================

Purpose: Comprehensive EDA focused on preparing flight telemetry data for unsupervised learning algorithms.

This script:
1. Loads flight telemetry CSV data
2. Performs data validation and quality checks
3. Analyzes skewness and outlier detection
4. Creates comprehensive visualizations (histograms, boxplots, pairplot, correlation)
5. Generates schema summary and analytical notes
6. Provides recommendations for preprocessing and clustering

Expected Columns (Numeric):
- duracao_voo_min (flight duration in minutes)
- distancia_percorrida_km (distance traveled in km)
- altitude_maxima_m (maximum altitude in meters)
- velocidade_media_kmh (average speed in km/h)
- consumo_combustivel_litros (fuel consumption in liters)
- variacao_vertical_total_m (total vertical variation in meters)

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Configuration Constants
SEED = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore')


class Config:
    """Exploratory analysis configuration parameters."""
    # Directories
    INPUT_DIR = Path('inputs')
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'
    DATA_PROCESSED_DIR = OUTPUT_DIR / 'data_processed'
    MODELS_DIR = OUTPUT_DIR / 'models'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'

    # Input file
    DATASET_PATH = INPUT_DIR / 'voos_telemetria_completa.csv'

    # Expected numeric columns for flight telemetry
    EXPECTED_COLUMNS = [
        'duracao_voo_min',
        'distancia_percorrida_km',
        'altitude_maxima_m',
        'velocidade_media_kmh',
        'consumo_combustivel_litros',
        'variacao_vertical_total_m'
    ]

    # Visualization settings
    DPI = 300
    FIGSIZE_LARGE = (16, 12)
    FIGSIZE_MEDIUM = (14, 10)
    FIGSIZE_SMALL = (12, 8)


# Visualization configuration
plt.style.use('default')
sns.set_palette("Set2")


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def save_plot(filename: str, dpi: int = 300) -> None:
    """
    Save current plot to graphics folder.

    Args:
        filename: Name of the output file
        dpi: Resolution in dots per inch
    """
    filepath = Config.GRAPHICS_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
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
# SECTION 3: DATA LOADING AND VALIDATION
# ==============================================================================


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV file and perform initial inspection.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.absolute()}")

    df = pd.read_csv(path, encoding='utf-8')
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset Information:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())

    return df


def validate_columns(df: pd.DataFrame, expected_cols: List[str]) -> List[str]:
    """
    Validate presence of expected columns and identify numeric features.

    Args:
        df: DataFrame to validate
        expected_cols: List of expected column names

    Returns:
        List of numeric columns to use for analysis

    Raises:
        ValueError: If no numeric columns found
    """
    missing_cols = [col for col in expected_cols if col not in df.columns]

    if missing_cols:
        print(f"⚠ WARNING: Missing expected columns: {missing_cols}")
        print("  Analysis will continue with available numeric columns.")
    else:
        print("✓ All expected columns present")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n✓ Found {len(numeric_cols)} numeric columns: {numeric_cols}")

    if not numeric_cols:
        raise ValueError("No numeric columns found in dataset")

    return numeric_cols


# ==============================================================================
# SECTION 4: STATISTICAL ANALYSIS
# ==============================================================================


def analyze_skewness(
        df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
    """
    Compute and rank features by skewness.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names

    Returns:
        Dictionary with feature: skewness pairs
    """
    print_section("SKEWNESS ANALYSIS", "-")

    skewness_dict = {}

    for col in numeric_cols:
        skew_value = stats.skew(df[col].dropna())
        skewness_dict[col] = skew_value

    sorted_skewness = sorted(
        skewness_dict.items(),
        key=lambda x: abs(
            x[1]),
        reverse=True)

    print("Skewness Ranking (most to least skewed):")
    print(f"{'Feature':<45} {'Skewness':>15} {'Interpretation':<20}")
    print("-" * 120)

    for feature, skew_val in sorted_skewness:
        if abs(skew_val) > 1.0:
            interpretation = "Highly skewed"
        elif abs(skew_val) > 0.5:
            interpretation = "Moderately skewed"
        else:
            interpretation = "Fairly symmetric"

        print(f"{feature:<45} {skew_val:>15.4f} {interpretation:<20}")

    return skewness_dict


def detect_outliers_iqr(
        df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
    """
    Detect outliers using IQR method and calculate outlier rates.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric columns to check

    Returns:
        Dictionary with feature: outlier_percentage pairs
    """
    print_section("OUTLIER DETECTION (IQR METHOD)", "-")

    outlier_dict = {}

    print(
        f"{
            'Feature':<45} {
            'Outliers':>12} {
                '% of Data':>12} {
                    'Lower Bound':>15} {
                        'Upper Bound':>15}")
    print("-" * 120)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_pct = (outliers / len(df)) * 100
        outlier_dict[col] = outlier_pct

        print(
            f"{
                col:<45} {
                outliers:>12} {
                outlier_pct:>11.2f}% {
                    lower_bound:>15.2f} {
                        upper_bound:>15.2f}")

    return outlier_dict


# ==============================================================================
# SECTION 5: VISUALIZATION FUNCTIONS
# ==============================================================================


def plot_histograms(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """
    Create histogram plots for all numeric features.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
    """
    print_section("GENERATING HISTOGRAMS", "-")

    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=Config.FIGSIZE_LARGE)
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(
            df[col].dropna(),
            bins=30,
            color='steelblue',
            edgecolor='black',
            alpha=0.7)
        axes[idx].set_title(f'{col}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Value', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3)

    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(
        'Distribution of Numeric Features',
        fontsize=14,
        fontweight='bold',
        y=1.00)
    save_plot('histograms.png', dpi=Config.DPI)


def plot_boxplots(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """
    Create boxplot visualization for all numeric features.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
    """
    print_section("GENERATING BOXPLOTS", "-")

    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=Config.FIGSIZE_LARGE)
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(numeric_cols):
        axes[idx].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', color='black'),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(color='black'),
                          capprops=dict(color='black'))
        axes[idx].set_title(f'{col}', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Value', fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3)

    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(
        'Boxplots - Outlier Detection',
        fontsize=14,
        fontweight='bold',
        y=1.00)
    save_plot('boxplots.png', dpi=Config.DPI)


def plot_pairplot(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """
    Create pairplot for exploring relationships between features.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
    """
    print_section("GENERATING PAIRPLOT", "-")

    pairplot_fig = sns.pairplot(
        df[numeric_cols], diag_kind='kde', plot_kws={
            'alpha': 0.6, 's': 30}, diag_kws={
            'linewidth': 2})
    pairplot_fig.fig.suptitle(
        'Pairplot - Feature Relationships',
        fontsize=14,
        fontweight='bold',
        y=1.00)

    pairplot_fig.savefig(
        Config.GRAPHICS_DIR /
        'pairplot.png',
        dpi=Config.DPI,
        bbox_inches='tight')
    pairplot_fig.savefig(
        Config.GRAPHICS_DIR /
        'pairplot.pdf',
        bbox_inches='tight')
    print("✓ Saved: pairplot.png")
    print("✓ Saved: pairplot.pdf")
    plt.close()


def plot_correlation_heatmap(
        df: pd.DataFrame,
        numeric_cols: List[str]) -> None:
    """
    Create correlation matrix heatmap.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
    """
    print_section("GENERATING CORRELATION HEATMAP", "-")

    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix', fontsize=13, fontweight='bold', pad=15)

    save_plot('correlation_heatmap.png', dpi=Config.DPI)


# ==============================================================================
# SECTION 6: EXPORT AND REPORTING FUNCTIONS
# ==============================================================================


def create_schema_summary(df: pd.DataFrame,
                          numeric_cols: List[str],
                          skewness_dict: Dict[str,
                                              float],
                          outlier_dict: Dict[str,
                                             float]) -> None:
    """
    Create and save schema summary CSV with feature statistics.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
        skewness_dict: Dictionary with skewness values
        outlier_dict: Dictionary with outlier percentages
    """
    print_section("CREATING SCHEMA SUMMARY", "-")

    schema_data = []

    for col in numeric_cols:
        schema_data.append({
            'feature': col,
            'dtype': str(df[col].dtype),
            'non_null_count': df[col].notna().sum(),
            'null_count': df[col].isna().sum(),
            'null_percentage': (df[col].isna().sum() / len(df)) * 100,
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'q25': df[col].quantile(0.25),
            'median': df[col].median(),
            'q75': df[col].quantile(0.75),
            'max': df[col].max(),
            'skewness': skewness_dict.get(col, np.nan),
            'outlier_percentage': outlier_dict.get(col, np.nan)
        })

    schema_df = pd.DataFrame(schema_data)
    output_path = Config.DATA_PROCESSED_DIR / 'schema_summary.csv'
    schema_df.to_csv(output_path, index=False, encoding='utf-8')
    print("✓ Saved: schema_summary.csv")


def create_eda_notes(df: pd.DataFrame,
                     numeric_cols: List[str],
                     skewness_dict: Dict[str,
                                         float],
                     outlier_dict: Dict[str,
                                        float]) -> None:
    """
    Generate detailed EDA notes in Markdown format.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
        skewness_dict: Dictionary with skewness values
        outlier_dict: Dictionary with outlier percentages
    """
    print_section("GENERATING EDA NOTES", "-")

    output_path = Config.OUTPUT_DIR / 'eda_notes.md'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Exploratory Data Analysis Notes\n\n")
        f.write("## 1. Skewness Analysis\n\n")
        f.write(
            "**Purpose:** Identify features requiring transformation before clustering.\n\n")

        highly_skewed = [
            col for col,
            skew in skewness_dict.items() if abs(skew) > 1.0]
        moderately_skewed = [
            col for col,
            skew in skewness_dict.items() if 0.5 < abs(skew) <= 1.0]

        if highly_skewed:
            f.write("**Highly skewed features (|skewness| > 1.0):**\n")
            for col in highly_skewed:
                f.write(f"- `{col}`: {skewness_dict[col]:.4f}\n")
            f.write("\n")

        if moderately_skewed:
            f.write("**Moderately skewed features (0.5 < |skewness| ≤ 1.0):**\n")
            for col in moderately_skewed:
                f.write(f"- `{col}`: {skewness_dict[col]:.4f}\n")
            f.write("\n")

        f.write("**Recommendations:**\n")
        if highly_skewed or moderately_skewed:
            f.write(
                "- Consider log transformation or Box-Cox transformation for highly skewed features\n")
            f.write("- Apply StandardScaler or RobustScaler after transformation\n")
            f.write(
                "- Alternatively, use PowerTransformer (Yeo-Johnson) for zero/negative values\n\n")
        else:
            f.write("- Features show relatively symmetric distributions\n")
            f.write("- Standard scaling should be sufficient\n\n")

        f.write("## 2. Why Feature Scaling is MANDATORY\n\n")
        f.write(
            "**Distance-based algorithms (K-Means, Hierarchical, DBSCAN) are scale-sensitive:**\n\n")

        scales = {col: df[col].std() for col in numeric_cols}
        max_scale = max(scales.values())
        min_scale = min(scales.values())
        scale_ratio = max_scale / min_scale if min_scale > 0 else np.inf

        f.write("**Current feature scales (standard deviation):**\n")
        for col, scale in sorted(
                scales.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- `{col}`: {scale:.2f}\n")
        f.write(f"\n**Scale ratio (max/min):** {scale_ratio:.2f}x\n\n")

        f.write("**Problem without scaling:**\n")
        f.write("- Features with larger scales dominate distance calculations\n")
        f.write("- Clusters will form primarily based on high-scale features\n")
        f.write(
            "- This violates the assumption that all features are equally important\n\n")

        f.write("**Mandatory preprocessing:**\n")
        f.write(
            "- **StandardScaler**: Recommended for features with roughly normal distributions\n")
        f.write(
            "- **RobustScaler**: Better for features with outliers (uses median and IQR)\n")
        f.write(
            "- **MinMaxScaler**: Alternative if bounded range [0,1] is preferred\n")
        f.write("- Apply scaling AFTER train-test split to avoid data leakage\n\n")

        f.write("## 3. Outlier Detection Results\n\n")
        f.write(
            "**Method:** IQR (Interquartile Range) - values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR\n\n")

        high_outliers = [
            col for col,
            rate in outlier_dict.items() if rate > 5.0]
        moderate_outliers = [
            col for col,
            rate in outlier_dict.items() if 1.0 < rate <= 5.0]

        if high_outliers:
            f.write("**Features with high outlier rates (> 5%):**\n")
            for col in high_outliers:
                f.write(f"- `{col}`: {outlier_dict[col]:.2f}%\n")
            f.write("\n")

        if moderate_outliers:
            f.write("**Features with moderate outlier rates (1-5%):**\n")
            for col in moderate_outliers:
                f.write(f"- `{col}`: {outlier_dict[col]:.2f}%\n")
            f.write("\n")

        f.write("**Implications for clustering:**\n")
        f.write(
            "- Outliers can form singleton clusters or distort centroids in K-Means\n")
        f.write("- DBSCAN is more robust to outliers (can classify them as noise)\n")
        f.write(
            "- Consider outlier removal or capping for K-Means if outlier rate > 5%\n")
        f.write(
            "- RobustScaler is preferable to StandardScaler when outliers are present\n\n")

        f.write("## 4. Early Signs of Cluster Structure\n\n")
        f.write("**From pairplot analysis:**\n")
        f.write("- Examine scatter plots for visible groupings or separations\n")
        f.write(
            "- Look for non-linear relationships indicating complex cluster shapes\n")
        f.write(
            "- Diagonal KDE plots show distribution modality (multiple peaks suggest clusters)\n\n")

        f.write("**From correlation analysis:**\n")
        corr_matrix = df[numeric_cols].corr()
        strong_corr_count = sum(
            1 for i in range(len(corr_matrix.columns))
            for j in range(i + 1, len(corr_matrix.columns))
            if abs(corr_matrix.iloc[i, j]) > 0.7
        )

        f.write(
            f"- Found {strong_corr_count} strong correlations (|r| > 0.7)\n")
        f.write("- Highly correlated features may indicate redundancy\n")
        f.write("- Consider PCA or feature selection to reduce dimensionality\n")
        f.write("- Lower-dimensional spaces often reveal clearer cluster structure\n\n")

        f.write("## 5. Recommended Next Steps\n\n")
        f.write("1. **Preprocessing pipeline:**\n")
        f.write("   - Handle missing values (if any)\n")
        f.write("   - Apply transformations for skewed features\n")
        f.write("   - Scale all features using StandardScaler or RobustScaler\n\n")

        f.write("2. **Dimensionality reduction:**\n")
        f.write("   - Apply PCA to reduce to 2-3 components for visualization\n")
        f.write("   - Check explained variance ratio\n")
        f.write("   - Visualize data in reduced space\n\n")

        f.write("3. **Optimal cluster determination:**\n")
        f.write("   - Elbow method (within-cluster sum of squares)\n")
        f.write("   - Silhouette score analysis\n")
        f.write("   - Calinski-Harabasz index\n")
        f.write("   - Dendrogram for hierarchical clustering\n\n")

        f.write("4. **Algorithm selection:**\n")
        f.write("   - K-Means: Good starting point, assumes spherical clusters\n")
        f.write("   - DBSCAN: Better for arbitrary shapes and outlier handling\n")
        f.write(
            "   - Hierarchical: Useful for understanding nested cluster structure\n")
        f.write(
            "   - Gaussian Mixture Models: Probabilistic clustering with soft assignments\n\n")

        f.write("---\n\n")
        f.write("**End of EDA Notes**\n")

    print("✓ Saved: eda_notes.md")


def print_final_summary() -> None:
    """Print final analysis summary with generated artifacts."""
    print_section("ANALYSIS COMPLETE - SUMMARY OF ARTIFACTS")

    print("\nGRAPHICS:")
    graphics_files = list(Config.GRAPHICS_DIR.glob('*'))
    for file in sorted(graphics_files):
        print(f"  • {file}")

    print("\nDATA PROCESSED:")
    data_files = list(Config.DATA_PROCESSED_DIR.glob('*'))
    for file in sorted(data_files):
        print(f"  • {file}")

    print("\nEDA NOTES:")
    if (Config.OUTPUT_DIR / 'eda_notes.md').exists():
        print(f"  • {Config.OUTPUT_DIR / 'eda_notes.md'}")

    print("\n" + "=" * 120)
    print("EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 120)


# ==============================================================================
# SECTION 7: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the entire exploratory analysis."""
    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    Config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("EXPLORATORY DATA ANALYSIS - FLIGHT TELEMETRY DATA")
    print("=" * 120)
    print(f"\nRandom seed: {SEED}")
    print(f"Dataset path: {Config.DATASET_PATH.absolute()}")

    print_section("CREATING OUTPUT DIRECTORIES")
    print(f"✓ Output directory: {Config.OUTPUT_DIR.absolute()}")
    print(f"✓ Graphics directory: {Config.GRAPHICS_DIR.absolute()}")
    print(
        f"✓ Data processed directory: {
            Config.DATA_PROCESSED_DIR.absolute()}")
    print(f"✓ Models directory: {Config.MODELS_DIR.absolute()}")
    print(f"✓ Predictions directory: {Config.PREDICTIONS_DIR.absolute()}")

    try:
        # 1. Load and validate data
        print_section("STEP 1: LOAD AND INSPECT DATASET")
        df = load_dataset(Config.DATASET_PATH)

        print_section("STEP 2: VALIDATE COLUMNS")
        numeric_cols = validate_columns(df, Config.EXPECTED_COLUMNS)

        # 2. Statistical analysis
        print_section("STEP 3: STATISTICAL ANALYSIS")
        skewness_dict = analyze_skewness(df, numeric_cols)
        outlier_dict = detect_outliers_iqr(df, numeric_cols)

        # 3. Generate visualizations
        print_section("STEP 4: GENERATE VISUALIZATIONS")
        plot_histograms(df, numeric_cols)
        plot_boxplots(df, numeric_cols)
        plot_pairplot(df, numeric_cols)
        plot_correlation_heatmap(df, numeric_cols)

        # 4. Export results
        print_section("STEP 5: EXPORT RESULTS")
        create_schema_summary(df, numeric_cols, skewness_dict, outlier_dict)
        create_eda_notes(df, numeric_cols, skewness_dict, outlier_dict)

        # 5. Final summary
        print_final_summary()

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
