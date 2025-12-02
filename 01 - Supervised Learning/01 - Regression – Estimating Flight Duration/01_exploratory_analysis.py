#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLIGHT TELEMETRY - EXPLORATORY DATA ANALYSIS
==============================================================================

Purpose: Understand and visualize flight duration patterns (telemetry analysis)

This script:
1. Loads raw flight telemetry data with planned distance, cargo, and altitude
2. Validates data quality and column structure
3. Analyzes target variable (duracao_voo_min) distribution and outliers
4. Creates comprehensive visualizations including histograms and boxplots
5. Examines relationships between numeric predictors and target variable
6. Analyzes categorical weather conditions impact on flight duration
7. Calculates correlation matrices and generates insights for modeling

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Exploratory analysis configuration parameters."""
    # Directories
    INPUT_DIR = Path('inputs')
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'
    RESULTS_DIR = OUTPUT_DIR / 'results'

    # Input files
    INPUT_FILE = INPUT_DIR / 'voos_telemetria.csv'

    # Visualization settings
    DPI = 300
    FIGSIZE = (10, 6)
    FONT_SIZE = 10

    # Expected columns
    NUMERIC_COLUMNS = [
        'distancia_planeada',
        'carga_util_kg',
        'altitude_media_m']
    CATEGORICAL_COLUMNS = ['condicao_meteo']
    TARGET_COLUMN = 'duracao_voo_min'


# Visualization configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = Config.FIGSIZE
plt.rcParams['font.size'] = Config.FONT_SIZE


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def save_plot(filename: str, dpi: int = 300) -> None:
    """Save current plot to outputs folder."""
    filepath = Path(Config.GRAPHICS_DIR) / f"{filename}.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 78)
    print(title)
    print(char * 78)


# ==============================================================================
# SECTION 3: DATA LOADING AND VALIDATION
# ==============================================================================


def load_and_validate_data() -> pd.DataFrame:
    """
    Load flight telemetry data and validate structure.

    Returns:
        pd.DataFrame: Loaded and validated dataset
    """
    print_section("SECTION 1: DATA LOADING AND VALIDATION", "-")

    # Create directories if they don't exist
    Config.INPUT_DIR.mkdir(exist_ok=True)
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    Config.GRAPHICS_DIR.mkdir(exist_ok=True)
    Config.RESULTS_DIR.mkdir(exist_ok=True)
    print(
        f"✓ Directories verified/created: {
            Config.INPUT_DIR}, {
            Config.GRAPHICS_DIR}, " f"{
                Config.RESULTS_DIR}")
    print()

    # Load dataset
    df = pd.read_csv(Config.INPUT_FILE)
    print(
        f"✓ Dataset loaded successfully: {
            df.shape[0]} rows × {
            df.shape[1]} columns")
    print()

    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())
    print()

    # Define expected columns
    all_columns = (Config.NUMERIC_COLUMNS +
                   Config.CATEGORICAL_COLUMNS +
                   [Config.TARGET_COLUMN])

    # Validate column presence
    print("Column validation:")
    for col in all_columns:
        status = "[OK]" if col in df.columns else "[MISSING]"
        print(f"  {status} {col}")
    print()

    # Check data types
    print("Data types (dtypes):")
    print(df.dtypes)
    print()

    # Check for missing values
    print("Missing values per column:")
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing_count,
        'Missing_Percentage': missing_pct
    })
    print(missing_df)
    print()

    # Validate categorical variable values
    if 'condicao_meteo' in df.columns:
        print("Unique values in 'condicao_meteo':")
        print(df['condicao_meteo'].value_counts())
        print()

        # Check if values match expected categories
        expected_categories = {'Bom', 'Moderado', 'Adverso'}
        found_categories = set(df['condicao_meteo'].unique())

        if found_categories == expected_categories:
            print(
                "✓ Categorical values are as expected: {Bom, Moderado, Adverso}")
        else:
            print(f"⚠️  Warning - Found categories: {found_categories}")
            print(f"⚠️  Warning - Expected categories: {expected_categories}")
        print()

    return df


# ==============================================================================
# SECTION 4: TARGET VARIABLE ANALYSIS
# ==============================================================================


def analyze_target_variable(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze target variable distribution, outliers, and statistics.

    Args:
        df: DataFrame with flight data

    Returns:
        Dictionary with target statistics
    """
    print_section("SECTION 2: TARGET VARIABLE ANALYSIS (duracao_voo_min)", "-")

    target = df[Config.TARGET_COLUMN]

    # Descriptive statistics
    print("Descriptive statistics of target variable:")
    stats_dict = {
        'mean': target.mean(),
        'median': target.median(),
        'std': target.std(),
        'min': target.min(),
        'max': target.max(),
        'q1': target.quantile(0.25),
        'q2': target.quantile(0.50),
        'q3': target.quantile(0.75),
    }

    for key, value in stats_dict.items():
        print(f"  {key.upper():20s}: {value:.2f}")
    print()

    # Skewness calculation
    skewness = target.skew()
    print(f"Skewness: {skewness:.4f}")

    # Interpret skewness
    if abs(skewness) < 0.5:
        skew_interpretation = "approximately symmetric"
    elif skewness > 0:
        skew_interpretation = "right-skewed (long tail to the right)"
    else:
        skew_interpretation = "left-skewed (long tail to the left)"

    print(f"Interpretation: Distribution is {skew_interpretation}")
    print()

    # Outlier detection using IQR method
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1

    # Calculate outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify outliers
    outliers_mask = (target < lower_bound) | (target > upper_bound)
    num_outliers = outliers_mask.sum()
    pct_outliers = (num_outliers / len(target)) * 100

    print("Outlier detection (IQR method):")
    print(f"  IQR: {iqr:.2f}")
    print(f"  Lower Bound: {lower_bound:.2f}")
    print(f"  Upper Bound: {upper_bound:.2f}")
    print(f"  Number of Outliers: {num_outliers} ({pct_outliers:.2f}%)")
    print()

    # Mark outliers in dataframe
    df['outlier_target'] = outliers_mask

    # Store statistics
    stats_dict.update({
        'skewness': skewness,
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'num_outliers': num_outliers,
        'pct_outliers': pct_outliers
    })

    return stats_dict


def create_target_visualizations(df: pd.DataFrame) -> None:
    """
    Create visualizations for target variable.

    Args:
        df: DataFrame with flight data
    """
    print("Generating target variable visualizations...")

    target = df[Config.TARGET_COLUMN]

    # Histogram with Kernel Density Estimation
    fig, ax = plt.subplots(figsize=Config.FIGSIZE)
    ax.hist(
        target,
        bins=30,
        density=True,
        alpha=0.7,
        color='skyblue',
        edgecolor='black')

    # Add KDE curve
    kde = gaussian_kde(target)
    x_range = np.linspace(target.min(), target.max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Kernel Density')

    # Add mean and median lines
    ax.axvline(target.mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {target.mean():.2f}')
    ax.axvline(target.median(), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {target.median():.2f}')

    ax.set_xlabel('Flight Duration (min)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(
        'Flight Duration Distribution',
        fontsize=14,
        fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_plot("target_hist", dpi=Config.DPI)

    # Boxplot
    fig, ax = plt.subplots(figsize=Config.FIGSIZE)
    bp = ax.boxplot(
        [target],
        labels=['Flight Duration'],
        vert=True,
        patch_artist=True,
        widths=0.6)

    # Color boxplot
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Customize median line
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)

    ax.set_ylabel('Flight Duration (min)', fontsize=12)
    ax.set_title('Flight Duration Boxplot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    save_plot("target_boxplot", dpi=Config.DPI)
    print()


# ==============================================================================
# SECTION 5: NUMERIC PREDICTORS ANALYSIS
# ==============================================================================


def analyze_numeric_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze numeric predictors and their relationship with target.

    Args:
        df: DataFrame with flight data

    Returns:
        Correlation matrix
    """
    print_section("SECTION 3: NUMERIC PREDICTORS ANALYSIS", "-")

    target = df[Config.TARGET_COLUMN]

    # Create scatter plots
    print("Generating scatter plots for numeric variables...")

    for var in Config.NUMERIC_COLUMNS:
        if var in df.columns:
            fig, ax = plt.subplots(figsize=Config.FIGSIZE)

            # Scatter plot with colors for outliers
            ax.scatter(df[var], target,
                       c=df['outlier_target'].map({True: 'red', False: 'blue'}),
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

            # Add trend line
            z = np.polyfit(df[var], target, 1)
            p = np.poly1d(z)
            ax.plot(df[var], p(df[var]), "g--",
                    linewidth=2, label='Linear Trend')

            # Calculate Pearson correlation
            corr = df[var].corr(target)

            # Format variable name for display
            var_display = var.replace('_', ' ').title()

            ax.set_xlabel(var_display, fontsize=12)
            ax.set_ylabel('Flight Duration (min)', fontsize=12)
            ax.set_title(f'Relationship: {var_display} vs Flight Duration\n'
                         f'(Pearson Correlation: r = {corr:.3f})',
                         fontsize=14, fontweight='bold')
            ax.legend(['Linear Trend', 'Normal Observations', 'Target Outliers'])
            ax.grid(True, alpha=0.3)

            save_plot(f"scatter_numeric_vs_target_{var}", dpi=Config.DPI)
            print(f"  (r = {corr:.3f})")

    print()

    # Calculate Pearson correlations
    print("Calculating Pearson correlations...")

    # Select numeric variables (including target)
    vars_for_corr = Config.NUMERIC_COLUMNS + [Config.TARGET_COLUMN]
    df_numeric = df[vars_for_corr]

    # Correlation matrix
    corr_matrix = df_numeric.corr()

    print("Correlation Matrix (Pearson):")
    print(corr_matrix)
    print()

    # Correlations with target specifically
    print(f"Correlations with target ({Config.TARGET_COLUMN}):")
    corr_with_target = corr_matrix[Config.TARGET_COLUMN].sort_values(
        ascending=False)
    for var, corr_val in corr_with_target.items():
        if var != Config.TARGET_COLUMN:
            interpretation = ""
            if abs(corr_val) > 0.7:
                interpretation = "(strong)"
            elif abs(corr_val) > 0.4:
                interpretation = "(moderate)"
            elif abs(corr_val) > 0.2:
                interpretation = "(weak)"
            else:
                interpretation = "(very weak)"

            print(f"  {var:25s}: {corr_val:+.4f} {interpretation}")
    print()

    # Create correlation heatmap
    print("Generating correlation heatmap...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)

    ax.set_title('Correlation Matrix (Pearson) - Numeric Variables',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    save_plot("heatmap_correlations", dpi=Config.DPI)
    print()

    return corr_matrix


# ==============================================================================
# SECTION 6: CATEGORICAL ANALYSIS
# ==============================================================================


def analyze_weather_conditions(df: pd.DataFrame) -> None:
    """
    Analyze weather conditions impact on target variable.

    Args:
        df: DataFrame with flight data
    """
    print("Generating boxplot for weather condition vs target...")

    if 'condicao_meteo' in df.columns:
        fig, ax = plt.subplots(figsize=Config.FIGSIZE)

        # Order categories logically
        category_order = ['Bom', 'Moderado', 'Adverso']
        existing_order = [
            cat for cat in category_order if cat in df['condicao_meteo'].unique()]

        bp = ax.boxplot([df[df['condicao_meteo'] == cat][Config.TARGET_COLUMN].values
                         for cat in existing_order],
                        labels=existing_order,
                        patch_artist=True,
                        widths=0.6)

        # Color boxplots
        colors = ['lightgreen', 'yellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(existing_order)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Customize median lines
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)

        ax.set_xlabel('Weather Condition', fontsize=12)
        ax.set_ylabel('Flight Duration (min)', fontsize=12)
        ax.set_title(
            'Flight Duration by Weather Condition',
            fontsize=14,
            fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add sample size annotations
        for i, cat in enumerate(existing_order, 1):
            subset = df[df['condicao_meteo'] == cat][Config.TARGET_COLUMN]
            ax.text(
                i,
                subset.median(),
                f'  n={
                    len(subset)}',
                verticalalignment='center',
                fontsize=9,
                color='darkred',
                fontweight='bold')

        save_plot("box_weather_vs_target", dpi=Config.DPI)

        # Print statistics by category
        print("Target statistics by weather condition:")
        for cat in existing_order:
            subset = df[df['condicao_meteo'] == cat][Config.TARGET_COLUMN]
            print(
                f"  {
                    cat:15s}: Mean = {
                    subset.mean():6.2f}, Median = {
                    subset.median():6.2f}, " f"SD = {
                    subset.std():5.2f}, n = {
                        len(subset)}")
        print()


# ==============================================================================
# SECTION 7: KEY LESSONS AND SUMMARY
# ==============================================================================


def save_target_statistics(stats: Dict[str, Any]) -> None:
    """
    Save target variable statistics to CSV file.

    Args:
        stats: Dictionary with target statistics
    """
    print("Saving target statistics to CSV...")

    stats_df = pd.DataFrame([
        {'Statistic': 'Mean', 'Value': stats['mean']},
        {'Statistic': 'Median', 'Value': stats['median']},
        {'Statistic': 'Std_Dev', 'Value': stats['std']},
        {'Statistic': 'Minimum', 'Value': stats['min']},
        {'Statistic': 'Maximum', 'Value': stats['max']},
        {'Statistic': 'Q1 (25%)', 'Value': stats['q1']},
        {'Statistic': 'Q2 (50%)', 'Value': stats['q2']},
        {'Statistic': 'Q3 (75%)', 'Value': stats['q3']},
        {'Statistic': 'IQR', 'Value': stats['iqr']},
        {'Statistic': 'Skewness', 'Value': stats['skewness']},
        {'Statistic': 'Lower_Bound', 'Value': stats['lower_bound']},
        {'Statistic': 'Upper_Bound', 'Value': stats['upper_bound']},
        {'Statistic': 'Number_Outliers', 'Value': stats['num_outliers']},
        {'Statistic': 'Percentage_Outliers', 'Value': stats['pct_outliers']},
    ])

    output_path = Config.RESULTS_DIR / 'target_statistics.csv'
    stats_df.to_csv(output_path, index=False)
    print(f"✓ Target statistics saved: {output_path}")
    print()


def create_eda_summary(
        df: pd.DataFrame, stats: Dict[str, Any], corr_matrix: pd.DataFrame) -> None:
    """
    Create comprehensive EDA summary report in Markdown format.

    Args:
        df: DataFrame with flight data
        stats: Dictionary with target statistics
        corr_matrix: Correlation matrix
    """
    print("Creating EDA summary report...")

    output_path = Config.RESULTS_DIR / 'eda_summary.md'

    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# Exploratory Data Analysis Summary\n")
        f.write("## Flight Telemetry Dataset\n\n")
        f.write(
            f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Section 1: Dataset Overview
        f.write("## 1. Dataset Overview\n\n")
        f.write(f"- **Total Records:** {len(df):,}\n")
        f.write(f"- **Total Features:** {len(df.columns)}\n")
        f.write(f"- **Numeric Features:** {len(Config.NUMERIC_COLUMNS)}\n")
        f.write(
            f"- **Categorical Features:** {len(Config.CATEGORICAL_COLUMNS)}\n")
        f.write(f"- **Target Variable:** {Config.TARGET_COLUMN}\n\n")

        # Section 2: Target Variable Statistics
        f.write("## 2. Target Variable Statistics (duracao_voo_min)\n\n")
        f.write("| Statistic | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Mean | {stats['mean']:.2f} min |\n")
        f.write(f"| Median | {stats['median']:.2f} min |\n")
        f.write(f"| Std Dev | {stats['std']:.2f} min |\n")
        f.write(f"| Minimum | {stats['min']:.2f} min |\n")
        f.write(f"| Maximum | {stats['max']:.2f} min |\n")
        f.write(f"| Q1 (25%) | {stats['q1']:.2f} min |\n")
        f.write(f"| Q2 (50%) | {stats['q2']:.2f} min |\n")
        f.write(f"| Q3 (75%) | {stats['q3']:.2f} min |\n")
        f.write(f"| IQR | {stats['iqr']:.2f} min |\n")
        f.write(f"| Skewness | {stats['skewness']:.4f} |\n\n")

        # Section 3: Outlier Detection
        f.write("## 3. Outlier Detection (IQR Method)\n\n")
        f.write(f"- **Lower Bound:** {stats['lower_bound']:.2f} min\n")
        f.write(f"- **Upper Bound:** {stats['upper_bound']:.2f} min\n")
        f.write(
            f"- **Outliers Detected:** {stats['num_outliers']} ({stats['pct_outliers']:.2f}%)\n\n")

        # Section 4: Distribution Analysis
        f.write("## 4. Distribution Analysis\n\n")
        if abs(stats['skewness']) < 0.5:
            skew_interp = "approximately symmetric"
            skew_rec = "Distribution is near normal. MAE and RMSE should behave similarly."
        elif stats['skewness'] > 0.5:
            skew_interp = "right-skewed (positive skew)"
            skew_rec = "Consider log transformation. RMSE will be more sensitive to high outliers."
        else:
            skew_interp = "left-skewed (negative skew)"
            skew_rec = "RMSE may penalize errors on low values more heavily."

        f.write(
            f"- **Skewness = {stats['skewness']:.4f}:** Distribution is {skew_interp}\n")
        f.write(f"- **Recommendation:** {skew_rec}\n\n")

        # Section 5: Correlation Analysis
        f.write("## 5. Correlation with Target Variable\n\n")
        f.write("| Feature | Pearson Correlation | Strength |\n")
        f.write("|---------|--------------------|-----------|\n")

        corr_with_target = corr_matrix[Config.TARGET_COLUMN].drop(
            Config.TARGET_COLUMN)
        for var, corr_val in corr_with_target.sort_values(
                ascending=False).items():
            if abs(corr_val) > 0.7:
                strength = "Strong"
            elif abs(corr_val) > 0.4:
                strength = "Moderate"
            elif abs(corr_val) > 0.2:
                strength = "Weak"
            else:
                strength = "Very Weak"
            f.write(f"| {var} | {corr_val:+.4f} | {strength} |\n")
        f.write("\n")

        # Section 6: Weather Condition Analysis
        f.write("## 6. Weather Condition Analysis\n\n")
        if 'condicao_meteo' in df.columns:
            f.write(
                "| Weather Condition | Count | Mean (min) | Median (min) | Std Dev (min) |\n")
            f.write(
                "|-------------------|-------|------------|--------------|---------------|\n")

            for cat in ['Bom', 'Moderado', 'Adverso']:
                if cat in df['condicao_meteo'].unique():
                    subset = df[df['condicao_meteo']
                                == cat][Config.TARGET_COLUMN]
                    f.write(f"| {cat} | {len(subset)} | {subset.mean():.2f} | "
                            f"{subset.median():.2f} | {subset.std():.2f} |\n")
            f.write("\n")

        # Section 7: Feature Scaling Requirements
        f.write("## 7. Feature Scaling Requirements\n\n")
        f.write("Numeric variables have very different scales:\n\n")
        f.write("| Feature | Min | Max | Range |\n")
        f.write("|---------|-----|-----|-------|\n")
        for var in Config.NUMERIC_COLUMNS:
            if var in df.columns:
                minv, maxv = df[var].min(), df[var].max()
                rangev = maxv - minv
                f.write(
                    f"| {var} | {
                        minv:.2f} | {
                        maxv:.2f} | {
                        rangev:.2f} |\n")
        f.write("\n")
        f.write(
            "**Implication:** Scale-sensitive algorithms (Linear Regression, KNN, Neural Networks) ")
        f.write("will require normalization/standardization.\n\n")

        # Section 8: Key Recommendations
        f.write("## 8. Key Recommendations for Modeling\n\n")
        f.write("### 8.1 Metric Selection\n")
        if stats['num_outliers'] > 0:
            f.write(
                f"- **{stats['num_outliers']} outliers detected** ({stats['pct_outliers']:.2f}%)\n")
            f.write("- **MAE** will be more robust to outliers\n")
            f.write("- **RMSE** will heavily penalize predictions at extremes\n")
            f.write(
                "- Consider robust models (Huber, RANSAC) or outlier treatment\n\n")
        else:
            f.write("- No significant outliers detected\n")
            f.write("- MAE and RMSE should perform similarly\n\n")

        f.write("### 8.2 Feature Engineering\n")
        f.write("- Apply feature scaling (StandardScaler or MinMaxScaler)\n")
        f.write("- Consider polynomial features or interactions\n")
        f.write("- Weather condition shows impact - consider one-hot encoding\n\n")

        f.write("### 8.3 Model Selection\n")
        strongest_corr_var = corr_with_target.abs().idxmax()
        strongest_corr_val = corr_with_target[strongest_corr_var]
        f.write(
            f"- Strongest predictor: **{strongest_corr_var}** (r={strongest_corr_val:.3f})\n")
        f.write("- Linear relationships exist but may not be perfect\n")
        f.write("- Test both linear (Linear Regression, Ridge, Lasso) and ")
        f.write("non-linear models (Random Forest, Gradient Boosting)\n\n")

        f.write("---\n\n")
        f.write(
            "*Analysis completed successfully. All visualizations saved to graphics folder.*\n")

    print(f"✓ EDA summary report saved: {output_path}")
    print()


def print_key_lessons(
        df: pd.DataFrame, stats: Dict[str, Any], corr_matrix: pd.DataFrame) -> None:
    """
    Print key lessons learned from exploratory analysis.

    Args:
        df: DataFrame with flight data
        stats: Dictionary with target statistics
        corr_matrix: Correlation matrix
    """
    print_section("KEY LESSONS FROM EDA")
    print()

    # Lesson 1: Feature scaling
    print("1. FEATURE SCALING REQUIREMENT")
    print("   " + "-" * 74)
    ranges = {
        var: (
            df[var].min(),
            df[var].max(),
            df[var].max() -
            df[var].min()) for var in Config.NUMERIC_COLUMNS}
    print("   Numeric variables have very different scales:")
    for var, (minv, maxv, rangev) in ranges.items():
        print(
            f"     • {
                var:25s}: [{
                minv:10.2f}, {
                maxv:10.2f}] (range: {
                    rangev:10.2f})")
    print()
    print("   IMPLICATION: Scale-sensitive algorithms (e.g., Linear Regression, KNN, Neural Networks)")
    print("   will require normalization/standardization before training.")
    print()

    # Lesson 2: Outliers and metrics
    print("2. OUTLIER RISK AND METRIC SELECTION")
    print("   " + "-" * 74)
    print(
        f"   Identified {
            stats['num_outliers']} outliers ({
            stats['pct_outliers']:.2f}% of data)")
    print(
        f"   in the target variable, with skewness = {
            stats['skewness']:.4f}.")
    print()
    print("   IMPLICATION:")
    print("     • MAE will be more robust, treating all errors proportionally.")
    print("     • RMSE will heavily penalize predictions at the extremes.")
    print("     • Models like Linear Regression may be influenced by outliers.")
    print("     • Consider robust models (e.g., Huber, RANSAC) or outlier treatment.")
    print()

    # Lesson 3: Non-linear relationships
    print("3. SUSPECTED NON-LINEARITY AND INTERACTIONS")
    print("   " + "-" * 74)
    corr_with_target = corr_matrix[Config.TARGET_COLUMN].drop(
        Config.TARGET_COLUMN)
    strongest_corr_var = corr_with_target.abs().idxmax()
    strongest_corr_val = corr_with_target[strongest_corr_var]

    print(
        f"   The most correlated variable with the target is '{strongest_corr_var}'")
    print(
        f"   (r = {
            strongest_corr_val:.3f}), but the relationship may not be perfectly linear.")
    print()
    print("   IMPLICATION:")
    print("     • Explore polynomial transformations or feature interactions.")
    print("     • Test non-linear models (e.g., Random Forest, Gradient Boosting).")
    print("     • The categorical variable 'condicao_meteo' shows distribution differences,")
    print("       suggesting it may have a moderating effect on numeric relationships.")
    print()


def print_final_summary() -> None:
    """Print final summary of the analysis."""
    print("=" * 78)
    print("All plots, tables, and notes have been saved to their respective directories:")
    print(f"   • Images: {Config.GRAPHICS_DIR}")
    print(f"   • Results: {Config.RESULTS_DIR}")
    print("=" * 78)
    print()
    print("✅ Script completed successfully!")


# ==============================================================================
# SECTION 8: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the entire exploratory analysis."""
    print("=" * 78)
    print("EXPLORATORY DATA ANALYSIS - FLIGHT TELEMETRY")
    print("=" * 78)
    print()

    try:
        # 1. Load and validate data
        df = load_and_validate_data()

        # 2. Analyze target variable
        stats = analyze_target_variable(df)

        # 3. Create target visualizations
        create_target_visualizations(df)

        # 4. Analyze numeric predictors
        corr_matrix = analyze_numeric_predictors(df)

        # 5. Analyze weather conditions
        analyze_weather_conditions(df)

        # 6. Save target statistics to CSV
        save_target_statistics(stats)

        # 7. Create EDA summary report
        create_eda_summary(df, stats, corr_matrix)

        # 8. Print key lessons
        print_key_lessons(df, stats, corr_matrix)

        # 9. Print final summary
        print_final_summary()

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
