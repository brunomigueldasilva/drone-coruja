#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLIGHT INCIDENT PREDICTION - EXPLORATORY DATA ANALYSIS
==============================================================================

Purpose: Perform EDA with simple graphics, feature engineering, and outlier removal.

This script:
1. Loads flight data CSV with incident reports
2. Performs data cleaning and validation
3. Engineers new features (interactions and encodings)
4. Optionally removes outliers using IQR method
5. Creates 6 comprehensive visualizations
6. Saves cleaned dataset and statistical summaries

Expected Columns (Portuguese):
- idade_aeronave_anos (numeric)
- horas_voo_desde_ultima_manutencao (numeric)
- previsao_turbulencia (categorical: Baixa, Média, Alta)
- tipo_missao (categorical: Vigilância, Carga, Transporte)
- experiencia_piloto_anos (numeric)
- incidente_reportado (target: 0 or 1)

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuration Constants
SEED = 42
np.random.seed(SEED)


class Config:
    """Exploratory analysis configuration parameters."""
    # Directories
    INPUT_DIR = Path('inputs')
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'
    TABLES_DIR = OUTPUT_DIR / 'results'

    # Input file
    DATASET_PATH = INPUT_DIR / 'voos_pre_voo.csv'

    # Expected columns
    EXPECTED_COLUMNS = [
        'idade_aeronave_anos',
        'horas_voo_desde_ultima_manutencao',
        'previsao_turbulencia',
        'tipo_missao',
        'experiencia_piloto_anos',
        'incidente_reportado'
    ]
    TARGET_COLUMN = 'incidente_reportado'

    # Feature engineering
    TURBULENCE_MAP = {'Baixa': 0, 'Média': 1, 'Alta': 2}
    REMOVE_OUTLIERS = False  # Set to True to enable outlier removal

    # Visualization settings
    DPI = 300
    FIGSIZE = (12, 6)
    # Green for no incident, Red for incident
    COLORS_TARGET = ['#90EE90', '#FF6B6B']


# Visualization configuration
plt.style.use('default')
sns.set_palette("Set2")


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def save_plot(filename: str, dpi: int = 300) -> None:
    """Save current plot to graphics folder."""
    filepath = Config.GRAPHICS_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def print_section(title: str, char: str = "-") -> None:
    """Print formatted section header."""
    print("\n" + char * 120)
    print(title)
    print(char * 120)


# ==============================================================================
# SECTION 3: DATA LOADING FUNCTIONS
# ==============================================================================


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV file.

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
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate that all expected columns are present.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If missing columns found
    """
    missing_columns = set(Config.EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    print("✓ All expected columns present")


# ==============================================================================
# SECTION 4: DATA CLEANING AND FEATURE ENGINEERING
# ==============================================================================


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create engineered features from raw data.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (DataFrame with new features, list of mission dummy column names)
    """
    print_section("DATA CLEANING AND FEATURE ENGINEERING")

    # 1. Convert turbulence to numeric
    df['turb_num'] = df['previsao_turbulencia'].map(Config.TURBULENCE_MAP)
    print("✓ Created turb_num: Baixa=0, Média=1, Alta=2")

    # 2. One-hot encoding for mission type
    mission_dummies = pd.get_dummies(df['tipo_missao'], prefix='missao')
    df = pd.concat([df, mission_dummies], axis=1)
    print(f"✓ Created mission dummies: {mission_dummies.columns.tolist()}")

    # 3. Create interaction features
    df['razao_manut_idade'] = (
        df['horas_voo_desde_ultima_manutencao'] / (df['idade_aeronave_anos'] + 0.001)
    )
    df['exp_x_turb'] = df['experiencia_piloto_anos'] * df['turb_num']
    print("✓ Created razao_manut_idade: maintenance hours / aircraft age")
    print("✓ Created exp_x_turb: pilot experience × turbulence")

    return df, mission_dummies.columns.tolist()


def remove_outliers_iqr(
        df: pd.DataFrame,
        numeric_cols: List[str]) -> pd.DataFrame:
    """
    Remove outliers using IQR method.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric columns to check for outliers

    Returns:
        DataFrame with outliers removed
    """
    print_section("OUTLIER REMOVAL (IQR Method)")

    initial_rows = len(df)
    outlier_mask = pd.Series([False] * len(df))

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = col_outliers.sum()
        outlier_mask |= col_outliers

        print(
            f"  {col}: {n_outliers} outliers ({
                n_outliers /
                initial_rows *
                100:.1f}%)")

    df = df[~outlier_mask].reset_index(drop=True)
    removed = initial_rows - len(df)
    print(
        f"\n✓ Removed {removed} rows with outliers ({
            removed /
            initial_rows *
            100:.1f}%)")
    print(f"✓ Final dataset: {len(df)} rows")

    return df


def save_cleaned_dataset(
        df: pd.DataFrame,
        mission_dummies_cols: List[str]) -> pd.DataFrame:
    """
    Save cleaned dataset with selected features.

    Args:
        df: Input DataFrame
        mission_dummies_cols: List of mission dummy column names

    Returns:
        DataFrame with selected features
    """
    cleaned_cols = [
        'idade_aeronave_anos',
        'horas_voo_desde_ultima_manutencao',
        'experiencia_piloto_anos',
        'turb_num'
    ] + mission_dummies_cols + [
        'razao_manut_idade',
        'exp_x_turb',
        'incidente_reportado'
    ]

    df_clean = df[cleaned_cols].copy()
    clean_path = Config.OUTPUT_DIR / 'voos_pre_voo_clean.csv'
    df_clean.to_csv(clean_path, index=False, encoding='utf-8')
    print(f"✓ Saved cleaned dataset: {clean_path}")
    print(f"  Final shape: {df_clean.shape}")

    return df_clean


# ==============================================================================
# SECTION 5: VISUALIZATION FUNCTIONS
# ==============================================================================


def plot_target_distribution(
        df: pd.DataFrame) -> Tuple[pd.Series, List[str], List[float]]:
    """
    Create target variable distribution plots.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (target counts, labels, percentages)
    """
    print_section("1. INCIDENT DISTRIBUTION")

    target_counts = df[Config.TARGET_COLUMN].value_counts().sort_index()
    target_labels = ['No Incident', 'With Incident']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot
    ax1.bar(target_labels, target_counts.values, color=Config.COLORS_TARGET,
            edgecolor='black', linewidth=1.5)
    for i, v in enumerate(target_counts.values):
        ax1.text(
            i,
            v + 10,
            str(v),
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Absolute Distribution', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Pie chart
    target_percent = (target_counts / target_counts.sum() * 100).values
    pie_labels = [
        f'{lbl} ({
            pct:.1f}%)' for lbl, pct in zip(
            target_labels, target_percent)]
    ax2.pie(
        target_counts.values,
        labels=pie_labels,
        colors=Config.COLORS_TARGET,
        autopct='%1.1f%%',
        startangle=90,
        textprops={
            'fontsize': 11})
    ax2.set_title('Percentage Distribution', fontsize=12, fontweight='bold')

    save_plot('1_target_distribution.png')

    for label, count, pct in zip(
            target_labels, target_counts.values, target_percent):
        print(f"  {label}: {count} ({pct:.1f}%)")

    return target_counts, target_labels, target_percent.tolist()


def plot_numeric_boxplots(
        df: pd.DataFrame,
        numeric_features: List[str]) -> None:
    """
    Create boxplots for numeric features grouped by target.

    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature column names
    """
    print_section("2. BOXPLOTS - NUMERIC FEATURES")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, feat in enumerate(numeric_features):
        data_sem = df[df[Config.TARGET_COLUMN] == 0][feat]
        data_com = df[df[Config.TARGET_COLUMN] == 1][feat]

        bp = axes[idx].boxplot([data_sem, data_com], labels=['No', 'Yes'],
                               patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#90EE90')
        bp['boxes'][1].set_facecolor('#FF6B6B')

        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)

        axes[idx].set_xlabel('Incident', fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel('Value', fontsize=11)
        axes[idx].set_title(
            feat.replace(
                '_',
                ' ').title(),
            fontsize=11,
            fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)

    save_plot('2_boxplots_numeric.png')


def plot_numeric_histograms(
        df: pd.DataFrame,
        numeric_features: List[str]) -> None:
    """
    Create histograms for numeric features grouped by target.

    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature column names
    """
    print_section("3. HISTOGRAMS - NUMERIC FEATURES")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, feat in enumerate(numeric_features):
        data_sem = df[df[Config.TARGET_COLUMN] == 0][feat]
        data_com = df[df[Config.TARGET_COLUMN] == 1][feat]

        axes[idx].hist(data_sem, bins=20, alpha=0.7, color='#90EE90',
                       edgecolor='black', label='No', linewidth=0.8)
        axes[idx].hist(data_com, bins=20, alpha=0.7, color='#FF6B6B',
                       edgecolor='black', label='Yes', linewidth=0.8)

        axes[idx].set_xlabel('Value', fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(
            feat.replace(
                '_',
                ' ').title(),
            fontsize=11,
            fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3)

    save_plot('3_histograms_numeric.png')


def plot_categorical_analysis(df_original: pd.DataFrame) -> None:
    """
    Create categorical feature analysis plots.

    Args:
        df_original: Original DataFrame with categorical columns
    """
    print_section("4. CATEGORICAL FEATURES ANALYSIS")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    cat_features_original = ['previsao_turbulencia', 'tipo_missao']

    for idx, feat in enumerate(cat_features_original):
        # Distribution
        counts = df_original[feat].value_counts()
        colors_cat = ['#FFB6C1', '#87CEEB']
        axes[idx * 2].bar(counts.index, counts.values, color=colors_cat[idx],
                          edgecolor='black', linewidth=1.5)
        axes[idx * 2].set_ylabel('Count', fontsize=11)
        title_dist = f'Distribution - {feat.replace("_", " ").title()}'
        axes[idx * 2].set_title(title_dist, fontsize=11, fontweight='bold')
        axes[idx * 2].grid(axis='y', alpha=0.3)

        # Incident rate
        incident_rate = df_original.groupby(feat)[Config.TARGET_COLUMN].apply(
            lambda x: (x == 1).sum() / len(x) * 100
        )
        axes[idx * 2 + 1].bar(incident_rate.index,
                              incident_rate.values,
                              color='#FF6B6B',
                              edgecolor='black',
                              linewidth=1.5)
        axes[idx * 2 + 1].set_ylabel('% Incidents', fontsize=11)
        axes[idx * 2 +
             1].set_xlabel(f'{feat.replace("_", " ")}'.title(), fontsize=11)
        title_rate = f'Incident Rate - {feat.replace("_", " ").title()}'
        axes[idx * 2 + 1].set_title(title_rate, fontsize=11, fontweight='bold')
        axes[idx * 2 + 1].grid(axis='y', alpha=0.3)

    save_plot('4_categorical_analysis.png')


def plot_correlation_matrix(df: pd.DataFrame, numeric_features: List[str],
                            mission_dummies_cols: List[str]) -> None:
    """
    Create correlation matrix heatmap.

    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature column names
        mission_dummies_cols: List of mission dummy column names
    """
    print_section("5. CORRELATION MATRIX")

    # Include all engineered features in correlation
    corr_features = (
        numeric_features +
        ['turb_num'] +
        mission_dummies_cols +
        ['razao_manut_idade', 'exp_x_turb', Config.TARGET_COLUMN]
    )
    corr_df = df[corr_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix', fontsize=13, fontweight='bold', pad=15)

    save_plot('5_correlation.png')


def plot_mean_comparison(
        df: pd.DataFrame,
        numeric_features: List[str]) -> None:
    """
    Create mean comparison bar plot.

    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature column names
    """
    print_section("6. MEAN COMPARISON")

    compare_features = numeric_features + ['turb_num']
    means_sem = df[df[Config.TARGET_COLUMN] == 0][compare_features].mean()
    means_com = df[df[Config.TARGET_COLUMN] == 1][compare_features].mean()

    x = np.arange(len(compare_features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, means_sem, width, label='No Incident',
           color='#90EE90', edgecolor='black', linewidth=1.5)
    ax.bar(x + width / 2, means_com, width, label='With Incident',
           color='#FF6B6B', edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Mean', fontsize=12)
    ax.set_xlabel('Variables', fontsize=12)
    ax.set_title('Mean Comparison: Flights WITH vs WITHOUT Incident',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n')
                       for f in compare_features], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    save_plot('6_mean_comparison.png')


# ==============================================================================
# SECTION 6: SUMMARY AND EXPORT FUNCTIONS
# ==============================================================================


def save_summary_statistics(df: pd.DataFrame, numeric_features: List[str],
                            mission_dummies_cols: List[str],
                            target_counts: pd.Series, target_labels: List[str],
                            target_percent: List[float]) -> None:
    """
    Save summary statistics to CSV files.

    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature column names
        mission_dummies_cols: List of mission dummy column names
        target_counts: Series with target value counts
        target_labels: List of target labels
        target_percent: List of target percentages
    """
    print_section("SAVING SUMMARY STATISTICS")

    # Descriptive statistics
    desc_stats = df[numeric_features].describe()
    desc_stats.to_csv(
        Config.TABLES_DIR /
        'descriptive_statistics.csv',
        encoding='utf-8')
    print("✓ Saved: descriptive_statistics.csv")

    # Correlation with target
    corr_features = (
        numeric_features +
        ['turb_num'] +
        mission_dummies_cols +
        ['razao_manut_idade', 'exp_x_turb', Config.TARGET_COLUMN]
    )
    target_corr = df[corr_features].corr()[
        Config.TARGET_COLUMN].drop(
        Config.TARGET_COLUMN)
    target_corr_df = pd.DataFrame({
        'Feature': target_corr.index,
        'Correlation': target_corr.values
    }).sort_values('Correlation', key=abs, ascending=False)
    target_corr_df.to_csv(Config.TABLES_DIR / 'correlation_with_target.csv',
                          index=False, encoding='utf-8')
    print("✓ Saved: correlation_with_target.csv")

    # Class distribution
    dist_df = pd.DataFrame({
        'Class': target_labels,
        'Count': target_counts.values,
        'Percentage': target_percent
    })
    dist_df.to_csv(Config.TABLES_DIR / 'class_distribution.csv',
                   index=False, encoding='utf-8')
    print("✓ Saved: class_distribution.csv")


def print_summary(df_clean: pd.DataFrame) -> None:
    """
    Print final analysis summary.

    Args:
        df_clean: Cleaned DataFrame
    """
    print_section("FEATURE ENGINEERING SUMMARY")

    print(f"\nFinal dataset: {df_clean.shape[1] - 1} features")
    print("  • 3 original numeric features")
    print("  • 1 turbulence numeric (turb_num)")
    print("  • 3 mission dummies (one-hot encoding)")
    print("  • 2 engineered interaction features")
    print("\nEngineered features:")
    print("  • razao_manut_idade: maintenance hours / aircraft age")
    print("  • exp_x_turb: pilot experience × turbulence")

    print("\n" + "=" * 120)
    print("EXPLORATORY DATA ANALYSIS COMPLETED")
    print("=" * 120)
    print(
        f"\nGenerated {len(list(Config.GRAPHICS_DIR.glob('*.png')))} PNG graphics")
    print(f"Generated {len(list(Config.TABLES_DIR.glob('*.csv')))} CSV tables")
    print("\n" + "=" * 120)


# ==============================================================================
# SECTION 7: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the entire exploratory analysis."""
    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    Config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("EXPLORATORY DATA ANALYSIS - FLIGHT INCIDENT PREDICTION")
    print("=" * 120)
    print(f"\nRandom seed: {SEED}")
    print(f"Remove outliers: {Config.REMOVE_OUTLIERS}")
    print(f"Dataset path: {Config.DATASET_PATH.absolute()}")

    print_section("CREATING OUTPUT DIRECTORIES")
    print(f"✓ Output directory: {Config.OUTPUT_DIR.absolute()}")
    print(f"✓ Graphics directory: {Config.GRAPHICS_DIR.absolute()}")
    print(f"✓ Results directory: {Config.TABLES_DIR.absolute()}")

    try:
        # 1. Load data
        print_section("LOAD DATASET")
        df = load_dataset(Config.DATASET_PATH)
        validate_columns(df)

        # 2. Feature engineering
        df, mission_dummies_cols = engineer_features(df)

        # 3. Outlier removal (optional)
        numeric_cols = [
            'idade_aeronave_anos',
            'horas_voo_desde_ultima_manutencao',
            'experiencia_piloto_anos'
        ]
        if Config.REMOVE_OUTLIERS:
            df = remove_outliers_iqr(df, numeric_cols)
        else:
            print("\n⚠️  Outlier removal disabled - keeping all data")

        # 4. Save cleaned dataset
        df_clean = save_cleaned_dataset(df, mission_dummies_cols)

        # Use cleaned dataset for visualizations
        df = df_clean

        # 5. Visualizations
        target_counts, target_labels, target_percent = plot_target_distribution(
            df)
        plot_numeric_boxplots(df, numeric_cols)
        plot_numeric_histograms(df, numeric_cols)

        # For categorical analysis, reload original data
        df_original = pd.read_csv(Config.DATASET_PATH, encoding='utf-8')
        if Config.REMOVE_OUTLIERS:
            outlier_mask = pd.Series([False] * len(df_original))
            for col in numeric_cols:
                Q1 = df_original[col].quantile(0.25)
                Q3 = df_original[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask |= (
                    df_original[col] < lower_bound) | (
                    df_original[col] > upper_bound)
            df_original = df_original[~outlier_mask].reset_index(drop=True)

        plot_categorical_analysis(df_original)
        plot_correlation_matrix(df, numeric_cols, mission_dummies_cols)
        plot_mean_comparison(df, numeric_cols)

        # 6. Save statistics
        save_summary_statistics(df, numeric_cols, mission_dummies_cols,
                                target_counts, target_labels, target_percent)

        # 7. Print summary
        print_summary(df_clean)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
