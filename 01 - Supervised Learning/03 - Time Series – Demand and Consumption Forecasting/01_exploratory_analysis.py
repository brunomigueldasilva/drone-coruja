#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
TIME SERIES EXPLORATORY DATA ANALYSIS
==============================================================================

Purpose: Perform comprehensive exploratory analysis on time series datasets.

This script:
1. Loads voltage and daily missions time series data
2. Performs data validation and gap detection
3. Creates time series visualizations
4. Performs seasonal decomposition (trend, seasonal, residual)
5. Analyzes autocorrelation (ACF) and partial autocorrelation (PACF)
6. Tests for stationarity using Augmented Dickey-Fuller test
7. Saves all plots and analysis results

Expected Datasets:
- voltagem_bateria.csv (columns: timestamp, voltagem)
- missoes_diarias.csv (columns: data, num_missoes)

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

from __future__ import annotations
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Configuration Constants
SEED = 42
np.random.seed(SEED)


class Config:
    """Time series analysis configuration parameters."""
    # Directories
    INPUT_DIR = Path('inputs')
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'

    # Input files
    VOLTAGE_PATH = INPUT_DIR / 'voltagem_bateria.csv'
    MISSIONS_PATH = INPUT_DIR / 'missoes_diarias.csv'

    # Analysis parameters
    ACF_LAGS = 40  # Number of lags for ACF/PACF plots
    DECOMPOSITION_MODEL = 'additive'  # 'additive' or 'multiplicative'
    ADF_SIGNIFICANCE = 0.05  # Significance level for ADF test

    # Visualization settings
    DPI = 300
    FIGSIZE = (14, 6)
    FIGSIZE_DECOMP = (14, 10)
    FIGSIZE_ACF = (14, 5)


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
        filename: Name of the file to save
        dpi: Dots per inch for image resolution
    """
    filepath = Config.GRAPHICS_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def print_section(title: str, char: str = "=") -> None:
    """
    Print formatted section header.

    Args:
        title: Title text to display
        char: Character to use for separator line
    """
    print("\n" + char * 120)
    print(title)
    print(char * 120)


# ==============================================================================
# SECTION 3: DATA LOADING AND PREPARATION
# ==============================================================================


def load_and_prepare_data(
        filepath: Path,
        time_col: str,
        target_col: str) -> pd.DataFrame:
    """
    Load time series data and prepare it for analysis.

    This function:
    1. Loads CSV data
    2. Converts time column to datetime format
    3. Sets time column as index
    4. Sorts by time
    5. Validates data integrity

    Args:
        filepath: Path to CSV file
        time_col: Name of the time column
        target_col: Name of the target variable column

    Returns:
        DataFrame with datetime index and prepared data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath.absolute()}")

    print(f"\n→ Loading data from: {filepath.name}")

    # Load data
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"  ✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Validate columns
    if time_col not in df.columns:
        raise ValueError(
            f"Time column '{time_col}' not found. Available: {
                df.columns.tolist()}")
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available: {
                df.columns.tolist()}")

    # Convert time column to datetime
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    print(f"  ✓ Converted '{time_col}' to datetime format")

    # Check for invalid dates
    invalid_dates = df[time_col].isna().sum()
    if invalid_dates > 0:
        print(
            f"  ⚠️  Warning: {invalid_dates} invalid dates found and will be removed")
        df = df.dropna(subset=[time_col])

    # Set time column as index
    df = df.set_index(time_col)
    df = df.sort_index()
    print(f"  ✓ Set '{time_col}' as index and sorted chronologically")

    # Keep only target column
    df = df[[target_col]]

    return df


def check_time_series_quality(df: pd.DataFrame, series_name: str) -> dict:
    """
    Check time series data quality and report issues.

    This function identifies:
    - Time series frequency (daily, hourly, etc.)
    - Missing timestamps (gaps in the series)
    - Duplicate timestamps
    - Missing values in the target variable

    Args:
        df: DataFrame with datetime index
        series_name: Name of the time series for reporting

    Returns:
        Dictionary with quality metrics
    """
    print(f"\n→ Checking data quality for {series_name}...")

    quality_report = {}

    # Infer frequency
    try:
        inferred_freq = pd.infer_freq(df.index)
        quality_report['frequency'] = inferred_freq
        if inferred_freq:
            print(f"  ✓ Detected frequency: {inferred_freq}")
        else:
            print("  ⚠️  Could not detect a consistent frequency")
    except Exception:
        quality_report['frequency'] = None
        print("  ⚠️  Could not infer frequency")

    # Check for duplicate timestamps
    duplicates = df.index.duplicated().sum()
    quality_report['duplicates'] = duplicates
    if duplicates > 0:
        print(f"  ⚠️  Warning: {duplicates} duplicate timestamps found")
    else:
        print("  ✓ No duplicate timestamps")

    # Check for gaps in time series
    if len(df) > 1:
        time_diffs = df.index.to_series().diff()
        mode_diff = time_diffs.mode()[0] if len(
            time_diffs.mode()) > 0 else None
        if mode_diff:
            expected_intervals = (df.index[-1] - df.index[0]) / mode_diff
            actual_intervals = len(df)
            missing_intervals = int(expected_intervals) - actual_intervals
            quality_report['missing_intervals'] = missing_intervals

            if missing_intervals > 0:
                print(
                    f"  ⚠️  Warning: Approximately {missing_intervals} missing time intervals detected")
            else:
                print("  ✓ No missing time intervals detected")
        else:
            quality_report['missing_intervals'] = None
            print("  ⚠️  Could not determine missing intervals")
    else:
        quality_report['missing_intervals'] = None

    # Check for missing values
    missing_values = df.iloc[:, 0].isna().sum()
    quality_report['missing_values'] = missing_values
    if missing_values > 0:
        print(
            f"  ⚠️  Warning: {missing_values} missing values in target variable")
    else:
        print("  ✓ No missing values in target variable")

    return quality_report


# ==============================================================================
# SECTION 4: STATISTICAL ANALYSIS
# ==============================================================================


def print_basic_statistics(df: pd.DataFrame, series_name: str) -> None:
    """
    Print basic descriptive statistics for the time series.

    Args:
        df: DataFrame with time series data
        series_name: Name of the time series for reporting
    """
    print(f"\n→ Basic statistics for {series_name}:")

    stats = df.iloc[:, 0].describe()
    print(f"\n  Count:       {stats['count']:.0f}")
    print(f"  Mean:        {stats['mean']:.4f}")
    print(f"  Std Dev:     {stats['std']:.4f}")
    print(f"  Min:         {stats['min']:.4f}")
    print(f"  25%:         {stats['25%']:.4f}")
    print(f"  Median:      {stats['50%']:.4f}")
    print(f"  75%:         {stats['75%']:.4f}")
    print(f"  Max:         {stats['max']:.4f}")

    # Additional time series info
    print(f"\n  Time Range:  {df.index.min()} to {df.index.max()}")
    print(f"  Duration:    {df.index.max() - df.index.min()}")


def test_stationarity(series: pd.Series, series_name: str) -> dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    STATIONARITY EXPLAINED:
    ------------------------
    A time series is stationary if its statistical properties (mean, variance,
    autocorrelation) don't change over time. Stationarity is crucial for many
    time series forecasting models because:

    1. STATIONARY = Predictable patterns
       - Constant mean (no trend)
       - Constant variance (no heteroscedasticity)
       - Constant autocorrelation structure

    2. NON-STATIONARY = Unpredictable changes
       - Trending mean
       - Changing variance
       - Time-dependent patterns

    AUGMENTED DICKEY-FULLER (ADF) TEST:
    ------------------------------------
    - Null Hypothesis (H0): Series has a unit root (NON-stationary)
    - Alternative Hypothesis (H1): Series is stationary

    INTERPRETATION:
    - p-value < 0.05 → REJECT H0 → Series IS stationary ✓
    - p-value ≥ 0.05 → FAIL TO REJECT H0 → Series is NON-stationary
    - More negative ADF statistic → Stronger evidence of stationarity

    WHY IT MATTERS:
    - Many forecasting models (ARIMA, etc.) require stationary data
    - If non-stationary, we may need to apply differencing or transformations

    Args:
        series: Time series data to test
        series_name: Name of the series for reporting

    Returns:
        Dictionary with test results
    """
    print(
        f"\n→ Testing stationarity for {series_name} (Augmented Dickey-Fuller Test)...")

    # Remove any NaN values
    series_clean = series.dropna()

    # Perform ADF test
    adf_result = adfuller(series_clean, autolag='AIC')

    results = {
        'adf_statistic': adf_result[0],
        'p_value': adf_result[1],
        'n_lags': adf_result[2],
        'n_obs': adf_result[3],
        'critical_values': adf_result[4],
        'is_stationary': adf_result[1] < Config.ADF_SIGNIFICANCE
    }

    # Print results
    print(f"\n  ADF Statistic:     {results['adf_statistic']:.6f}")
    print(f"  p-value:           {results['p_value']:.6f}")
    print(f"  Lags used:         {results['n_lags']}")
    print(f"  Observations:      {results['n_obs']}")
    print("\n  Critical Values:")
    for key, value in results['critical_values'].items():
        print(f"    {key:>4}: {value:.4f}")

    # Interpretation
    print("\n  " + "-" * 60)
    if results['is_stationary']:
        print(
            f"  ✓ CONCLUSION: Series IS STATIONARY (p-value = {results['p_value']:.6f} < 0.05)")
        print("    → The series has constant statistical properties over time")
        print("    → Suitable for standard forecasting models without transformation")
    else:
        print(
            f"  ⚠️  CONCLUSION: Series is NON-STATIONARY (p-value = {results['p_value']:.6f} ≥ 0.05)")
        print("    → The series has time-dependent statistical properties")
        print("    → Consider applying differencing or other transformations")
        print("    → Check decomposition for trend or seasonality")

    print("  " + "-" * 60)

    return results


# ==============================================================================
# SECTION 5: TIME SERIES VISUALIZATION
# ==============================================================================


def plot_time_series(
        df: pd.DataFrame,
        target_col: str,
        series_name: str,
        filename: str) -> None:
    """
    Create line plot of time series over time.

    Args:
        df: DataFrame with time series data
        target_col: Name of the target column
        series_name: Name for plot title
        filename: Name for saved file
    """
    print(f"\n→ Creating time series plot for {series_name}...")

    fig, ax = plt.subplots(figsize=Config.FIGSIZE)

    ax.plot(
        df.index,
        df[target_col],
        linewidth=1.5,
        color='#2E86AB',
        alpha=0.8)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel(
        target_col.replace(
            '_',
            ' ').title(),
        fontsize=12,
        fontweight='bold')
    ax.set_title(
        f'{series_name} - Time Series',
        fontsize=14,
        fontweight='bold',
        pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Format x-axis
    fig.autofmt_xdate()

    save_plot(filename)


def plot_seasonal_decomposition(
        df: pd.DataFrame,
        target_col: str,
        series_name: str,
        filename: str,
        period: Optional[int] = None) -> None:
    """
    Perform and plot seasonal decomposition.

    SEASONAL DECOMPOSITION EXPLAINED:
    ----------------------------------
    Decomposes a time series into three components:

    1. TREND: Long-term progression of the series
       - Shows overall direction (increasing, decreasing, stable)
       - Removes short-term fluctuations
       - Useful for understanding long-term behavior

    2. SEASONAL: Regular, periodic patterns
       - Repeating patterns at fixed intervals (daily, weekly, yearly)
       - Same pattern occurs at regular intervals
       - Useful for understanding cyclical behavior

    3. RESIDUAL (RANDOM): What remains after removing trend and seasonality
       - Irregular, random fluctuations
       - Should ideally be random noise if decomposition is good
       - Large residuals may indicate missing components or anomalies

    MODELS:
    - Additive: Y = Trend + Seasonal + Residual (constant seasonality)
    - Multiplicative: Y = Trend × Seasonal × Residual (proportional seasonality)

    INTERPRETATION:
    - Strong trend → Consider differencing for stationarity
    - Strong seasonality → Use seasonal models (SARIMA, etc.)
    - Large residuals → May need better model or transformation

    Args:
        df: DataFrame with time series data
        target_col: Name of the target column
        series_name: Name for plot title
        filename: Name for saved file
        period: Period for seasonal decomposition (auto-detected if None)
    """
    print(f"\n→ Performing seasonal decomposition for {series_name}...")

    # Remove NaN values
    series_clean = df[target_col].dropna()

    # Auto-detect period if not specified
    if period is None:
        # Try to infer from frequency
        freq = pd.infer_freq(series_clean.index)
        if freq:
            # Common period mappings
            period_map = {
                'D': 7,      # Daily → weekly pattern
                'H': 24,     # Hourly → daily pattern
                'T': 60,     # Minute → hourly pattern
                'W': 52,     # Weekly → yearly pattern
                'M': 12,     # Monthly → yearly pattern
            }
            period = period_map.get(freq[0], 7)  # Default to 7
        else:
            period = 7  # Default period

    print(f"  Using period: {period}")

    try:
        # Perform decomposition
        decomposition = seasonal_decompose(
            series_clean,
            model=Config.DECOMPOSITION_MODEL,
            period=period,
            extrapolate_trend='freq')

        # Create plot
        fig, axes = plt.subplots(
            4, 1, figsize=Config.FIGSIZE_DECOMP, sharex=True)

        # Original
        axes[0].plot(decomposition.observed.index, decomposition.observed,
                     color='#2E86AB', linewidth=1.5, label='Original')
        axes[0].set_ylabel('Original', fontsize=11, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(
            f'{series_name} - Seasonal Decomposition ({
                Config.DECOMPOSITION_MODEL.title()} Model)',
            fontsize=13,
            fontweight='bold',
            pad=15)

        # Trend
        axes[1].plot(decomposition.trend.index, decomposition.trend,
                     color='#A23B72', linewidth=2, label='Trend')
        axes[1].set_ylabel('Trend', fontsize=11, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # Seasonal
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal,
                     color='#F18F01', linewidth=1.5, label='Seasonal')
        axes[2].set_ylabel('Seasonal', fontsize=11, fontweight='bold')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # Residual
        axes[3].plot(decomposition.resid.index, decomposition.resid,
                     color='#C73E1D', linewidth=1, label='Residual', alpha=0.8)
        axes[3].set_ylabel('Residual', fontsize=11, fontweight='bold')
        axes[3].set_xlabel('Time', fontsize=11, fontweight='bold')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

        fig.autofmt_xdate()
        save_plot(filename)

        print("  ✓ Decomposition completed successfully")
        print("\n  INTERPRETATION GUIDE:")
        print("  - Trend: Long-term direction of the series")
        print(f"  - Seasonal: Regular, repeating patterns (period={period})")
        print("  - Residual: Random fluctuations after removing trend and seasonality")

    except Exception as e:
        print(f"  ✗ Could not perform decomposition: {e}")
        print("  → This may happen if the series is too short or irregular")


def plot_acf_pacf(df: pd.DataFrame, target_col: str, series_name: str,
                  filename_acf: str, filename_pacf: str) -> None:
    """
    Plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

    AUTOCORRELATION (ACF) EXPLAINED:
    ---------------------------------
    ACF measures correlation between a time series and its lagged values.

    WHAT IT SHOWS:
    - Correlation at lag k = how related values are k time steps apart
    - Values range from -1 (perfect negative) to +1 (perfect positive)
    - Blue shaded area = confidence interval (values outside are significant)

    INTERPRETATION:
    - Gradual decay → Autoregressive (AR) process
    - Sharp cutoff at lag q → Moving Average (MA) of order q
    - No significant lags → White noise (random)
    - Persistent high values → Non-stationary (trending)

    PARTIAL AUTOCORRELATION (PACF) EXPLAINED:
    ------------------------------------------
    PACF measures DIRECT correlation at each lag, removing indirect effects.

    WHAT IT SHOWS:
    - Correlation at lag k after removing effects of lags 1 to k-1
    - Isolates the "pure" relationship at each lag
    - Complements ACF for model identification

    INTERPRETATION:
    - Sharp cutoff at lag p → AR process of order p
    - Gradual decay → MA process
    - Compare with ACF to identify appropriate model:
      * ACF tails off, PACF cuts off → AR model
      * ACF cuts off, PACF tails off → MA model
      * Both tail off → ARMA model

    PRACTICAL USE:
    1. Check stationarity: Non-stationary series have slowly decaying ACF
    2. Identify model order: Use cutoff patterns to determine p and q for ARIMA(p,d,q)
    3. Detect seasonality: Significant spikes at seasonal lags (7, 12, 24, etc.)

    Args:
        df: DataFrame with time series data
        target_col: Name of the target column
        series_name: Name for plot title
        filename_acf: Name for ACF plot file
        filename_pacf: Name for PACF plot file
    """
    print(f"\n→ Creating ACF and PACF plots for {series_name}...")

    # Remove NaN values
    series_clean = df[target_col].dropna()

    # ACF Plot
    fig, ax = plt.subplots(figsize=Config.FIGSIZE_ACF)
    plot_acf(series_clean, lags=Config.ACF_LAGS, ax=ax, alpha=0.05)
    ax.set_title(f'{series_name} - Autocorrelation Function (ACF)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Lag', fontsize=11, fontweight='bold')
    ax.set_ylabel('Correlation', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    save_plot(filename_acf)

    # PACF Plot
    fig, ax = plt.subplots(figsize=Config.FIGSIZE_ACF)
    plot_pacf(
        series_clean,
        lags=Config.ACF_LAGS,
        ax=ax,
        alpha=0.05,
        method='ywm')
    ax.set_title(f'{series_name} - Partial Autocorrelation Function (PACF)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Lag', fontsize=11, fontweight='bold')
    ax.set_ylabel('Partial Correlation', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    save_plot(filename_pacf)

    print("\n  INTERPRETATION GUIDE:")
    print("  ACF:")
    print("    - Shows correlation at different time lags")
    print("    - Values outside blue region are statistically significant")
    print("    - Slow decay suggests non-stationarity or AR process")
    print("    - Sharp cutoff suggests MA process")
    print("\n  PACF:")
    print("    - Shows direct correlation after removing indirect effects")
    print("    - Sharp cutoff at lag p suggests AR(p) model")
    print("    - Use with ACF to identify appropriate forecasting model")


# ==============================================================================
# SECTION 6: COMPREHENSIVE ANALYSIS FUNCTION
# ==============================================================================


def analyze_time_series(df: pd.DataFrame, target_col: str, series_name: str,
                        filename_prefix: str) -> dict:
    """
    Perform comprehensive time series analysis.

    This function orchestrates all analysis steps:
    1. Data quality checks
    2. Basic statistics
    3. Time series visualization
    4. Seasonal decomposition
    5. ACF/PACF analysis
    6. Stationarity testing

    Args:
        df: DataFrame with time series data (datetime index)
        target_col: Name of the target variable column
        series_name: Descriptive name for the series
        filename_prefix: Prefix for saved plot filenames

    Returns:
        Dictionary with analysis results
    """
    print_section(f"ANALYZING: {series_name}", "=")

    results = {}

    # 1. Data quality
    results['quality'] = check_time_series_quality(df, series_name)

    # 2. Basic statistics
    print_basic_statistics(df, series_name)

    # 3. Time series plot
    plot_time_series(
        df,
        target_col,
        series_name,
        f"{filename_prefix}_timeseries.png")

    # 4. Seasonal decomposition
    plot_seasonal_decomposition(df, target_col, series_name,
                                f"{filename_prefix}_decomposition.png")

    # 5. ACF and PACF
    plot_acf_pacf(df, target_col, series_name,
                  f"{filename_prefix}_acf.png",
                  f"{filename_prefix}_pacf.png")

    # 6. Stationarity test
    results['stationarity'] = test_stationarity(df[target_col], series_name)

    return results


# ==============================================================================
# SECTION 7: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main function that coordinates the entire time series exploratory analysis.

    This function:
    1. Creates output directories
    2. Loads and prepares both datasets
    3. Performs comprehensive analysis on each dataset
    4. Saves all visualizations and results
    """
    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("TIME SERIES EXPLORATORY DATA ANALYSIS")
    print("=" * 120)
    print(f"\nRandom seed: {SEED}")
    print(f"ACF/PACF lags: {Config.ACF_LAGS}")
    print(f"Decomposition model: {Config.DECOMPOSITION_MODEL}")

    print_section("CREATING OUTPUT DIRECTORIES")
    print(f"✓ Output directory: {Config.OUTPUT_DIR.absolute()}")
    print(f"✓ Graphics directory: {Config.GRAPHICS_DIR.absolute()}")

    try:
        # ======================================================================
        # DATASET 1: BATTERY VOLTAGE
        # ======================================================================
        print_section("DATASET 1: BATTERY VOLTAGE", "=")

        df_voltage = load_and_prepare_data(
            Config.VOLTAGE_PATH,
            time_col='timestamp',
            target_col='voltagem'
        )

        results_voltage = analyze_time_series(
            df_voltage,
            target_col='voltagem',
            series_name='Battery Voltage',
            filename_prefix='voltage'
        )

        # ======================================================================
        # DATASET 2: DAILY MISSIONS
        # ======================================================================
        print_section("DATASET 2: DAILY MISSIONS", "=")

        df_missions = load_and_prepare_data(
            Config.MISSIONS_PATH,
            time_col='data',
            target_col='num_missoes'
        )

        results_missions = analyze_time_series(
            df_missions,
            target_col='num_missoes',
            series_name='Daily Missions',
            filename_prefix='missions'
        )

        # ======================================================================
        # SUMMARY
        # ======================================================================
        print_section("ANALYSIS SUMMARY", "=")

        print("\nBATTERY VOLTAGE:")
        print(
            f"  Stationarity: {
                'YES ✓' if results_voltage['stationarity']['is_stationary'] else 'NO ✗'}")
        print(
            f"  ADF p-value: {results_voltage['stationarity']['p_value']:.6f}")

        print("\nDAILY MISSIONS:")
        print(
            f"  Stationarity: {
                'YES ✓' if results_missions['stationarity']['is_stationary'] else 'NO ✗'}")
        print(
            f"  ADF p-value: {results_missions['stationarity']['p_value']:.6f}")

        # Count generated files
        n_plots = len(list(Config.GRAPHICS_DIR.glob('*.png')))

        print("\n" + "=" * 120)
        print("EXPLORATORY DATA ANALYSIS COMPLETED")
        print("=" * 120)
        print(
            f"\nGenerated {n_plots} plots in: {
                Config.GRAPHICS_DIR.absolute()}")
        print("\nPlots created:")
        print("  • Time series visualizations (2)")
        print("  • Seasonal decompositions (2)")
        print("  • ACF plots (2)")
        print("  • PACF plots (2)")
        print("\nNext steps:")
        print("  1. Review stationarity results - may need differencing")
        print("  2. Check ACF/PACF patterns for model selection")
        print("  3. Examine decomposition for trend and seasonality")
        print("  4. Consider appropriate forecasting models based on findings")
        print("\n" + "=" * 120)

    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("\n⚠️  Please ensure the following files exist:")
        print(f"   - {Config.VOLTAGE_PATH.absolute()}")
        print(f"   - {Config.MISSIONS_PATH.absolute()}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
