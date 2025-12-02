#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
TIME SERIES RESIDUAL ANALYSIS - DIAGNOSTIC EVALUATION
==============================================================================

Purpose: Perform comprehensive residual analysis for the best-performing model

This script:
1. Loads test data and best model predictions
2. Calculates residuals (errors) and their statistical properties
3. Creates 4 diagnostic plots:
   - Residual distribution (histogram with normal overlay)
   - Residuals over time (temporal patterns)
   - Autocorrelation function (ACF of residuals)
   - Q-Q plot (normality check)
4. Performs statistical tests (Shapiro-Wilk, ACF significance)
5. Generates detailed interpretation and recommendations

WHAT ARE RESIDUALS?
===================
Residuals = Actual - Predicted = Unexplained Variance = Model Errors

Residuals represent what the model FAILED to capture. They are the leftover
errors after the model has done its best to explain the data.

WHY ANALYZE RESIDUALS?
=======================
Residuals reveal systematic deficiencies in the model:
- Patterns in residuals = model is missing something
- Trends = model doesn't capture long-term drift
- Cycles = model misses seasonal/periodic patterns
- Increasing variance = heteroscedasticity (error depends on level)
- Autocorrelation = model hasn't captured temporal dependencies

IDEAL RESIDUALS:
================
1. RANDOM: No patterns, trends, or structure
2. NORMAL: Follow normal distribution (bell curve)
3. ZERO MEAN: Average residual ≈ 0 (no systematic bias)
4. CONSTANT VARIANCE: Error size doesn't change over time (homoscedastic)
5. UNCORRELATED: No autocorrelation (errors are independent)

If residuals violate these properties, it indicates the model can be improved
with better features, transformations, or different modeling approaches.

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import shapiro

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


# Configuration
class Config:
    """Configuration for residual analysis."""
    # Directories
    OUTPUT_DIR = Path("outputs")
    DATA_DIR = OUTPUT_DIR / "data_processed"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    GRAPHICS_DIR = OUTPUT_DIR / "graphics"

    # Analysis parameters
    HISTOGRAM_BINS = 50
    ACF_LAGS = 40
    ALPHA = 0.05  # Significance level for tests

    # Plot settings
    FIGSIZE = (16, 12)
    DPI = 300


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 100)
    print(title)
    print(char * 100)


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_test_data_and_predictions(
        dataset_name='voltage') -> Tuple[pd.Series, np.ndarray, str, Dict]:
    """
    Load test data and best model predictions.

    Args:
        dataset_name: 'voltage' or 'missions'

    Returns:
        Tuple: (y_test, predictions, model_name, metrics)
    """
    print_section("LOADING DATA")

    # Load test data
    y_test_file = Config.DATA_DIR / dataset_name / 'y_test.pkl'
    if not y_test_file.exists():
        print(f"✗ ERROR: Test data not found: {y_test_file}")
        exit(1)

    with open(y_test_file, 'rb') as f:
        y_test = pickle.load(f)
    print(f"✓ Loaded y_test: shape {y_test.shape}")

    # Load metrics to find best model
    metrics_file = Config.OUTPUT_DIR / f'{dataset_name}_metrics_comparison.csv'
    if not metrics_file.exists():
        print(f"✗ ERROR: Metrics file not found: {metrics_file}")
        exit(1)

    metrics_df = pd.read_csv(metrics_file)

    # Find best model (lowest RMSE)
    best_idx = metrics_df['RMSE'].idxmin()
    best_row = metrics_df.loc[best_idx]
    model_name = best_row['Model']

    metrics = {
        'MAE': best_row['MAE'],
        'RMSE': best_row['RMSE'],
        'MAPE': best_row['MAPE']
    }

    print(f"✓ Best model: {model_name} (RMSE: {metrics['RMSE']:.4f})")

    # Load predictions
    predictions_file = Config.PREDICTIONS_DIR / \
        dataset_name / f'{model_name}_predictions.pkl'
    if not predictions_file.exists():
        print(f"✗ ERROR: Predictions not found: {predictions_file}")
        exit(1)

    with open(predictions_file, 'rb') as f:
        predictions = pickle.load(f)
    print(f"✓ Loaded predictions: shape {predictions.shape}")

    return y_test, predictions, model_name, metrics


# ==============================================================================
# SECTION 4: RESIDUAL CALCULATION
# ==============================================================================


def calculate_residuals(y_test: pd.Series, y_pred: np.ndarray) -> pd.Series:
    """
    Calculate residuals (errors).

    RESIDUALS EXPLAINED:
    ====================
    Residuals = Actual - Predicted = Unexplained Variance

    Residuals represent what the model FAILED to explain:
    - Positive residual: Model under-predicted (predicted too low)
    - Negative residual: Model over-predicted (predicted too high)
    - Zero residual: Perfect prediction

    IDEAL RESIDUALS:
    - Mean ≈ 0 (no systematic bias)
    - Random distribution (no patterns)
    - Constant variance (homoscedastic)
    - Normal distribution (for statistical tests)
    - No autocorrelation (errors are independent)

    Args:
        y_test: Actual values
        y_pred: Predicted values

    Returns:
        pd.Series: Residuals with datetime index
    """
    print_section("CALCULATING RESIDUALS")

    # Calculate residuals
    residuals = y_test.values - y_pred

    # Create Series with datetime index
    residuals_series = pd.Series(
        residuals,
        index=y_test.index,
        name='residuals')

    # Summary statistics
    print("\nRESIDUAL SUMMARY STATISTICS:")
    print(f"  Mean:     {np.mean(residuals):>10.4f}  (ideal: ≈ 0)")
    print(f"  Std Dev:  {np.std(residuals):>10.4f}")
    print(f"  Min:      {np.min(residuals):>10.4f}")
    print(f"  Max:      {np.max(residuals):>10.4f}")
    print(f"  Median:   {np.median(residuals):>10.4f}")
    print(f"  Q1 (25%): {np.percentile(residuals, 25):>10.4f}")
    print(f"  Q3 (75%): {np.percentile(residuals, 75):>10.4f}")

    print("\n💡 INTERPRETATION:")
    print("   Ideal residuals should have:")
    print("   - Mean ≈ 0 (no systematic bias)")
    print("   - Random distribution (no patterns)")
    print("   - Constant variance across time")
    print("   - Approximately normal distribution")

    return residuals_series


# ==============================================================================
# SECTION 5: DIAGNOSTIC PLOTS
# ==============================================================================


def plot_residual_distribution(ax, residuals: np.ndarray) -> None:
    """
    Plot residual distribution with normal overlay.

    WHAT THIS SHOWS:
    ================
    - Shape of residual distribution
    - Whether residuals are symmetric
    - Presence of outliers
    - Deviation from normality

    INTERPRETATION:
    - Bell curve centered at 0 = GOOD (normal residuals)
    - Skewed distribution = model bias in one direction
    - Multiple peaks = model behaves differently in different regimes
    - Heavy tails = more extreme errors than expected

    Args:
        ax: Matplotlib axis
        residuals: Residual values
    """
    # Plot histogram
    n, bins, patches = ax.hist(
        residuals,
        bins=Config.HISTOGRAM_BINS,
        density=True,
        alpha=0.7,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5
    )

    # Overlay normal distribution
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_dist = stats.norm.pdf(x, mu, sigma)
    ax.plot(
        x,
        normal_dist,
        'r-',
        linewidth=2,
        label=f'Normal(μ={
            mu:.2f}, σ={
            sigma:.2f})')

    # Add vertical line at x=0 (ideal center)
    ax.axvline(
        x=0,
        color='green',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label='Ideal (zero)')

    # Formatting
    ax.set_xlabel('Residual Value', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title(
        'Distribution of Residuals\n(Should be approximately normal and centered at zero)',
        fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_residuals_over_time(ax, residuals: pd.Series) -> None:
    """
    Plot residuals over time.

    WHAT THIS SHOWS:
    ================
    - Temporal patterns in errors
    - Trends in residuals (systematic drift)
    - Changes in error variance over time
    - Periods of good vs. poor performance

    INTERPRETATION:
    - Flat line around 0 = GOOD (random errors)
    - Trend up/down = model drift (not capturing long-term changes)
    - Cycles/waves = model missing seasonal patterns
    - Increasing spread = heteroscedasticity (variance grows)
    - Clusters of same-sign errors = autocorrelation

    PATTERNS INDICATE:
    - Upward trend: Model increasingly under-predicts
    - Downward trend: Model increasingly over-predicts
    - Funnel shape: Variance changes with time
    - Waves: Missed cyclical/seasonal component

    Args:
        ax: Matplotlib axis
        residuals: Residuals with datetime index
    """
    # Color points by sign
    colors = ['red' if r < 0 else 'blue' for r in residuals.values]

    # Plot residuals
    ax.scatter(residuals.index, residuals.values, c=colors, alpha=0.6, s=20)
    ax.plot(
        residuals.index,
        residuals.values,
        color='gray',
        alpha=0.3,
        linewidth=0.5)

    # Add horizontal line at y=0
    ax.axhline(
        y=0,
        color='green',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label='Zero line (ideal)')

    # Add trend line
    x_numeric = np.arange(len(residuals))
    z = np.polyfit(x_numeric, residuals.values, 1)
    p = np.poly1d(z)
    ax.plot(residuals.index, p(x_numeric), "r-", linewidth=2,
            alpha=0.7, label=f'Trend (slope={z[0]:.4f})')

    # Formatting
    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Residual Value', fontweight='bold')
    ax.set_title(
        'Residuals Over Time\n(Should show no patterns, trends, or increasing variance)',
        fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def plot_residual_acf(ax, residuals: np.ndarray) -> int:
    """
    Plot autocorrelation function of residuals.

    WHAT THIS SHOWS:
    ================
    - Correlation between residuals at different time lags
    - Whether errors are independent or correlated
    - Presence of temporal structure in errors

    AUTOCORRELATION IN RESIDUALS MEANS:
    ====================================
    The model hasn't captured all temporal dependencies in the data.

    INTERPRETATION:
    - All bars within blue bands = GOOD (no autocorrelation)
    - Bars exceeding bands = significant autocorrelation
    - Pattern of significant lags = missed temporal structure

    IMPLICATIONS:
    - Lag 1 significant: Model missing immediate past influence
    - Seasonal lags significant: Model missing seasonal patterns
    - Many lags significant: Model fundamentally misses temporal dynamics

    FIXES:
    - Add more lag features
    - Use autoregressive models (ARIMA, SARIMA)
    - Include rolling window features
    - Try LSTM or other sequence models

    Args:
        ax: Matplotlib axis
        residuals: Residual values

    Returns:
        int: Number of significant autocorrelations
    """
    # Plot ACF
    plot_acf(residuals, lags=Config.ACF_LAGS, ax=ax, alpha=Config.ALPHA)

    # Formatting
    ax.set_xlabel('Lag', fontweight='bold')
    ax.set_ylabel('Autocorrelation', fontweight='bold')
    ax.set_title(
        'Autocorrelation Function of Residuals\n'
        '(Should show no significant correlations - stay within confidence bands)',
        fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Count significant autocorrelations
    from statsmodels.tsa.stattools import acf
    acf_values = acf(residuals, nlags=Config.ACF_LAGS, alpha=Config.ALPHA)

    # acf returns (acf_values, confidence_intervals)
    if isinstance(acf_values, tuple):
        acf_vals = acf_values[0][1:]  # Skip lag 0
        conf_int = acf_values[1][1:]  # Skip lag 0
        # Count how many exceed confidence interval
        significant = np.sum(
            (acf_vals < conf_int[:, 0]) | (acf_vals > conf_int[:, 1]))
    else:
        # Fallback: use 1.96/sqrt(n) as approximate confidence bound
        conf_bound = 1.96 / np.sqrt(len(residuals))
        acf_vals = acf_values[1:]  # Skip lag 0
        significant = np.sum(np.abs(acf_vals) > conf_bound)

    return significant


def plot_qq(ax, residuals: np.ndarray) -> None:
    """
    Create Q-Q plot (Quantile-Quantile plot).

    WHAT THIS SHOWS:
    ================
    Compares residual distribution to theoretical normal distribution.

    Q-Q PLOT EXPLAINED:
    ===================
    - X-axis: Theoretical quantiles (what normal distribution would give)
    - Y-axis: Sample quantiles (actual residuals)
    - Diagonal line: Perfect match to normal distribution

    INTERPRETATION:
    - Points on diagonal = GOOD (residuals are normal)
    - Points above diagonal = residuals larger than expected
    - Points below diagonal = residuals smaller than expected
    - S-curve = heavy tails (more extreme values)
    - Deviation at ends = outliers

    WHY NORMALITY MATTERS:
    ======================
    - Many statistical tests assume normal residuals
    - Confidence intervals require normality
    - Non-normal residuals suggest:
      * Outliers in data
      * Wrong model family
      * Need for transformation (log, sqrt, etc.)

    Args:
        ax: Matplotlib axis
        residuals: Residual values
    """
    stats.probplot(residuals, dist="norm", plot=ax)

    # Formatting
    ax.set_title('Q-Q Plot: Residuals vs. Normal Distribution\n'
                 '(Points should follow the diagonal line closely)',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Make diagonal line more visible
    line = ax.get_lines()[0]
    line.set_color('red')
    line.set_linewidth(2)


def create_diagnostic_plots(
        residuals: pd.Series,
        model_name: str,
        dataset_name: str) -> int:
    """
    Create comprehensive 4-panel diagnostic plot.

    Args:
        residuals: Residuals with datetime index
        model_name: Name of the model
        dataset_name: 'voltage' or 'missions'

    Returns:
        int: Number of significant autocorrelations
    """
    print_section("CREATING DIAGNOSTIC PLOTS")

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=Config.FIGSIZE, dpi=Config.DPI)

    # Overall title
    fig.suptitle(
        f'Residual Analysis: {model_name} ({
            dataset_name.upper()} Dataset)',
        fontsize=16,
        fontweight='bold',
        y=0.995)

    # Subplot 1: Histogram
    print("  Creating subplot 1/4: Residual distribution...")
    plot_residual_distribution(axes[0, 0], residuals.values)

    # Subplot 2: Residuals over time
    print("  Creating subplot 2/4: Residuals over time...")
    plot_residuals_over_time(axes[0, 1], residuals)

    # Subplot 3: ACF
    print("  Creating subplot 3/4: Autocorrelation function...")
    significant_acf = plot_residual_acf(axes[1, 0], residuals.values)

    # Subplot 4: Q-Q plot
    print("  Creating subplot 4/4: Q-Q plot...")
    plot_qq(axes[1, 1], residuals.values)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_file = Config.GRAPHICS_DIR / f'{dataset_name}_residual_analysis.png'
    plt.savefig(output_file, dpi=Config.DPI, bbox_inches='tight')
    print(f"\n✓ Saved diagnostic plots: {output_file}")

    plt.close()

    return significant_acf


# ==============================================================================
# SECTION 6: STATISTICAL TESTS
# ==============================================================================


def test_residual_properties(
        residuals: np.ndarray,
        significant_acf: int) -> Dict:
    """
    Perform statistical tests on residuals.

    TESTS PERFORMED:
    ================

    1. SHAPIRO-WILK TEST (Normality):
       - Null hypothesis (H0): Residuals are normally distributed
       - Alternative (H1): Residuals are NOT normally distributed
       - p > 0.05: Fail to reject H0 → Residuals are approximately normal ✓
       - p ≤ 0.05: Reject H0 → Residuals are NOT normal ✗

    2. AUTOCORRELATION CHECK:
       - Count significant lags in ACF
       - 0 significant lags = Good (independent errors) ✓
       - Many significant lags = Bad (temporal dependence remains) ✗

    Args:
        residuals: Residual values
        significant_acf: Number of significant autocorrelations

    Returns:
        dict: Test results
    """
    print_section("STATISTICAL TESTS")

    results = {}

    # 1. Shapiro-Wilk test for normality
    print("\n1. SHAPIRO-WILK TEST (Normality):")
    print("   H0: Residuals are normally distributed")
    print("   H1: Residuals are NOT normally distributed")

    # Limit sample size for Shapiro-Wilk (it's sensitive to large samples)
    sample_size = min(5000, len(residuals))
    if len(residuals) > sample_size:
        residuals_sample = np.random.choice(
            residuals, size=sample_size, replace=False)
        print(f"   (Using random sample of {sample_size} residuals)")
    else:
        residuals_sample = residuals

    stat, p_value = shapiro(residuals_sample)
    results['shapiro_stat'] = stat
    results['shapiro_p'] = p_value

    print(f"\n   Test statistic: {stat:.6f}")
    print(f"   p-value:        {p_value:.6f}")

    if p_value > Config.ALPHA:
        print(
            f"   ✓ Result: p > {
                Config.ALPHA} → Residuals are approximately normal")
        print("     This is GOOD - supports model assumptions")
        results['is_normal'] = True
    else:
        print(
            f"   ✗ Result: p ≤ {
                Config.ALPHA} → Residuals are NOT normally distributed")
        print("     Consider: transformations (log, sqrt) or robust methods")
        results['is_normal'] = False

    # 2. Autocorrelation check
    print("\n2. AUTOCORRELATION CHECK:")
    print(
        f"   Number of significant autocorrelations: {significant_acf} out of {
            Config.ACF_LAGS}")

    results['significant_acf'] = significant_acf

    if significant_acf == 0:
        print("   ✓ Result: No significant autocorrelations")
        print("     Residuals are independent - model captured temporal structure well")
        results['has_autocorr'] = False
    elif significant_acf <= 3:
        print("   ⚠ Result: Few significant autocorrelations")
        print("     Minor temporal dependence remains - consider additional lag features")
        results['has_autocorr'] = True
    else:
        print("   ✗ Result: Many significant autocorrelations")
        print("     Substantial temporal dependence - model missing temporal patterns")
        print("     Recommendation: Use ARIMA, add more lags, or try sequence models")
        results['has_autocorr'] = True

    return results


# ==============================================================================
# SECTION 7: INTERPRETATION
# ==============================================================================


def generate_interpretation(
        residuals: pd.Series,
        test_results: Dict,
        model_name: str) -> None:
    """
    Generate comprehensive written interpretation.

    This provides actionable insights about:
    - What the residual patterns mean
    - Whether model assumptions are met
    - What improvements could be made

    Args:
        residuals: Residuals with datetime index
        test_results: Results from statistical tests
        model_name: Name of the model
    """
    print_section("RESIDUAL ANALYSIS SUMMARY")

    res_values = residuals.values
    mean_res = np.mean(res_values)
    std_res = np.std(res_values)

    print(f"\nModel: {model_name}")
    print("=" * 100)

    # 1. Distribution
    print("\n1. DISTRIBUTION:")
    if test_results['is_normal']:
        print("   ✓ NORMAL - Residuals follow approximately normal distribution")
        print("     → Model assumptions are met")
        print("     → Statistical inference is valid")
    else:
        # Check skewness
        from scipy.stats import skew, kurtosis
        skewness = skew(res_values)
        kurt = kurtosis(res_values)

        if abs(skewness) > 0.5:
            direction = "right" if skewness > 0 else "left"
            print(f"   ✗ SKEWED ({direction}) - Distribution is not symmetric")
            print(f"     Skewness: {skewness:.3f}")
            print("     → Model has directional bias")
            print("     → Consider: transformation or robust methods")

        if abs(kurt) > 1:
            print("   ✗ NON-NORMAL KURTOSIS - Heavy tails detected")
            print(f"     Excess kurtosis: {kurt:.3f}")
            print("     → More extreme errors than expected")
            print("     → Possible outliers in data")

    # 2. Mean
    print(f"\n2. MEAN OF RESIDUALS: {mean_res:.6f}")
    if abs(mean_res) < 0.01 * std_res:
        print("   ✓ EXCELLENT - Mean is very close to 0")
        print("     → No systematic bias")
    elif abs(mean_res) < 0.1 * std_res:
        print("   ✓ GOOD - Mean is close to 0")
        print("     → Minimal systematic bias")
    else:
        bias_direction = "under-prediction" if mean_res > 0 else "over-prediction"
        print("   ⚠ BIAS DETECTED - Mean deviates from 0")
        print(f"     → Model shows systematic {bias_direction}")
        print("     → Consider: recalibration or bias correction")

    # 3. Temporal patterns
    print("\n3. TEMPORAL PATTERNS:")
    x_numeric = np.arange(len(res_values))
    slope, _, r_value, p_value, _ = stats.linregress(x_numeric, res_values)

    if abs(r_value) < 0.1:
        print("   ✓ NO TREND - Residuals are stable over time")
        print("     → Model performance is consistent")
    else:
        trend_direction = "upward" if slope > 0 else "downward"
        print(
            f"   ✗ TREND DETECTED - {trend_direction.upper()} trend in residuals")
        print(f"     Slope: {slope:.6f}, R²: {r_value**2:.4f}")
        if slope > 0:
            print("     → Model increasingly under-predicts over time")
        else:
            print("     → Model increasingly over-predicts over time")
        print("     → Recommendation: Add time-based features or use adaptive models")

    # 4. Heteroscedasticity
    print("\n4. HETEROSCEDASTICITY:")
    # Split into three periods and compare variances
    n = len(res_values)
    third = n // 3
    var1 = np.var(res_values[:third])
    var2 = np.var(res_values[third:2 * third])
    var3 = np.var(res_values[2 * third:])

    max_var = max(var1, var2, var3)
    min_var = min(var1, var2, var3)
    var_ratio = max_var / min_var if min_var > 0 else np.inf

    if var_ratio < 2:
        print("   ✓ NOT DETECTED - Error variance is relatively constant")
        print("     → Homoscedastic residuals (good)")
    else:
        print("   ✗ DETECTED - Error variance changes over time")
        print(f"     Variance ratio: {var_ratio:.2f}x")
        print("     → Error magnitude depends on time period")
        print("     → Recommendation: Use weighted regression or variance-stabilizing transformation")

    # 5. Autocorrelation
    print("\n5. AUTOCORRELATION:")
    if not test_results['has_autocorr']:
        print("   ✓ ABSENT - Residuals are independent")
        print("     → Model captured temporal dependencies well")
    else:
        print(
            f"   ✗ PRESENT - {test_results['significant_acf']} significant lags detected")
        print("     → Model hasn't captured all temporal structure")
        print("     → Implications:")
        print("       - Predictions are not utilizing all available information")
        print("       - Forecast intervals may be too narrow")
        print("     → Recommendation:")
        print("       - Add more lag features")
        print("       - Use ARIMA/SARIMA models")
        print("       - Try sequence models (LSTM)")

    # 6. Overall assessment
    print("\n6. OVERALL ASSESSMENT:")

    # Count issues
    issues = []
    if not test_results['is_normal']:
        issues.append("non-normal distribution")
    if abs(mean_res) > 0.1 * std_res:
        issues.append("systematic bias")
    if abs(r_value) >= 0.1:
        issues.append("temporal trend")
    if var_ratio >= 2:
        issues.append("heteroscedasticity")
    if test_results['has_autocorr']:
        issues.append("autocorrelation")

    if len(issues) == 0:
        print("   ✅ EXCELLENT")
        print("     All diagnostic checks passed")
        print("     Model residuals meet ideal properties")
        print("     → Ready for deployment")
    elif len(issues) <= 2:
        print("   ✓ GOOD")
        print(f"     Minor issues detected: {', '.join(issues)}")
        print("     Model is functional but could be improved")
        print("     → Consider refinements but model is usable")
    else:
        print("   ⚠ NEEDS IMPROVEMENT")
        print(f"     Multiple issues detected: {', '.join(issues)}")
        print("     Model has systematic deficiencies")
        print("     → Recommendations:")
        for issue in issues:
            if issue == "non-normal distribution":
                print("       • Apply transformations (log, Box-Cox)")
            elif issue == "systematic bias":
                print("       • Add bias correction or recalibrate")
            elif issue == "temporal trend":
                print("       • Add time-based features or trend component")
            elif issue == "heteroscedasticity":
                print("       • Use weighted regression or robust methods")
            elif issue == "autocorrelation":
                print("       • Add lag features or use ARIMA/sequence models")

    print("\n" + "=" * 100)

    print("\n💡 KEY INSIGHT:")
    print("   Patterns in residuals indicate systematic model deficiencies.")
    print("   These can often be addressed with:")
    print("   - Better feature engineering (more lags, transformations)")
    print("   - Different model architectures (ARIMA, LSTM)")
    print("   - Data preprocessing (outlier handling, variance stabilization)")
    print("   - Ensemble methods (combining multiple models)")


# ==============================================================================
# SECTION 8: MAIN EXECUTION
# ==============================================================================


def main(dataset_name='voltage'):
    """
    Main execution pipeline for residual analysis.

    Args:
        dataset_name: 'voltage' or 'missions'
    """
    print("=" * 100)
    print(f"TIME SERIES RESIDUAL ANALYSIS: {dataset_name.upper()}")
    print("=" * 100)

    try:
        # Ensure output directory exists
        Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load data
        y_test, predictions, model_name, metrics = load_test_data_and_predictions(
            dataset_name)

        # 2. Calculate residuals
        residuals = calculate_residuals(y_test, predictions)

        # 3. Create diagnostic plots
        significant_acf = create_diagnostic_plots(
            residuals, model_name, dataset_name)

        # 4. Perform statistical tests
        test_results = test_residual_properties(
            residuals.values, significant_acf)

        # 5. Generate interpretation
        generate_interpretation(residuals, test_results, model_name)

        # Final message
        print("\n" + "=" * 100)
        print("✅ RESIDUAL ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 100)
        print(
            f"\nOutput saved to: {
                Config.GRAPHICS_DIR /
                f'{dataset_name}_residual_analysis.png'}")
        print("\nNext steps:")
        print("  1. Review diagnostic plots for patterns")
        print("  2. Consider recommendations from interpretation")
        print("  3. Implement improvements if needed")
        print("=" * 100)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import sys

    # Allow user to specify dataset via command line
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
        if dataset not in ['voltage', 'missions', 'both']:
            print(
                f"Error: Invalid dataset '{dataset}'. Choose 'voltage', 'missions', or 'both'")
            sys.exit(1)
    else:
        dataset = 'both'  # Default: analyze both

    # Process datasets
    if dataset == 'both':
        datasets_to_analyze = ['voltage', 'missions']
    else:
        datasets_to_analyze = [dataset]

    # Analyze each dataset
    for ds in datasets_to_analyze:
        print(f"\n{'=' * 100}")
        print(f"PROCESSING DATASET: {ds.upper()}")
        print(f"{'=' * 100}")
        main(ds)
