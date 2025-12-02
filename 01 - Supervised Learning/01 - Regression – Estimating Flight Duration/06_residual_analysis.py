#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLIGHT TELEMETRY - RESIDUAL ANALYSIS
==============================================================================

Purpose: Perform comprehensive residual analysis for the best performing model

This script:
1. Identifies the best model and loads its predictions
2. Calculates residuals (actual - predicted)
3. Computes descriptive statistics of residuals
4. Creates diagnostic plots:
   a) Histogram with KDE of residuals (checks normality assumption)
   b) Residuals vs Predicted values (checks homoscedasticity and patterns)
5. Interprets results to identify model weaknesses
6. Provides recommendations for model improvement

Residual Analysis Purpose:
- Validate model assumptions (normality, homoscedasticity, independence)
- Detect systematic patterns indicating model misspecification
- Identify outliers and influential observations
- Assess prediction reliability across different ranges
- Guide model improvement strategies

Key Assumptions for Linear Regression:
1. Linearity: Relationship between features and target is linear
2. Independence: Observations are independent
3. Homoscedasticity: Constant variance of residuals
4. Normality: Residuals are normally distributed (for inference)

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
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_palette("Set2")
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Set random seed for reproducibility
np.random.seed(42)


# Configuration Constants
class Config:
    """Residual analysis configuration parameters."""
    # Directories
    OUTPUT_DIR = Path("outputs")
    ARTIFACTS_DIR = OUTPUT_DIR / "models"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    IMAGES_DIR = OUTPUT_DIR / "graphics"

    # Input files
    BEST_MODEL_FILE = ARTIFACTS_DIR / "best_model.txt"
    Y_TEST_FILE = ARTIFACTS_DIR / "y_test.pkl"

    # Model mapping
    MODEL_TO_FILE = {
        'Simple Linear Regression': 'preds_linear_simple.csv',
        'Multiple Linear Regression': 'preds_linear_multiple.csv',
        'Ridge Regression': 'preds_ridge.csv',
        'Lasso Regression': 'preds_lasso.csv',
        'Polynomial Regression (degree=2)': 'preds_polynomial_deg2.csv'
    }

    # Plot settings
    DPI = 300
    HIST_BINS = 30
    SCATTER_SIZE = 50
    SCATTER_ALPHA = 0.5


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 120)
    print(title)
    print(char * 120)


def print_subsection(title: str) -> None:
    """Print formatted subsection header."""
    print(f"\n{title}")
    print("-" * 120)


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_best_model_data() -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Load best model predictions and ground truth.

    Returns:
        Tuple[str, np.ndarray, np.ndarray]: (model_name, y_actual, y_predicted)

    Raises:
        SystemExit: If required files not found
        ValueError: If model name cannot be parsed
    """
    print_section("[SECTION 1] IDENTIFY BEST MODEL AND LOAD DATA")

    # Ensure images directory exists
    Config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Images directory verified/created: {Config.IMAGES_DIR}")
    print()

    # Read best model information
    print(f"Reading best model information from: {Config.BEST_MODEL_FILE}")

    if not Config.BEST_MODEL_FILE.exists():
        print(f"✗ ERROR: Best model file not found: {Config.BEST_MODEL_FILE}")
        exit(1)

    with open(Config.BEST_MODEL_FILE, 'r') as f:
        best_model_content = f.read()

    print("✓ Best model file loaded")
    print()

    # Parse model name from the file
    best_model_name = None
    for line in best_model_content.split('\n'):
        if line.startswith('Model Name:'):
            best_model_name = line.split('Model Name:')[1].strip()
            break

    if best_model_name is None:
        raise ValueError("Could not parse model name from best model file")

    print(f"Best model identified: {best_model_name}")
    print()

    # Verify model name is valid
    if best_model_name not in Config.MODEL_TO_FILE:
        raise ValueError(f"Unknown model name: {best_model_name}")

    predictions_file = Config.MODEL_TO_FILE[best_model_name]
    print(f"Predictions file: {predictions_file}")
    print()

    # Load test target (ground truth)
    print("Loading test target (ground truth)...")
    if not Config.Y_TEST_FILE.exists():
        print(f"✗ ERROR: Ground truth not found: {Config.Y_TEST_FILE}")
        exit(1)

    with open(Config.Y_TEST_FILE, 'rb') as f:
        y_test = pickle.load(f)

    print(f"✓ y_test loaded from: {Config.Y_TEST_FILE}")
    print(f"  Shape: {y_test.shape}")
    print()

    # Load predictions for best model
    print("Loading predictions for best model...")
    predictions_path = Config.PREDICTIONS_DIR / predictions_file

    if not predictions_path.exists():
        print(f"✗ ERROR: Predictions not found: {predictions_path}")
        exit(1)

    predictions_df = pd.read_csv(predictions_path)
    print(f"✓ Predictions loaded from: {predictions_path}")
    print(f"  Shape: {predictions_df.shape}")
    print()

    # Extract actual and predicted values
    y_actual = predictions_df['y_true'].values
    y_predicted = predictions_df['y_pred'].values

    # Verify consistency
    if not np.allclose(y_actual, y_test):
        print("  WARNING: y_true in predictions file does not match y_test!")
    else:
        print("  ✓ Data consistency verified")
    print()

    return best_model_name, y_actual, y_predicted


# ==============================================================================
# SECTION 4: RESIDUAL CALCULATION
# ==============================================================================


def calculate_residuals(
        y_actual: np.ndarray,
        y_predicted: np.ndarray) -> np.ndarray:
    """
    Calculate residuals (actual - predicted).

    Args:
        y_actual: Ground truth values
        y_predicted: Predicted values

    Returns:
        np.ndarray: Residuals
    """
    print_section("[SECTION 2] CALCULATE RESIDUALS")

    print("RESIDUAL DEFINITION:")
    print("  Residual = Actual - Predicted")
    print("  • Positive residual: Model UNDER-predicted (actual > predicted)")
    print("  • Negative residual: Model OVER-predicted (actual < predicted)")
    print("  • Zero residual: Perfect prediction")
    print()

    residuals = y_actual - y_predicted

    print(f"✓ Residuals calculated for {len(residuals)} test samples")
    print()

    return residuals


# ==============================================================================
# SECTION 5: STATISTICAL ANALYSIS
# ==============================================================================


def analyze_residuals(residuals: np.ndarray, y_predicted: np.ndarray) -> Dict:
    """
    Perform comprehensive statistical analysis of residuals.

    Args:
        residuals: Residual values
        y_predicted: Predicted values

    Returns:
        Dict: Dictionary with statistical results
    """
    print_section("[SECTION 3] DESCRIPTIVE STATISTICS OF RESIDUALS")

    print("Computing comprehensive residual statistics...")
    print()

    # Basic statistics
    mean_residual = residuals.mean()
    median_residual = np.median(residuals)
    std_residual = residuals.std()
    min_residual = residuals.min()
    max_residual = residuals.max()

    print("CENTRAL TENDENCY:")
    print(f"  Mean:   {mean_residual:+.4f} minutes")
    print(f"  Median: {median_residual:+.4f} minutes")
    print()

    # Interpret mean residual
    print("INTERPRETATION OF MEAN RESIDUAL:")
    if abs(mean_residual) < 0.1:
        print("  → Excellent: Mean ≈ 0 indicates unbiased predictions")
    elif abs(mean_residual) < 0.5:
        print("  → Good: Small bias, predictions are approximately centered")
    elif abs(mean_residual) < 1.0:
        print("  → Acceptable: Slight bias present")
        if mean_residual > 0:
            print(
                f"     Model tends to UNDER-predict by {mean_residual:.2f} minutes on average")
        else:
            print(
                f"     Model tends to OVER-predict by {abs(mean_residual):.2f} minutes on average")
    else:
        print("  → Concerning: Significant bias detected")
        if mean_residual > 0:
            print(
                f"     Model systematically UNDER-predicts by {mean_residual:.2f} minutes")
        else:
            print(
                f"     Model systematically OVER-predicts by {abs(mean_residual):.2f} minutes")
    print()

    print("DISPERSION:")
    print(f"  Standard Deviation: {std_residual:.4f} minutes")
    print(f"  Range: [{min_residual:+.2f}, {max_residual:+.2f}] minutes")
    print()

    # Quartiles and IQR
    q1 = np.percentile(residuals, 25)
    q2 = np.percentile(residuals, 50)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1

    print("QUARTILES:")
    print(f"  Q1 (25%): {q1:+.4f} minutes")
    print(f"  Q2 (50%): {q2:+.4f} minutes (median)")
    print(f"  Q3 (75%): {q3:+.4f} minutes")
    print(f"  IQR:      {iqr:.4f} minutes")
    print()

    # Outlier detection
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_mask = (residuals < lower_fence) | (residuals > upper_fence)
    num_outliers = outliers_mask.sum()
    pct_outliers = 100.0 * num_outliers / len(residuals)

    print("OUTLIER DETECTION (1.5×IQR rule):")
    print(f"  Lower fence: {lower_fence:+.4f} minutes")
    print(f"  Upper fence: {upper_fence:+.4f} minutes")
    print(f"  Outliers:    {num_outliers} ({pct_outliers:.1f}%)")
    print()

    # Shape statistics
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    print("DISTRIBUTION SHAPE:")
    print(f"  Skewness: {skewness:+.4f}", end="")
    if abs(skewness) < 0.5:
        print(" → Approximately symmetric")
    elif skewness > 0.5:
        print(" → Right-skewed (positive tail longer)")
    else:
        print(" → Left-skewed (negative tail longer)")

    print(f"  Kurtosis: {kurtosis:+.4f}", end="")
    if abs(kurtosis) < 0.5:
        print(" → Normal-like tails")
    elif kurtosis > 0.5:
        print(" → Heavy tails (more outliers than normal)")
    else:
        print(" → Light tails (fewer outliers than normal)")
    print()

    # Normality test
    _, normality_p = stats.shapiro(
        residuals[:5000] if len(residuals) > 5000 else residuals)

    print("NORMALITY TEST (Shapiro-Wilk):")
    print(f"  p-value: {normality_p:.6f}")
    if normality_p > 0.05:
        print("  → Residuals appear normally distributed (p > 0.05)")
    else:
        print("  → Residuals deviate from normality (p ≤ 0.05)")
    print()

    # Heteroscedasticity analysis
    n_bins = 10
    pred_bins = pd.qcut(y_predicted, q=n_bins, duplicates='drop')
    residuals_series = pd.Series(residuals)
    variance_by_bin = residuals_series.groupby(pred_bins).var()
    variance_ratio = variance_by_bin.max() / variance_by_bin.min()

    print("HETEROSCEDASTICITY ANALYSIS:")
    print(f"  Variance ratio: {variance_ratio:.2f}")
    if variance_ratio < 1.5:
        print("  → Homoscedastic: Constant variance across prediction range")
    elif variance_ratio < 2.5:
        print("  → Mild heteroscedasticity detected")
    else:
        print("  → Strong heteroscedasticity detected")
    print()

    # Create statistics dictionary
    stats_dict = {
        'mean': mean_residual,
        'median': median_residual,
        'std': std_residual,
        'min': min_residual,
        'max': max_residual,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'normality_p': normality_p,
        'n_outliers': num_outliers,
        'pct_outliers': pct_outliers,
        'variance_ratio': variance_ratio
    }

    return stats_dict


# ==============================================================================
# SECTION 6: RESIDUAL VISUALIZATION - HISTOGRAM
# ==============================================================================


def create_residual_histogram(
        residuals: np.ndarray,
        model_name: str,
        stats_dict: Dict) -> None:
    """
    Create histogram of residuals with normal distribution overlay.

    Args:
        residuals: Residual values
        model_name: Name of the model
        stats_dict: Dictionary with statistical results
    """
    print_section("[SECTION 4] CREATE RESIDUAL HISTOGRAM")

    print("Creating histogram with KDE and normal distribution overlay...")
    print()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram
    n, bins, patches = ax.hist(residuals, bins=Config.HIST_BINS, density=True,
                               alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)

    # KDE
    kde = gaussian_kde(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(
        x_range,
        kde(x_range),
        'r-',
        linewidth=2,
        label='KDE (actual distribution)')

    # Normal distribution overlay
    mean_res = stats_dict['mean']
    std_res = stats_dict['std']
    normal_dist = stats.norm(mean_res, std_res)
    ax.plot(x_range, normal_dist.pdf(x_range), 'g--', linewidth=2,
            label='Normal distribution (theoretical)')

    # Vertical line at zero
    ax.axvline(
        0,
        color='black',
        linestyle='--',
        linewidth=1.5,
        alpha=0.7,
        label='Zero residual')

    # Vertical line at mean
    ax.axvline(
        mean_res,
        color='red',
        linestyle='-',
        linewidth=1.5,
        alpha=0.7,
        label='Mean residual')

    # Labels and title
    ax.set_xlabel('Residual (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Residual Distribution - {model_name}',
        fontsize=14,
        fontweight='bold',
        pad=20)

    # Legend
    ax.legend(loc='upper right', frameon=True, shadow=True)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add statistics text box
    textstr = '\n'.join([
        f"Mean: {mean_res:+.4f} min",
        f"Std: {std_res:.4f} min",
        f"Skewness: {stats_dict['skewness']:+.4f}",
        f"Kurtosis: {stats_dict['kurtosis']:+.4f}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save plots
    hist_png_path = Config.IMAGES_DIR / 'residuals_hist.png'

    plt.savefig(hist_png_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()

    print("✓ Histogram saved:")
    print(f"  PNG: {hist_png_path}")
    print()


# ==============================================================================
# SECTION 7: RESIDUAL VISUALIZATION - SCATTER PLOT
# ==============================================================================


def create_residual_scatter(residuals: np.ndarray, y_predicted: np.ndarray,
                            model_name: str, stats_dict: Dict) -> None:
    """
    Create scatter plot of residuals vs predicted values.

    Args:
        residuals: Residual values
        y_predicted: Predicted values
        model_name: Name of the model
        stats_dict: Dictionary with statistical results
    """
    print_section("[SECTION 5] CREATE RESIDUALS VS PREDICTED PLOT")

    print("Creating residuals vs predicted values scatter plot...")
    print()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Scatter plot
    ax.scatter(
        y_predicted,
        residuals,
        alpha=Config.SCATTER_ALPHA,
        s=Config.SCATTER_SIZE,
        color='steelblue',
        edgecolors='darkblue',
        linewidth=0.5)

    # Zero line
    ax.axhline(0, color='black', linestyle='--', linewidth=2,
               label='Zero residual (perfect prediction)')

    # Mean line
    mean_res = stats_dict['mean']
    ax.axhline(mean_res, color='red', linestyle='-', linewidth=2,
               label=f'Mean residual ({mean_res:+.4f} min)')

    # Standard deviation bands
    std_res = stats_dict['std']
    ax.axhline(mean_res + std_res, color='orange', linestyle=':',
               linewidth=1.5, label=f'±1 SD ({std_res:.4f} min)')
    ax.axhline(
        mean_res - std_res,
        color='orange',
        linestyle=':',
        linewidth=1.5)

    # Moving average trend line
    sorted_indices = np.argsort(y_predicted)
    y_pred_sorted = y_predicted[sorted_indices]
    residuals_sorted = residuals[sorted_indices]

    # Apply Savitzky-Golay filter for smoothing
    window_length = min(
        51, len(residuals_sorted) if len(residuals_sorted) %
        2 == 1 else len(residuals_sorted) - 1)
    if window_length >= 5:
        trend = savgol_filter(
            residuals_sorted,
            window_length=window_length,
            polyorder=3)
        ax.plot(y_pred_sorted, trend, 'g-', linewidth=2.5,
                label='Smoothed trend (non-linearity indicator)')

    # Labels and title
    ax.set_xlabel('Predicted Values (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals (minutes)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Residuals vs Predicted Values - {model_name}',
        fontsize=14,
        fontweight='bold',
        pad=20)

    # Legend
    ax.legend(loc='best', frameon=True, shadow=True)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add statistics text box
    textstr = '\n'.join([
        f"Mean: {mean_res:+.4f} min",
        f"Std: {std_res:.4f} min",
        f"Outliers: {stats_dict['n_outliers']} ({stats_dict['pct_outliers']:.1f}%)"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save plots
    scatter_png_path = Config.IMAGES_DIR / 'residuals_vs_predictions.png'

    plt.savefig(scatter_png_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()

    print("✓ Scatter plot saved:")
    print(f"  PNG: {scatter_png_path}")
    print()


# ==============================================================================
# SECTION 8: INTERPRETATION
# ==============================================================================


def interpret_residuals(
        stats_dict: Dict,
        y_predicted: np.ndarray,
        residuals: np.ndarray) -> None:
    """
    Provide comprehensive interpretation of residual analysis.

    Args:
        stats_dict: Dictionary with statistical results
        y_predicted: Predicted values
        residuals: Residual values
    """
    print_section("[SECTION 6] COMPREHENSIVE INTERPRETATION")

    print("DIAGNOSTIC ASSESSMENT OF MODEL ASSUMPTIONS:")
    print()

    # 1. Bias
    print("1. BIAS (SYSTEMATIC ERROR):")
    print()
    mean_res = stats_dict['mean']

    if abs(mean_res) < 0.1:
        print("   ✓ EXCELLENT: No bias detected")
        print("      Mean residual ≈ 0, indicating unbiased predictions")
    elif abs(mean_res) < 0.5:
        print("   ✓ GOOD: Minimal bias")
        print("      Predictions are approximately centered")
    elif abs(mean_res) < 1.0:
        print("   ⚠ ACCEPTABLE: Slight bias present")
        if mean_res > 0:
            print(
                f"      Model tends to UNDER-predict by {mean_res:.2f} minutes on average")
        else:
            print(
                f"      Model tends to OVER-predict by {abs(mean_res):.2f} minutes on average")
    else:
        print("   ⚠ CONCERNING: Significant bias detected")
        if mean_res > 0:
            print(
                f"      Model systematically UNDER-predicts by {mean_res:.2f} minutes")
        else:
            print(
                f"      Model systematically OVER-predicts by {abs(mean_res):.2f} minutes")
        print()
        print("   IMPLICATIONS:")
        print("      • Predictions are consistently off in one direction")
        print("      • May indicate missing important features")
        print("      • Could suggest model misspecification")
        print()
        print("   POTENTIAL SOLUTIONS:")
        print("      • Add relevant features")
        print("      • Consider non-linear transformations")
        print("      • Try different model architectures")
    print()

    # 2. Heteroscedasticity
    print("2. HETEROSCEDASTICITY (VARIANCE CONSISTENCY):")
    print()
    variance_ratio = stats_dict['variance_ratio']

    if variance_ratio < 1.5:
        print("   ✓ EXCELLENT: Homoscedastic residuals")
        print("      Constant variance across all prediction levels")
        print("      Regression assumptions satisfied")
    elif variance_ratio < 2.5:
        print("   ⚠ MODERATE: Mild heteroscedasticity detected")
        print(f"      Variance ratio = {variance_ratio:.2f}")
        print()
        print("   IMPLICATIONS:")
        print("      • Error variance increases with prediction level")
        print("      • Standard errors may be underestimated")
        print("      • Confidence intervals less reliable")
        print()
        print("   POTENTIAL SOLUTIONS:")
        print("      • Log transformation: y_new = log(y)")
        print("      • Square root transformation: y_new = sqrt(y)")
        print("      • Weighted least squares regression")
    else:
        print("   ⚠ CONCERNING: Strong heteroscedasticity detected")
        print(f"      Variance ratio = {variance_ratio:.2f}")
        print()
        print("   IMPLICATIONS:")
        print("      • Severe violation of regression assumptions")
        print("      • Unreliable confidence intervals")
        print("      • Hypothesis tests invalid")
        print()
        print("   POTENTIAL SOLUTIONS:")
        print("      • Transform target variable (log, Box-Cox)")
        print("      • Use robust regression methods")
        print("      • Consider different models for different ranges")
    print()

    # 3. Non-linearity
    print("3. NON-LINEARITY DETECTION:")
    print()

    # Calculate trend deviation
    sorted_indices = np.argsort(y_predicted)
    y_predicted[sorted_indices]
    residuals_sorted = residuals[sorted_indices]

    window_length = min(
        51, len(residuals_sorted) if len(residuals_sorted) %
        2 == 1 else len(residuals_sorted) - 1)
    if window_length >= 5:
        trend = savgol_filter(
            residuals_sorted,
            window_length=window_length,
            polyorder=3)
        max_mean_deviation = np.max(np.abs(trend))

        if max_mean_deviation < 0.5:
            print("   ✓ GOOD: No significant non-linear patterns detected")
            print("      Residuals randomly distributed around zero")
            print("      Linear model appears appropriate")
        elif max_mean_deviation < 1.5:
            print("   ⚠ MODERATE: Some non-linear patterns detected")
            print(
                f"      Max deviation from zero trend: {
                    max_mean_deviation:.2f} minutes")
            print()
            print("   POTENTIAL SOLUTIONS:")
            print("      • Add polynomial features (X², X³)")
            print("      • Include interaction terms (X₁×X₂)")
            print("      • Consider splines or GAMs")
        else:
            print("   ⚠ CONCERNING: Strong non-linear patterns detected")
            print(
                f"      Max deviation from zero trend: {
                    max_mean_deviation:.2f} minutes")
            print()
            print("   IMPLICATIONS:")
            print("      • Linear model inadequate for this relationship")
            print("      • Systematic prediction errors at certain ranges")
            print()
            print("   POTENTIAL SOLUTIONS:")
            print("      • Polynomial regression")
            print("      • Tree-based models (Random Forest, Gradient Boosting)")
            print("      • Neural networks")
    print()

    # 4. Normality
    print("4. NORMALITY OF RESIDUALS:")
    print()
    skewness = stats_dict['skewness']
    kurtosis = stats_dict['kurtosis']

    if abs(skewness) < 0.5 and abs(kurtosis) < 1.0:
        print("   ✓ GOOD: Residuals are approximately normally distributed")
        print("      Skewness and kurtosis within acceptable range")
        print("      Confidence intervals and hypothesis tests are valid")
    elif abs(skewness) < 1.0 and abs(kurtosis) < 2.0:
        print("   ⚠ MODERATE: Residuals show some deviation from normality")
        print("      Not critical for prediction, but affects inference")
        print()
        print("   IMPLICATIONS:")
        print("      • Predictions remain unbiased")
        print("      • Confidence intervals may be approximate")
        print("      • For large samples, Central Limit Theorem helps")
    else:
        print("   ⚠ CONCERNING: Residuals deviate substantially from normality")
        print()
        print("   IMPLICATIONS:")
        print("      • Predictions still unbiased, but...")
        print("      • Confidence intervals unreliable")
        print("      • Hypothesis tests may give incorrect p-values")
        print()
        print("   POTENTIAL CAUSES:")
        print("      • Heavy-tailed distribution (more extreme values than normal)")
        print("      • Skewed distribution (asymmetric errors)")
        print("      • Presence of outliers")
        print("      • Model misspecification")
        print()
        print("   POTENTIAL SOLUTIONS:")
        print("      • Transform target variable (Box-Cox, log, sqrt)")
        print("      • Use robust regression methods")
        print("      • Bootstrap for confidence intervals")
        print("      • Consider non-parametric methods")
    print()


# ==============================================================================
# SECTION 9: RECOMMENDATIONS
# ==============================================================================


def provide_recommendations() -> None:
    """Provide actionable recommendations for model improvement."""
    print_section("[SECTION 7] RECOMMENDATIONS FOR MODEL IMPROVEMENT")

    print("Based on residual analysis, here are prioritized recommendations:")
    print()

    print("1. TARGET TRANSFORMATION:")
    print("   Purpose: Address heteroscedasticity and non-normality")
    print()
    print("   Options:")
    print("     a) LOG TRANSFORMATION: y_new = log(y) or log1p(y)")
    print("        • Best when: Variance increases with y")
    print("        • Effect: Reduces right skewness, stabilizes variance")
    print("        • Note: Cannot use if y has zero or negative values")
    print()
    print("     b) SQUARE ROOT: y_new = sqrt(y)")
    print("        • Best when: Moderate heteroscedasticity")
    print("        • Effect: Less aggressive than log, handles zeros")
    print()
    print("     c) BOX-COX TRANSFORMATION: Finds optimal power transform")
    print("        • Best when: Unsure which transformation to use")
    print("        • Effect: Automatically selects best λ parameter")
    print()
    print("   Implementation:")
    print("     from sklearn.preprocessing import PowerTransformer")
    print("     pt = PowerTransformer(method='box-cox')  # or 'yeo-johnson' for negative values")
    print("     y_transformed = pt.fit_transform(y.reshape(-1, 1))")
    print()

    print("-" * 120)
    print()

    print("2. FEATURE ENGINEERING:")
    print("   Purpose: Capture non-linear relationships and interactions")
    print()
    print("   Strategies:")
    print("     a) POLYNOMIAL FEATURES:")
    print("        • Add X², X³ for individual features")
    print("        • Captures curved relationships")
    print()
    print("     b) INTERACTION TERMS:")
    print("        • Add X₁×X₂ for feature pairs")
    print("        • Captures combined effects (e.g., distance × cargo weight)")
    print()
    print("     c) DOMAIN-SPECIFIC FEATURES:")
    print("        • Distance/altitude ratio (flight efficiency)")
    print("        • Cargo density (weight per unit distance)")
    print("        • Categorical interactions (e.g., weather × season)")
    print()
    print("     d) BINNING/DISCRETIZATION:")
    print("        • Convert continuous features to categories")
    print("        • Captures non-linear effects without polynomial explosion")
    print()

    print("-" * 120)
    print()

    print("3. ROBUST REGRESSION METHODS:")
    print("   Purpose: Handle outliers and violations of assumptions")
    print()
    print("   Options:")
    print("     a) HUBER REGRESSION (sklearn.linear_model.HuberRegressor):")
    print("        • Less sensitive to outliers than OLS")
    print("        • Uses robust loss function")
    print()
    print("     b) RANSAC (RANdom SAmple Consensus):")
    print("        • Fits model to subset of inliers")
    print("        • Ignores outliers automatically")
    print()
    print("     c) QUANTILE REGRESSION:")
    print("        • Predict median (50th percentile) instead of mean")
    print("        • Or predict 90th percentile for conservative estimates")
    print()

    print("-" * 120)
    print()

    print("4. ADVANCED MODELING APPROACHES:")
    print("   Purpose: Capture complex patterns without manual feature engineering")
    print()
    print("   Recommended models:")
    print("     a) RANDOM FOREST:")
    print("        • Handles non-linearity automatically")
    print("        • Robust to outliers")
    print("        • Provides feature importance")
    print()
    print("     b) GRADIENT BOOSTING (XGBoost, LightGBM, CatBoost):")
    print("        • Often best predictive performance")
    print("        • Handles non-linearity and interactions")
    print("        • Requires hyperparameter tuning")
    print()
    print("     c) NEURAL NETWORKS:")
    print("        • Highly flexible, captures complex patterns")
    print("        • Requires larger datasets and careful tuning")
    print()

    print("-" * 120)
    print()

    print("5. MODEL VALIDATION IMPROVEMENTS:")
    print()
    print("   a) CROSS-VALIDATION:")
    print("      • Use K-fold (e.g., K=5 or 10) for robust performance estimate")
    print("      • Detect overfitting early")
    print()
    print("   b) PREDICTION INTERVALS:")
    print("      • Quantify uncertainty in predictions")
    print("      • Use bootstrap or quantile regression")
    print()
    print("   c) RESIDUAL MONITORING:")
    print("      • Track residual patterns over time")
    print("      • Detect model degradation (concept drift)")
    print()


# ==============================================================================
# SECTION 10: FINAL SUMMARY
# ==============================================================================


def print_final_summary(model_name: str, stats_dict: Dict) -> None:
    """
    Print final summary of residual analysis.

    Args:
        model_name: Name of the model
        stats_dict: Dictionary with statistical results
    """
    print_section("RESIDUAL ANALYSIS COMPLETED SUCCESSFULLY", "=")

    print("SUMMARY:")
    print()
    print(f"  Model: {model_name}")
    print(
        f"  Number of test samples: {
            len(stats_dict) if isinstance(
                stats_dict,
                np.ndarray) else 'N/A'}")
    print()
    print("RESIDUAL STATISTICS:")
    print(f"  • Mean: {stats_dict['mean']:+.4f} minutes (bias indicator)")
    print(
        f"  • Std:  {
            stats_dict['std']:.4f} minutes (prediction uncertainty)")
    print(
        f"  • Range: [{stats_dict['min']:+.2f}, {stats_dict['max']:+.2f}] minutes")
    print(
        f"  • Outliers: {
            stats_dict['n_outliers']} ({
            stats_dict['pct_outliers']:.1f}%)")
    print()

    print("FILES SAVED:")
    print(f"  • Histogram PNG: {Config.IMAGES_DIR / 'residuals_hist.png'}")
    print(
        f"  • Residuals plot PNG: {
            Config.IMAGES_DIR /
            'residuals_vs_predictions.png'}")
    print()

    print("KEY FINDINGS:")
    print()

    # Summary assessment
    issues_found = []

    if abs(stats_dict['mean']) > 1.0:
        issues_found.append("Significant bias detected")
    if stats_dict['variance_ratio'] > 3.0:
        issues_found.append("Heteroscedasticity present")
    if stats_dict['pct_outliers'] > 5.0:
        issues_found.append("High proportion of outliers")
    if abs(stats_dict['skewness']) > 1.0:
        issues_found.append("Non-normal residuals")

    if not issues_found:
        print("  ✓ Residuals appear well-behaved")
        print("  ✓ Model assumptions largely satisfied")
        print("  ✓ Predictions are reliable across all ranges")
    else:
        print("  Issues detected:")
        for issue in issues_found:
            print(f"    • {issue}")

    print()

    print("RECOMMENDED NEXT STEPS:")
    print("  1. Review residual plots carefully for patterns")
    print("  2. Investigate outliers and extreme residuals")
    print("  3. Consider target transformation if heteroscedasticity present")
    print("  4. Try advanced models if non-linearity detected")
    print("  5. Implement cross-validation for robust performance estimate")
    print()

    print("=" * 120)
    print()


# ==============================================================================
# SECTION 11: MAIN EXECUTION
# ==============================================================================


def main():
    """Main residual analysis pipeline execution."""
    print("=" * 120)
    print("RESIDUAL ANALYSIS - FLIGHT TELEMETRY")
    print("=" * 120)
    print()

    try:
        # 1. Load data
        model_name, y_actual, y_predicted = load_best_model_data()

        # 2. Calculate residuals
        residuals = calculate_residuals(y_actual, y_predicted)

        # 3. Statistical analysis
        stats_dict = analyze_residuals(residuals, y_predicted)

        # 4. Create histogram
        create_residual_histogram(residuals, model_name, stats_dict)

        # 5. Create scatter plot
        create_residual_scatter(residuals, y_predicted, model_name, stats_dict)

        # 6. Interpretation
        interpret_residuals(stats_dict, y_predicted, residuals)

        # 7. Recommendations
        provide_recommendations()

        # 8. Final summary
        print_final_summary(model_name, stats_dict)

        print("Script completed successfully!")

    except Exception as e:
        print("\n✗ ERROR during analysis:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
