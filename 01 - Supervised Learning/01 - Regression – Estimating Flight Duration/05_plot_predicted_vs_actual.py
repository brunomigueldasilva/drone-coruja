#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLIGHT TELEMETRY - PREDICTED VS ACTUAL VISUALIZATION
==============================================================================

Purpose: Visualize predicted vs actual values for the best performing model

This script:
1. Identifies the best model by reading the best model summary file
2. Loads test target values and corresponding predictions
3. Creates scatter plot of predicted vs actual values with identity line
4. Calculates and displays R² score on the plot
5. Analyzes prediction patterns and error distributions
6. Saves high-quality plots in PNG and PDF formats

Plot Interpretation:
- Points on diagonal line (y=x): Perfect predictions
- Points above line: Over-predictions (model predicts higher than actual)
- Points below line: Under-predictions (model predicts lower than actual)
- Spread around line: Prediction error magnitude
- Systematic patterns: Indicate model bias in certain ranges

The identity line (45-degree line) represents perfect predictions where
predicted values exactly match actual values. Deviations from this line
reveal where and how the model makes errors.

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

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")

plt.rcParams['figure.figsize'] = (10, 8)
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
    """Flight telemetry visualization configuration parameters."""
    # Directories
    OUTPUT_DIR = Path("outputs")
    ARTIFACTS_DIR = OUTPUT_DIR / "models"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    IMAGES_DIR = OUTPUT_DIR / "graphics"

    # Input files
    BEST_MODEL_FILE = ARTIFACTS_DIR / "best_model.txt"
    Y_TEST_FILE = ARTIFACTS_DIR / "y_test.pkl"

    # Output files
    PLOT_PNG = IMAGES_DIR / "predicted_vs_actual.png"
    DATA_CSV = OUTPUT_DIR / "predictions" / "predicted_vs_actual_data.csv"

    # Model name to file mapping
    MODEL_TO_FILE = {
        'Simple Linear Regression': 'preds_linear_simple.csv',
        'Multiple Linear Regression': 'preds_linear_multiple.csv',
        'Ridge Regression': 'preds_ridge.csv',
        'Lasso Regression': 'preds_lasso.csv',
        'Polynomial Regression (degree=2)': 'preds_polynomial_deg2.csv'
    }

    # Plot settings
    DPI = 300
    FIGSIZE = (10, 8)
    SCATTER_SIZE = 50
    SCATTER_ALPHA = 0.6


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 120)
    print(title)
    print(char * 120)


# ==============================================================================
# SECTION 3: BEST MODEL IDENTIFICATION
# ==============================================================================


def identify_best_model() -> Tuple[str, str]:
    """
    Read best model file and identify best model name and predictions file.

    Returns:
        Tuple[str, str]: (best_model_name, predictions_file)

    Raises:
        SystemExit: If best model file not found or invalid
    """
    print_section("SECTION 1: IDENTIFY BEST MODEL")

    if not Config.BEST_MODEL_FILE.exists():
        print(f"✗ ERROR: Best model file not found: {Config.BEST_MODEL_FILE}")
        print("  Please ensure the model training pipeline has completed.")
        exit(1)

    # Read best model file
    with open(Config.BEST_MODEL_FILE, 'r') as f:
        best_model_content = f.read()

    print(f"✓ Loaded best model info from: {Config.BEST_MODEL_FILE}")
    print("\nBest model file content:")
    print("-" * 80)
    print(best_model_content)
    print("-" * 80)

    # Parse model name
    best_model_name = None
    for line in best_model_content.split('\n'):
        if line.startswith('Model Name:'):
            best_model_name = line.split('Model Name:')[1].strip()
            break

    if best_model_name is None:
        print("✗ ERROR: Could not parse model name from best model file")
        exit(1)

    if best_model_name not in Config.MODEL_TO_FILE:
        print(f"✗ ERROR: Unknown model name: {best_model_name}")
        exit(1)

    predictions_file = Config.MODEL_TO_FILE[best_model_name]

    print(f"\n✓ Best model identified: {best_model_name}")
    print(f"✓ Predictions file: {predictions_file}")

    return best_model_name, predictions_file


# ==============================================================================
# SECTION 4: DATA LOADING
# ==============================================================================


def load_test_data(predictions_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ground truth and best model predictions.

    Args:
        predictions_file: Name of predictions CSV file

    Returns:
        Tuple[np.ndarray, np.ndarray]: (y_actual, y_predicted)

    Raises:
        SystemExit: If required files not found
    """
    print_section("SECTION 2: LOAD TEST DATA AND PREDICTIONS")

    # Load ground truth
    if not Config.Y_TEST_FILE.exists():
        print(f"✗ ERROR: Ground truth file not found: {Config.Y_TEST_FILE}")
        print("  Please run the data preprocessing pipeline first.")
        exit(1)

    with open(Config.Y_TEST_FILE, 'rb') as f:
        y_test = pickle.load(f)

    print(f"✓ Ground truth loaded from: {Config.Y_TEST_FILE}")
    print(f"  Shape: {y_test.shape}")
    print(f"  Range: [{y_test.min():.2f}, {y_test.max():.2f}] minutes")
    print(f"  Mean: {y_test.mean():.2f} minutes")

    # Load predictions
    predictions_path = Config.PREDICTIONS_DIR / predictions_file

    if not predictions_path.exists():
        print(f"✗ ERROR: Prediction file not found: {predictions_path}")
        print("  Please run the model training pipeline first.")
        exit(1)

    predictions_df = pd.read_csv(predictions_path)
    print(f"\n✓ Predictions loaded from: {predictions_path}")
    print(f"  Shape: {predictions_df.shape}")

    # Extract actual and predicted values
    y_actual = predictions_df['y_true'].values
    y_predicted = predictions_df['y_pred'].values

    print("\nPredicted values:")
    print(
        f"  Range: [{
            y_predicted.min():.2f}, {
            y_predicted.max():.2f}] minutes")
    print(f"  Mean: {y_predicted.mean():.2f} minutes")

    # Verify data consistency
    if not np.allclose(y_actual, y_test):
        print("\n⚠ WARNING: y_true in predictions file does not match y_test!")
    else:
        print("\n✓ Data consistency verified")

    return y_actual, y_predicted


# ==============================================================================
# SECTION 5: ERROR ANALYSIS
# ==============================================================================


def analyze_prediction_errors(
        y_actual: np.ndarray,
        y_predicted: np.ndarray) -> Dict:
    """
    Perform detailed error analysis with operational interpretation.

    Args:
        y_actual: Ground truth values
        y_predicted: Predicted values

    Returns:
        Dict: Dictionary with error analysis results
    """
    print_section("SECTION 3: CALCULATE METRICS AND ANALYZE PREDICTIONS")

    # Calculate metrics
    r2 = r2_score(y_actual, y_predicted)
    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))

    # Calculate residuals
    residuals = y_actual - y_predicted
    abs_residuals = np.abs(residuals)

    print("\n📊 CORE METRICS:")
    print(f"  R² Score:        {r2:.4f} ({r2 * 100:.2f}% variance explained)")
    print(f"  MAE:             {mae:.4f} minutes")
    print(f"  RMSE:            {rmse:.4f} minutes")
    print(f"  Mean Residual:   {residuals.mean():.4f} minutes")
    print(f"  Std Residual:    {residuals.std():.4f} minutes")

    # Analyze prediction bias
    underestimate_mask = residuals > 0  # Actual > Predicted
    overestimate_mask = residuals < 0   # Actual < Predicted

    n_underestimate = np.sum(underestimate_mask)
    n_overestimate = np.sum(overestimate_mask)

    mean_underestimate = residuals[underestimate_mask].mean(
    ) if n_underestimate > 0 else 0
    mean_overestimate = residuals[overestimate_mask].mean(
    ) if n_overestimate > 0 else 0

    print("\n🎯 PREDICTION BIAS:")
    print(
        f"  Under-predictions: {n_underestimate}/{
            len(y_actual)} ({
            n_underestimate / len(y_actual) * 100:.1f}%)")
    print("    → Model predicts TOO LOW")
    print(f"    → Average error: {mean_underestimate:.2f} minutes")
    print(
        f"  Over-predictions:  {n_overestimate}/{
            len(y_actual)} ({
            n_overestimate / len(y_actual) * 100:.1f}%)")
    print("    → Model predicts TOO HIGH")
    print(f"    → Average error: {abs(mean_overestimate):.2f} minutes")

    # Overall bias interpretation
    if abs(residuals.mean()) < 0.5:
        bias_interpretation = "approximately unbiased"
    elif residuals.mean() > 0:
        bias_interpretation = f"slight tendency to UNDER-predict by {
            residuals.mean():.2f} min on average"
    else:
        bias_interpretation = f"slight tendency to OVER-predict by {
            abs(
                residuals.mean()):.2f} min on average"

    print(f"\nOverall prediction bias: {bias_interpretation}")

    # Analyze by error threshold
    print("\n📏 PREDICTION ACCURACY BY THRESHOLD:")
    thresholds = [1, 2, 5, 10, 15]

    for threshold in thresholds:
        within_threshold = np.sum(abs_residuals <= threshold)
        percentage = (within_threshold / len(residuals)) * 100
        print(
            f"  ±{
                threshold:2d} minutes: {
                percentage:5.1f}% ({within_threshold}/{
                len(residuals)} samples)")

    # Range analysis
    print("\n📊 PREDICTION RANGE ANALYSIS:")
    actual_range = y_actual.max() - y_actual.min()
    predicted_range = y_predicted.max() - y_predicted.min()
    print(f"  Actual values range:    {actual_range:.2f} minutes")
    print(f"  Predicted values range: {predicted_range:.2f} minutes")
    print(f"  Range ratio:            {predicted_range / actual_range:.2f}")

    if predicted_range < actual_range * 0.9:
        print("  ⚠ Model underpredicts the range (conservative predictions)")
    elif predicted_range > actual_range * 1.1:
        print("  ⚠ Model overpredicts the range (volatile predictions)")
    else:
        print("  ✓ Model captures the range well")

    # Flight duration quartile analysis
    print("\n📊 ERROR ANALYSIS BY FLIGHT DURATION QUARTILES:")

    quartiles = np.percentile(y_actual, [25, 50, 75])
    q1_mask = y_actual <= quartiles[0]
    q2_mask = (y_actual > quartiles[0]) & (y_actual <= quartiles[1])
    q3_mask = (y_actual > quartiles[1]) & (y_actual <= quartiles[2])
    q4_mask = y_actual > quartiles[2]

    for i, (mask, label) in enumerate([
        (q1_mask, f"Q1 (≤{quartiles[0]:.0f} min)"),
        (q2_mask, f"Q2 ({quartiles[0]:.0f}-{quartiles[1]:.0f} min)"),
        (q3_mask, f"Q3 ({quartiles[1]:.0f}-{quartiles[2]:.0f} min)"),
        (q4_mask, f"Q4 (>{quartiles[2]:.0f} min)")
    ], 1):
        if np.sum(mask) > 0:
            q_mae = np.mean(abs_residuals[mask])
            q_bias = np.mean(residuals[mask])
            print(f"  {label}:")
            print(f"    Samples: {np.sum(mask)}")
            print(f"    MAE:     {q_mae:.2f} minutes")
            print(f"    Bias:    {q_bias:+.2f} minutes")

    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'residuals': residuals,
        'abs_residuals': abs_residuals,
        'n_underestimate': n_underestimate,
        'n_overestimate': n_overestimate,
        'mean_underestimate': mean_underestimate,
        'mean_overestimate': mean_overestimate,
        'bias_interpretation': bias_interpretation
    }


# ==============================================================================
# SECTION 6: VISUALIZATION CREATION
# ==============================================================================


def create_prediction_plot(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    model_name: str,
    analysis: Dict
) -> None:
    """
    Create professional scatter plot of predicted vs actual values.

    Args:
        y_actual: Ground truth values
        y_predicted: Predicted values
        model_name: Name of best model
        analysis: Dictionary with error analysis results
    """
    print_section("SECTION 4: CREATE VISUALIZATION")

    # Ensure output directory exists
    Config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=Config.FIGSIZE)

    # Color-code points by error magnitude
    abs_residuals = analysis['abs_residuals']
    mae = analysis['mae']

    colors = np.where(
        abs_residuals <= mae,
        'green',
        np.where(abs_residuals <= 2 * mae, 'orange', 'red')
    )

    # Scatter plot
    ax.scatter(
        y_actual,
        y_predicted,
        c=colors,
        alpha=Config.SCATTER_ALPHA,
        s=Config.SCATTER_SIZE,
        edgecolors='black',
        linewidth=0.5
    )

    # Identity line (perfect prediction)
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        'b--',
        linewidth=2,
        label='Perfect Prediction (y=x)',
        alpha=0.7
    )

    # Labels and title
    ax.set_xlabel(
        'Actual Flight Duration (minutes)',
        fontsize=12,
        fontweight='bold')
    ax.set_ylabel(
        'Predicted Flight Duration (minutes)',
        fontsize=12,
        fontweight='bold')
    ax.set_title(
        f'Predicted vs Actual Flight Duration\nBest Model: {model_name}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Add metrics annotation
    r2 = analysis['r2']
    mae_val = analysis['mae']
    rmse = analysis['rmse']

    textstr = (
        f"R² = {r2:.4f}\n"
        f"MAE = {mae_val:.2f} min\n"
        f"RMSE = {rmse:.2f} min\n"
        f"Samples = {len(y_actual)}"
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(
        0.05, 0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=props,
        family='monospace'
    )

    # Add error legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(
            facecolor='green',
            edgecolor='black',
            label=f'Good (|error| ≤ {
                mae:.2f} min)'),
        Patch(
            facecolor='orange',
            edgecolor='black',
            label=f'Moderate ({
                mae:.2f} < |error| ≤ {
                2 * mae:.2f} min)'),
        Patch(
            facecolor='red',
            edgecolor='black',
            label=f'Large (|error| > {
                2 * mae:.2f} min)'),
        plt.Line2D(
            [0],
            [0],
            color='blue',
            linestyle='--',
            linewidth=2,
            label='Perfect Prediction')]

    ax.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=10,
        framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Equal aspect ratio for better visual comparison
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save files
    plt.savefig(Config.PLOT_PNG, dpi=Config.DPI, bbox_inches='tight')
    print(f"✓ Saved PNG: {Config.PLOT_PNG} ({Config.DPI} DPI)")

    plt.close()


# ==============================================================================
# SECTION 7: DATA EXPORT
# ==============================================================================


def export_plot_data(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    model_name: str,
    analysis: Dict
) -> None:
    """
    Export plot data to CSV for auditing and transparency.

    Args:
        y_actual: Ground truth values
        y_predicted: Predicted values
        model_name: Name of best model
        analysis: Dictionary with error analysis results
    """
    print_section("SECTION 5: EXPORT PLOT DATA")

    # Create DataFrame
    residuals = analysis['residuals']
    abs_residuals = analysis['abs_residuals']
    mae = analysis['mae']

    df = pd.DataFrame({
        'sample_id': range(len(y_actual)),
        'actual_duration_minutes': y_actual,
        'predicted_duration_minutes': y_predicted,
        'residual_minutes': residuals,
        'absolute_error_minutes': abs_residuals,
        'percentage_error': (residuals / y_actual) * 100,
        'error_category': np.where(
            abs_residuals <= mae,
            'good',
            np.where(abs_residuals <= 2 * mae, 'moderate', 'large')
        )
    })

    # Sort by absolute error (largest errors first)
    df = df.sort_values(
        'absolute_error_minutes',
        ascending=False).reset_index(
        drop=True)

    # Save to CSV
    df.to_csv(Config.DATA_CSV, index=False, float_format='%.4f')
    print(f"✓ Saved plot data: {Config.DATA_CSV}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    # Summary statistics
    print("\n📊 DATA SUMMARY:")
    print(f"  Good predictions:     {(df['error_category'] == 'good').sum()} "
          f"({(df['error_category'] == 'good').sum() / len(df) * 100:.1f}%)")
    print(f"  Moderate errors:      {(df['error_category'] == 'moderate').sum()} "
          f"({(df['error_category'] == 'moderate').sum() / len(df) * 100:.1f}%)")
    print(f"  Large errors:         {(df['error_category'] == 'large').sum()} "
          f"({(df['error_category'] == 'large').sum() / len(df) * 100:.1f}%)")


# ==============================================================================
# SECTION 8: FINAL SUMMARY
# ==============================================================================


def print_final_summary(model_name: str, analysis: Dict) -> None:
    """
    Print final summary of visualization generation.

    Args:
        model_name: Name of best model
        analysis: Dictionary with error analysis results
    """
    print_section("SECTION 6: FINAL SUMMARY AND RECOMMENDATIONS")

    r2 = analysis['r2']
    mae = analysis['mae']
    rmse = analysis['rmse']

    summary = f"""
✅ VISUALIZATION COMPLETED SUCCESSFULLY!

MODEL: {model_name.upper()}

KEY METRICS:
  • R² Score:     {r2:.4f} ({r2 * 100:.2f}% variance explained)
  • MAE:          {mae:.2f} minutes
  • RMSE:         {rmse:.2f} minutes

PREDICTION BIAS:
  • Under-predictions: {analysis['n_underestimate']} samples (avg: {analysis['mean_underestimate']:.2f} min)
  • Over-predictions:  {analysis['n_overestimate']} samples (avg: {abs(analysis['mean_overestimate']):.2f} min)

OUTPUT FILES:
  📊 Visualizations:
     • {Config.PLOT_PNG.name} (high-res PNG, {Config.DPI} DPI)

  📄 Data:
     • {Config.DATA_CSV.name} (audit data)

LOCATION: {Config.IMAGES_DIR.absolute()}

INTERPRETATION:
  The scatter plot shows the relationship between actual and predicted flight durations.
  Points closer to the blue dashed line (y=x) indicate better predictions.
  Color coding helps identify prediction quality at a glance:
    - Green: Good predictions (|error| ≤ MAE)
    - Orange: Moderate errors (MAE < |error| ≤ 2×MAE)
    - Red: Large errors (|error| > 2×MAE)

OPERATIONAL SIGNIFICANCE:
  • ±5 minutes: Acceptable for most operational planning
  • ±10 minutes: May require buffer adjustments but manageable
  • >15 minutes: Significant error, may impact scheduling and connections

NEXT STEPS:
  1. Review the visualization to assess model performance
  2. Investigate samples with large errors (red points)
  3. Consider operational implications of prediction bias
  4. Use audit CSV for detailed error analysis
  5. Deploy model if performance meets operational requirements

RECOMMENDATIONS:
  1. IMMEDIATE ACTIONS:
     • Review predictions with large errors (outliers)
     • Investigate systematic patterns in residuals
     • Create additional diagnostic plots (residual plot, Q-Q plot)

  2. MODEL IMPROVEMENT:
     • If non-linear patterns detected: Add polynomial features or use non-linear models
     • If range-specific bias detected: Consider segmented models or stratified approach
     • If outliers present: Investigate data quality, consider robust regression

  3. OPERATIONAL DEPLOYMENT:
     • Define acceptable error thresholds for your use case
     • Add prediction intervals to quantify uncertainty
     • Implement monitoring to track performance over time
     • Consider separate models for different flight duration ranges

POTENTIAL SOURCES OF PREDICTION ERRORS:
  • Non-linearity: Relationship may not be perfectly linear
  • Missing variables: Weather, traffic congestion, aircraft type, etc.
  • Outliers: Diversions, mechanical issues, unusual circumstances
  • Heteroscedasticity: Prediction uncertainty may vary with flight duration
"""

    print(summary)


# ==============================================================================
# SECTION 9: MAIN EXECUTION
# ==============================================================================


def main():
    """Main visualization pipeline execution."""
    print("=" * 120)
    print("PREDICTED VS ACTUAL VISUALIZATION - FLIGHT TELEMETRY")
    print("=" * 120)

    try:
        # 1. Identify best model
        model_name, predictions_file = identify_best_model()

        # 2. Load data
        y_actual, y_predicted = load_test_data(predictions_file)

        # 3. Analyze errors
        analysis = analyze_prediction_errors(y_actual, y_predicted)

        # 4. Create visualization
        create_prediction_plot(y_actual, y_predicted, model_name, analysis)

        # 5. Export data
        export_plot_data(y_actual, y_predicted, model_name, analysis)

        # 6. Print summary
        print_final_summary(model_name, analysis)

        print("\n" + "=" * 120)
        print("✅ VISUALIZATION GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 120)

    except Exception as e:
        print("\n✗ ERROR during visualization:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
