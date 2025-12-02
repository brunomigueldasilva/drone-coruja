#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
TIME SERIES FORECAST VISUALIZATION - BEST MODEL PERFORMANCE
==============================================================================

Purpose: Visualize the best model's predictions against actual values over time

This script:
1. Automatically selects the best model based on lowest RMSE from metrics comparison
2. Loads training data, test data, and predictions preserving datetime indices
3. Creates a comprehensive timeline visualization showing:
   - Historical training data (context for model learning)
   - Actual test values (ground truth)
   - Model predictions (model performance)
4. Adds visual separators between training and test periods
5. Displays key performance metrics on the plot
6. Provides detailed interpretation of model performance
7. Saves publication-quality plots (PNG and PDF)

Why Visualize the Full Timeline (Train + Test):
- Context: Shows what data the model learned from
- Continuity: Reveals if model captures the natural flow of the time series
- Transition: Shows how well model performs immediately after training period
- Trend: Visualizes if model understands long-term patterns
- Seasonality: Shows if model captures cyclical patterns across the split

How to Spot Overfitting:
- Perfect fit on training data but poor on test data
- Predictions follow training data closely then diverge dramatically
- Model captures noise in training but misses patterns in test
- High variance in test predictions vs smooth training fit

What Makes a Good Forecast:
- Follows overall trend: Predictions move in same direction as actuals
- Captures cycles: If data is seasonal, predictions show similar patterns
- Small systematic errors: Predictions cluster around actuals, not biased
- Stable performance: Error magnitude consistent across test period
- Natural transition: No abrupt changes at train/test boundary

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

warnings.filterwarnings('ignore')

# Set plotting style for professional appearance
sns.set_style("whitegrid")
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


# Configuration
class Config:
    """Configuration for time series forecast visualization."""
    # Directories
    OUTPUT_DIR = Path("outputs")
    DATA_DIR = OUTPUT_DIR / "data_processed"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    GRAPHICS_DIR = OUTPUT_DIR / "graphics"

    # Plot settings
    DPI = 300
    FIGSIZE = (15, 6)
    TRAIN_COLOR = '#7fa8c9'  # Light blue
    TRAIN_ALPHA = 0.6
    ACTUAL_COLOR = '#1f77b4'  # Blue
    PRED_COLOR = '#ff7f0e'  # Orange
    SEPARATOR_COLOR = '#d62728'  # Red
    GRID_ALPHA = 0.3


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header for better readability."""
    print("\n" + char * 120)
    print(title)
    print(char * 120)


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {Config.GRAPHICS_DIR}")


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_evaluation_results(
        dataset_name='voltage') -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Load training data, test data, and metrics comparison.

    This function loads the complete dataset needed for visualization:
    - y_train: Historical data used to train models (with datetime index)
    - y_test: Actual values in test period (with datetime index)
    - metrics_df: Performance metrics for all models

    The datetime indices are crucial for creating an accurate timeline
    that shows when each data point occurred in the time series.

    Args:
        dataset_name: 'voltage' or 'missions'

    Returns:
        Tuple containing:
        - y_train (pd.Series): Training target values with datetime index
        - y_test (pd.Series): Test target values with datetime index
        - metrics_df (pd.DataFrame): Model performance metrics

    Raises:
        SystemExit: If required files are not found
    """
    print_section("SECTION 1: LOAD EVALUATION RESULTS")

    # Load training data with datetime index
    y_train_file = Config.DATA_DIR / dataset_name / 'y_train.pkl'
    if not y_train_file.exists():
        print(f"✗ ERROR: Training data file not found: {y_train_file}")
        print("  Please run the data preprocessing pipeline first.")
        exit(1)

    with open(y_train_file, 'rb') as f:
        y_train = pickle.load(f)

    print(f"✓ Training data loaded: {y_train_file}")
    print(f"  Shape: {y_train.shape}")
    print(f"  Date range: {y_train.index.min()} to {y_train.index.max()}")
    print(f"  Value range: [{y_train.min():.2f}, {y_train.max():.2f}]")

    # Load test data with datetime index
    y_test_file = Config.DATA_DIR / dataset_name / 'y_test.pkl'
    if not y_test_file.exists():
        print(f"\n✗ ERROR: Test data file not found: {y_test_file}")
        print("  Please run the data preprocessing pipeline first.")
        exit(1)

    with open(y_test_file, 'rb') as f:
        y_test = pickle.load(f)

    print(f"\n✓ Test data loaded: {y_test_file}")
    print(f"  Shape: {y_test.shape}")
    print(f"  Date range: {y_test.index.min()} to {y_test.index.max()}")
    print(f"  Value range: [{y_test.min():.2f}, {y_test.max():.2f}]")

    # Load metrics comparison
    metrics_file = Config.OUTPUT_DIR / f'{dataset_name}_metrics_comparison.csv'
    if not metrics_file.exists():
        print(f"\n✗ ERROR: Metrics file not found: {metrics_file}")
        print("  Please run the model evaluation pipeline first.")
        exit(1)

    metrics_df = pd.read_csv(metrics_file)

    print(f"\n✓ Metrics comparison loaded: {metrics_file}")
    print(f"  Models evaluated: {len(metrics_df)}")
    print("\nAvailable models:")
    for idx, row in metrics_df.iterrows():
        print(f"  {idx + 1}. {row['Model']} - RMSE: {row['RMSE']:.4f}")

    return y_train, y_test, metrics_df


def select_best_model(metrics_df: pd.DataFrame,
                      dataset_name='voltage') -> Tuple[str,
                                                       np.ndarray,
                                                       Dict[str,
                                                            float]]:
    """
    Select the best performing model based on lowest RMSE.

    RMSE (Root Mean Squared Error) is used as the primary metric because:
    - It penalizes large errors more heavily than MAE
    - It's in the same units as the target variable (interpretable)
    - It's the most commonly used metric for regression tasks

    Args:
        metrics_df: DataFrame containing model performance metrics
        dataset_name: 'voltage' or 'missions'

    Returns:
        Tuple containing:
        - model_name (str): Name of the best model
        - predictions (np.ndarray): Predictions array
        - metrics (dict): Performance metrics for the best model

    Raises:
        SystemExit: If predictions file not found
    """
    print_section("SECTION 2: SELECT BEST MODEL")

    # Find model with lowest RMSE
    best_idx = metrics_df['RMSE'].idxmin()
    best_row = metrics_df.loc[best_idx]

    model_name = best_row['Model']
    rmse = best_row['RMSE']

    # Extract metrics
    metrics = {
        'MAE': best_row['MAE'],
        'RMSE': rmse,
        'MAPE': best_row['MAPE']
    }

    print(f"🏆 Best model selected: {model_name}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {metrics['MAE']:.4f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")

    # Load predictions
    predictions_file = Config.PREDICTIONS_DIR / \
        dataset_name / f'{model_name}_predictions.pkl'

    if not predictions_file.exists():
        print(f"\n✗ ERROR: Predictions file not found: {predictions_file}")
        print(f"  Expected location: {predictions_file}")
        exit(1)

    with open(predictions_file, 'rb') as f:
        predictions = pickle.load(f)

    print(f"\n✓ Predictions loaded: {predictions_file.name}")
    print(f"  Shape: {predictions.shape}")
    print(f"  Range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    return model_name, predictions, metrics


# ==============================================================================
# SECTION 4: VISUALIZATION
# ==============================================================================


def create_forecast_plot(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    metrics: Dict[str, float],
    dataset_name: str = 'voltage'
) -> None:
    """
    Create comprehensive time series forecast visualization.

    This plot shows three critical components:
    1. Historical training data - provides context for what model learned
    2. Actual test values - ground truth for model evaluation
    3. Model predictions - shows how well model performs on unseen data

    The visualization helps identify:
    - Trend capture: Does model follow overall direction?
    - Seasonality capture: Does model detect cyclical patterns?
    - Prediction lag: Does model react too slowly to changes?
    - Systematic bias: Does model consistently over/under-predict?
    - Performance stability: Are errors consistent or variable?

    Args:
        y_train: Training data with datetime index
        y_test: Test data with datetime index
        y_pred: Predictions array
        model_name: Name of the model being visualized
        metrics: Dictionary with MAE, RMSE, MAPE values
        dataset_name: 'voltage' or 'missions'
    """
    print_section("SECTION 3: CREATE FORECAST VISUALIZATION")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=Config.FIGSIZE, dpi=Config.DPI)

    # Get target variable name (from series name or default)
    target_name = y_test.name if y_test.name else "Target Variable"

    # COMPONENT 1: Plot historical training data
    # This shows the context - what data the model learned from
    # Semi-transparent to not overshadow test period
    ax.plot(
        y_train.index,
        y_train.values,
        color=Config.TRAIN_COLOR,
        alpha=Config.TRAIN_ALPHA,
        linewidth=1.5,
        label='Historical Training Data'
    )
    print("✓ Plotted training data (historical context)")

    # COMPONENT 2: Plot actual test values
    # This is the ground truth for model evaluation
    # Solid line with higher opacity for emphasis
    ax.plot(
        y_test.index,
        y_test.values,
        color=Config.ACTUAL_COLOR,
        linewidth=2,
        label='Actual Test Values',
        zorder=3
    )
    print("✓ Plotted actual test values (ground truth)")

    # COMPONENT 3: Plot model predictions
    # Dashed line to distinguish from actuals
    # Allows visual comparison of prediction vs reality
    # Create predictions series with same index as y_test
    y_pred_series = pd.Series(y_pred, index=y_test.index)

    ax.plot(
        y_pred_series.index,
        y_pred_series.values,
        color=Config.PRED_COLOR,
        linewidth=2,
        linestyle='--',
        label=f'Predictions ({model_name})',
        zorder=4
    )
    print(f"✓ Plotted {model_name} predictions")

    # Add vertical separator between train and test periods
    # This helps identify where model transitions from seen to unseen data
    # Important for detecting overfitting or underfitting patterns
    separator_date = y_test.index[0]
    ax.axvline(
        x=separator_date,
        color=Config.SEPARATOR_COLOR,
        linestyle=':',
        linewidth=2,
        alpha=0.7,
        label='Train/Test Split',
        zorder=2
    )
    print("✓ Added train/test separator")

    # Add shaded region for test period
    # Subtle background shading to emphasize evaluation period
    ax.axvspan(
        y_test.index[0],
        y_test.index[-1],
        alpha=0.05,
        color='gray',
        zorder=1
    )

    # Format plot
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel(target_name, fontsize=12, fontweight='bold')
    ax.set_title(
        f'Time Series Forecast: {model_name} - Prediction vs. Actual\n'
        f'Dataset: {dataset_name.upper()}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Grid for easier reading
    ax.grid(True, alpha=Config.GRID_ALPHA, linestyle='--', linewidth=0.5)

    # Legend with frame
    ax.legend(
        loc='best',
        fontsize=10,
        framealpha=0.9,
        edgecolor='gray'
    )

    # Add metrics text box
    # Display key performance metrics on the plot for quick reference
    # Position automatically to avoid overlapping data
    metrics_text = (
        f"Performance Metrics:\n"
        f"MAE:  {metrics['MAE']:.4f}\n"
        f"RMSE: {metrics['RMSE']:.4f}\n"
        f"MAPE: {metrics['MAPE']:.2f}%"
    )

    ax.text(
        0.02, 0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='gray',
            alpha=0.9
        ),
        family='monospace'
    )
    print("✓ Added metrics text box")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save plots in multiple formats
    png_file = Config.GRAPHICS_DIR / f"{dataset_name}_best_model_forecast.png"
    pdf_file = Config.GRAPHICS_DIR / f"{dataset_name}_best_model_forecast.pdf"

    plt.savefig(png_file, dpi=Config.DPI, bbox_inches='tight')
    print(f"\n✓ Saved PNG: {png_file}")

    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✓ Saved PDF: {pdf_file} (publication quality)")

    plt.close()


# ==============================================================================
# SECTION 5: INTERPRETATION AND ANALYSIS
# ==============================================================================


def generate_interpretation(
        y_test: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        metrics: Dict) -> None:
    """
    Generate detailed interpretation of model performance.

    This analysis helps answer critical questions:
    - Does the model capture the overall trend?
    - Does it capture seasonality/cyclical patterns?
    - Where does it perform worst?
    - Are there systematic errors?

    The interpretation provides actionable insights for model improvement
    and deployment decisions.

    Args:
        y_test: Actual test values
        y_pred: Predicted values
        model_name: Name of the model
        metrics: Performance metrics dictionary
    """
    print_section("SECTION 4: VISUAL ANALYSIS AND INTERPRETATION")

    # Calculate residuals (errors)
    residuals = y_test.values - y_pred

    # 1. TREND CAPTURE ANALYSIS
    # Check if predictions follow the same direction as actuals
    test_diff = np.diff(y_test.values)
    pred_diff = np.diff(y_pred)

    # Direction agreement: do predictions move in same direction as actuals?
    direction_agreement = np.mean((test_diff * pred_diff) > 0)

    trend_captured = direction_agreement > 0.7  # 70% threshold

    print("\n📊 VISUAL ANALYSIS:")
    print("=" * 100)

    print("\n1. OVERALL TREND CAPTURE:")
    print(f"   Direction Agreement: {direction_agreement:.1%}")
    if trend_captured:
        print("   ✓ YES - The model successfully captures the overall trend.")
        print("   The predictions move in the same direction as actual values most of the time.")
    else:
        print("   ✗ NO - The model struggles to capture the overall trend.")
        print(
            "   The predictions frequently move in opposite directions from actual values.")
        print("   → Recommendation: Consider adding lag features or using a more sophisticated model.")

    # 2. SEASONALITY ANALYSIS
    # Check for periodic patterns by analyzing autocorrelation
    print("\n2. SEASONALITY PATTERN CAPTURE:")

    # Simple seasonality check: compare variance in different periods
    n_test = len(y_test)
    if n_test >= 12:  # Need sufficient data for seasonality analysis
        # Split into thirds to check consistency
        third = n_test // 3

        error_first_third = np.mean(np.abs(residuals[:third]))
        error_second_third = np.mean(np.abs(residuals[third:2 * third]))
        error_last_third = np.mean(np.abs(residuals[2 * third:]))

        # Calculate coefficient of variation in errors
        error_cv = np.std([error_first_third,
                           error_second_third,
                           error_last_third]) / np.mean([error_first_third,
                                                         error_second_third,
                                                         error_last_third])

        if error_cv < 0.2:  # Low variation in errors suggests consistent performance
            print("   ✓ YES - Error magnitude is consistent across the test period.")
            print(
                "   The model shows stable performance, suggesting it captures patterns well.")
        else:
            print("   ⚠ PARTIAL - Error magnitude varies across the test period.")
            print("   The model may struggle with certain seasonal patterns or periods.")
            print("   → Recommendation: Investigate time periods with higher errors.")
    else:
        print("   ℹ INSUFFICIENT DATA - Need longer test period for seasonality analysis.")

    # 3. WORST PERFORMANCE PERIOD
    print("\n3. WORST PERFORMANCE LOCATION:")

    # Calculate rolling mean absolute error
    window = max(5, len(residuals) // 10)  # 10% of data or minimum 5 points
    rolling_mae = pd.Series(
        np.abs(residuals)).rolling(
        window=window,
        center=True).mean()

    worst_idx = rolling_mae.idxmax()
    worst_period = worst_idx / len(residuals)

    if worst_period < 0.33:
        period_name = "BEGINNING"
        explanation = "Model may need warm-up time or initial conditions are challenging."
    elif worst_period < 0.67:
        period_name = "MIDDLE"
        explanation = "Model may struggle with specific patterns or regime changes."
    else:
        period_name = "END"
        explanation = "Model performance degrades over time, suggesting drift or concept shift."

    print(f"   Period: {period_name} of test period")
    print(
        f"   Location: Around sample {worst_idx}/{len(residuals)} ({worst_period:.0%})")
    print(f"   Explanation: {explanation}")
    print("   → Recommendation: Examine data around this period for anomalies or changes.")

    # 4. SYSTEMATIC ERRORS
    print("\n4. SYSTEMATIC ERROR ANALYSIS:")

    # Check for bias (consistent over/under-prediction)
    mean_error = np.mean(residuals)
    mae = np.mean(np.abs(residuals))

    # Normalize bias by MAE to determine significance
    bias_ratio = abs(mean_error) / mae if mae > 0 else 0

    if bias_ratio < 0.1:
        print("   ✓ NO SIGNIFICANT BIAS - Errors are balanced.")
        print("   The model does not systematically over-predict or under-predict.")
    elif mean_error > 0:
        print(
            f"   ⚠ UNDER-PREDICTION BIAS detected (avg error: {mean_error:.4f})")
        print("   The model tends to predict lower values than actual.")
        print("   → Recommendation: Check if target variable scaling or feature engineering is appropriate.")
    else:
        print(
            f"   ⚠ OVER-PREDICTION BIAS detected (avg error: {mean_error:.4f})")
        print("   The model tends to predict higher values than actual.")
        print("   → Recommendation: Review training data for outliers or consider regularization.")

    # Check for lag (predictions following actuals with delay)
    correlation_lag0 = np.corrcoef(y_test.values, y_pred)[0, 1]

    if len(y_test) > 1:
        correlation_lag1 = np.corrcoef(y_test.values[1:], y_pred[:-1])[0, 1]

        if correlation_lag1 > correlation_lag0:
            print("\n   ⚠ PREDICTION LAG detected:")
            print(f"   Correlation with actual: {correlation_lag0:.3f}")
            print(f"   Correlation with lagged actual: {correlation_lag1:.3f}")
            print("   The model's predictions lag behind actual changes.")
            print(
                "   → Recommendation: Add recent lag features or use autoregressive models.")

    # 5. ERROR DISTRIBUTION
    print("\n5. ERROR DISTRIBUTION:")

    # Check for heteroscedasticity (non-constant error variance)
    residuals_abs = np.abs(residuals)
    correlation_errors_level = np.corrcoef(y_test.values, residuals_abs)[0, 1]

    if abs(correlation_errors_level) > 0.3:
        print(
            f"   ⚠ HETEROSCEDASTICITY detected (correlation: {
                correlation_errors_level:.3f})")
        print("   Error magnitude depends on the level of the target variable.")
        if correlation_errors_level > 0:
            print("   Errors increase with higher target values.")
            print(
                "   → Recommendation: Consider log transformation or weighted regression.")
        else:
            print("   Errors increase with lower target values.")
            print("   → Recommendation: Check for data quality issues in lower ranges.")
    else:
        print("   ✓ HOMOSCEDASTIC ERRORS - Error variance is relatively constant.")
        print("   This is ideal for most forecasting models.")

    # SUMMARY
    print("\n" + "=" * 100)
    print("📋 SUMMARY:")
    print(f"   Model: {model_name}")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   MAE:  {metrics['MAE']:.4f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")

    # Overall assessment
    # Note: R² not available in metrics, using MAPE instead
    if metrics['MAPE'] < 10 and bias_ratio < 0.1:
        print("\n   ✅ EXCELLENT - Model shows strong performance with minimal bias.")
        print("   Ready for deployment with standard monitoring.")
    elif metrics['MAPE'] < 20:
        print("\n   ✓ GOOD - Model performs adequately but has room for improvement.")
        print("   Consider feature engineering or trying alternative models.")
    else:
        print("\n   ⚠ NEEDS IMPROVEMENT - Model performance is below expectations.")
        print("   Significant refinement needed before deployment.")

    print("=" * 100)


# ==============================================================================
# SECTION 6: MAIN EXECUTION
# ==============================================================================


def main(dataset_name='voltage'):
    """
    Main execution pipeline for time series forecast visualization.

    Pipeline Steps:
    1. Load training data, test data, and metrics comparison
    2. Select best model based on lowest RMSE
    3. Load predictions for the best model
    4. Create comprehensive timeline visualization
    5. Generate detailed interpretation and recommendations

    The pipeline preserves datetime indices throughout to ensure
    accurate temporal visualization and analysis.

    Args:
        dataset_name: 'voltage' or 'missions'
    """
    print("=" * 120)
    print(
        f"TIME SERIES FORECAST VISUALIZATION - BEST MODEL PERFORMANCE: {dataset_name.upper()}")
    print("=" * 120)

    try:
        # Ensure output directories exist
        ensure_directories()

        # 1. Load data
        y_train, y_test, metrics_df = load_evaluation_results(dataset_name)

        # 2. Select best model
        model_name, predictions, metrics = select_best_model(
            metrics_df, dataset_name)

        # 3. Create visualization
        create_forecast_plot(
            y_train,
            y_test,
            predictions,
            model_name,
            metrics,
            dataset_name)

        # 4. Generate interpretation
        generate_interpretation(y_test, predictions, model_name, metrics)

        # Final summary
        print("\n" + "=" * 120)
        print("✅ FORECAST VISUALIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 120)
        print(f"\nOutput files saved to: {Config.GRAPHICS_DIR.absolute()}")
        print(f"  • {dataset_name}_best_model_forecast.png (high-resolution)")
        print(
            f"  • {dataset_name}_best_model_forecast.pdf (publication quality)")
        print("\nNext steps:")
        print("  1. Review the visualization to assess model performance")
        print("  2. Consider the interpretation recommendations")
        print("  3. Decide on model deployment or further refinement")
        print("=" * 120)

    except Exception as e:
        print("\n✗ ERROR during visualization generation:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import sys

    # Allow user to specify dataset via command line argument
    # Usage: python 05_plot_predicted_vs_actual.py voltage
    #    or: python 05_plot_predicted_vs_actual.py missions
    #    or: python 05_plot_predicted_vs_actual.py both (plots both)

    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
        if dataset not in ['voltage', 'missions', 'both']:
            print(
                f"Error: Invalid dataset '{dataset}'. Choose 'voltage', 'missions', or 'both'")
            sys.exit(1)
    else:
        dataset = 'both'  # Default: plot both datasets

    # Process datasets
    if dataset == 'both':
        datasets_to_plot = ['voltage', 'missions']
    else:
        datasets_to_plot = [dataset]

    # Create visualizations for each dataset
    for ds in datasets_to_plot:
        print(f"\n{'=' * 120}")
        print(f"PROCESSING DATASET: {ds.upper()}")
        print(f"{'=' * 120}")
        main(ds)
