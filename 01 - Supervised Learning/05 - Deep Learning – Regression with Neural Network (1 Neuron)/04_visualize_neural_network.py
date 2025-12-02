#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLIGHT TELEMETRY - NEURAL NETWORK VISUALIZATIONS
==============================================================================

Purpose: Create visualizations for neural network training and comparison

This script:
1. Loads training history from neural network training
2. Creates learning curve plot (training vs validation loss)
3. Visualizes weight comparison between PyTorch and sklearn
4. Generates predicted vs actual scatter plot for neural network
5. Provides insights into convergence and model equivalence

Visualizations:
- Learning curve: Shows how loss decreases during training
- Weight comparison: Barplot comparing PyTorch vs sklearn coefficients
- Predicted vs actual: Standard regression evaluation plot
- Residuals distribution: Error analysis

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


class Config:
    """Visualization configuration."""
    # Directories
    OUTPUT_DIR = Path("outputs")
    IMAGES_DIR = OUTPUT_DIR / "graphics"
    TABLES_DIR = OUTPUT_DIR / "results"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

    # Input files
    TRAINING_HISTORY_FILE = TABLES_DIR / "neural_training_history.csv"
    WEIGHTS_COMPARISON_FILE = TABLES_DIR / "weights_comparison.csv"
    PYTORCH_PREDS_FILE = PREDICTIONS_DIR / "preds_pytorch_single_neuron.csv"
    SKLEARN_PREDS_FILE = PREDICTIONS_DIR / "preds_sklearn_linear.csv"

    # Output files
    LEARNING_CURVE_PNG = IMAGES_DIR / "neural_learning_curve.png"
    WEIGHTS_COMPARISON_PNG = IMAGES_DIR / "neural_weights_comparison.png"
    MODEL_COMPARISON_PNG = IMAGES_DIR / "neural_vs_sklearn_comparison.png"

    # Plot settings
    DPI = 300


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 120)
    print(title)
    print(char * 120)


def safe_read_csv(filepath: Path) -> pd.DataFrame:
    """Safely read CSV file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)


# ==============================================================================
# SECTION 3: LEARNING CURVE VISUALIZATION
# ==============================================================================


def plot_learning_curve(history_df: pd.DataFrame) -> None:
    """
    Plot training and validation loss over epochs.

    Args:
        history_df: DataFrame with columns [epoch, train_loss, val_loss]
    """
    print_section("[SECTION 1] CREATE LEARNING CURVE PLOT")

    print("Plotting training and validation loss curves...")
    print()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot losses
    ax.plot(history_df['epoch'], history_df['train_loss'],
            'b-', linewidth=2, label='Training Loss', marker='o',
            markersize=3, markevery=10)
    ax.plot(history_df['epoch'], history_df['val_loss'],
            'r-', linewidth=2, label='Validation Loss', marker='s',
            markersize=3, markevery=10)

    # Find minimum validation loss
    min_val_idx = history_df['val_loss'].idxmin()
    min_val_epoch = history_df.loc[min_val_idx, 'epoch']
    min_val_loss = history_df.loc[min_val_idx, 'val_loss']

    # Mark best epoch
    ax.axvline(min_val_epoch, color='green', linestyle='--',
               linewidth=1.5, alpha=0.7,
               label=f'Best Epoch ({int(min_val_epoch)})')
    ax.plot(min_val_epoch, min_val_loss, 'g*',
            markersize=15, label=f'Min Val Loss: {min_val_loss:.6f}')

    # Labels and title
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Neural Network Training: Learning Curve',
                 fontsize=14, fontweight='bold', pad=20)

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)

    # Add statistics textbox
    final_train_loss = history_df['train_loss'].iloc[-1]
    final_val_loss = history_df['val_loss'].iloc[-1]
    total_epochs = len(history_df)

    textstr = '\n'.join([
        f'Total Epochs: {total_epochs}',
        f'Final Train Loss: {final_train_loss:.6f}',
        f'Final Val Loss: {final_val_loss:.6f}',
        f'Best Val Loss: {min_val_loss:.6f}',
        f'Best Epoch: {int(min_val_epoch)}'
    ])

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()
    plt.savefig(Config.LEARNING_CURVE_PNG, dpi=Config.DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ Learning curve saved: {Config.LEARNING_CURVE_PNG}")
    print()

    # Print insights
    print("LEARNING CURVE INSIGHTS:")
    print()

    # Check convergence
    if final_val_loss < min_val_loss * 1.1:
        print("  ✓ Model converged successfully")
        print("    Validation loss is close to minimum")
    else:
        print("  ⚠ Model may have overfit")
        print(
            f"    Final val loss ({
                final_val_loss:.6f}) > Min val loss ({
                min_val_loss:.6f})")
    print()

    # Check overfitting
    loss_gap = final_val_loss - final_train_loss
    if loss_gap < 0.1:
        print("  ✓ No significant overfitting detected")
        print(f"    Gap between train and val loss: {loss_gap:.6f}")
    elif loss_gap < 0.5:
        print("  ⚠ Mild overfitting detected")
        print(f"    Gap between train and val loss: {loss_gap:.6f}")
    else:
        print("  ⚠ Significant overfitting detected")
        print(f"    Gap between train and val loss: {loss_gap:.6f}")
    print()


# ==============================================================================
# SECTION 4: WEIGHTS COMPARISON VISUALIZATION
# ==============================================================================


def plot_weights_comparison(weights_df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Plot comparison of PyTorch vs sklearn weights.

    Args:
        weights_df: DataFrame with weight comparison
        top_n: Number of top features to display
    """
    print_section("[SECTION 2] CREATE WEIGHTS COMPARISON PLOT")

    print(f"Plotting weight comparison for top {top_n} features...")
    print()

    # Select top features by absolute sklearn weight
    weights_sorted = weights_df.copy()
    weights_sorted['Abs_Sklearn_Weight'] = weights_sorted['Sklearn_Weight'].abs()
    weights_sorted = weights_sorted.nlargest(top_n, 'Abs_Sklearn_Weight')

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Side-by-side bar comparison
    x = np.arange(len(weights_sorted))
    width = 0.35

    ax1.bar(x - width / 2, weights_sorted['PyTorch_Weight'], width,
            label='PyTorch', alpha=0.8, color='steelblue')
    ax1.bar(x + width / 2, weights_sorted['Sklearn_Weight'], width,
            label='Sklearn', alpha=0.8, color='coral')

    ax1.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Weight Value', fontsize=12, fontweight='bold')
    ax1.set_title(f'Weight Comparison: PyTorch vs Sklearn (Top {top_n})',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(weights_sorted['Feature'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Scatter plot (correlation)
    ax2.scatter(weights_sorted['Sklearn_Weight'],
                weights_sorted['PyTorch_Weight'],
                alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    # Add diagonal line (perfect agreement)
    min_val = min(weights_sorted['Sklearn_Weight'].min(),
                  weights_sorted['PyTorch_Weight'].min())
    max_val = max(weights_sorted['Sklearn_Weight'].max(),
                  weights_sorted['PyTorch_Weight'].max())
    ax2.plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Perfect Agreement', alpha=0.7)

    ax2.set_xlabel('Sklearn Weights', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PyTorch Weights', fontsize=12, fontweight='bold')
    ax2.set_title('Weight Correlation: PyTorch vs Sklearn',
                  fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add correlation coefficient
    corr = np.corrcoef(weights_sorted['Sklearn_Weight'],
                       weights_sorted['PyTorch_Weight'])[0, 1]
    textstr = f'Correlation: {corr:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props, fontweight='bold')

    plt.tight_layout()
    plt.savefig(
        Config.WEIGHTS_COMPARISON_PNG,
        dpi=Config.DPI,
        bbox_inches='tight')
    plt.close()

    print(f"✓ Weights comparison saved: {Config.WEIGHTS_COMPARISON_PNG}")
    print()

    # Print statistics
    print("WEIGHTS COMPARISON STATISTICS:")
    print()
    print(f"  Correlation coefficient: {corr:.6f}")

    mean_diff = weights_sorted['Absolute_Difference'].mean()
    max_diff = weights_sorted['Absolute_Difference'].max()

    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Max absolute difference:  {max_diff:.6f}")
    print()

    if corr > 0.99:
        print("  ✓ EXCELLENT: Near-perfect correlation!")
        print("    This confirms mathematical equivalence")
    elif corr > 0.95:
        print("  ✓ VERY GOOD: Strong correlation")
        print("    Minor differences due to optimization")
    else:
        print("  ⚠ Correlation lower than expected")
        print("    Consider: more training epochs, different learning rate")
    print()


# ==============================================================================
# SECTION 5: PREDICTED VS ACTUAL COMPARISON
# ==============================================================================


def plot_model_comparison(
        pytorch_df: pd.DataFrame,
        sklearn_df: pd.DataFrame) -> None:
    """
    Compare PyTorch and sklearn predictions side-by-side.

    Args:
        pytorch_df: PyTorch predictions DataFrame
        sklearn_df: Sklearn predictions DataFrame
    """
    print_section("[SECTION 3] CREATE MODEL COMPARISON PLOT")

    print("Creating side-by-side comparison of predictions...")
    print()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    pytorch_r2 = r2_score(pytorch_df['y_true'], pytorch_df['y_pred'])
    pytorch_mae = mean_absolute_error(
        pytorch_df['y_true'], pytorch_df['y_pred'])
    pytorch_rmse = np.sqrt(
        mean_squared_error(
            pytorch_df['y_true'],
            pytorch_df['y_pred']))

    sklearn_r2 = r2_score(sklearn_df['y_true'], sklearn_df['y_pred'])
    sklearn_mae = mean_absolute_error(
        sklearn_df['y_true'], sklearn_df['y_pred'])
    sklearn_rmse = np.sqrt(
        mean_squared_error(
            sklearn_df['y_true'],
            sklearn_df['y_pred']))

    # Subplot 1: PyTorch predictions
    ax1.scatter(pytorch_df['y_true'], pytorch_df['y_pred'],
                alpha=0.5, s=50, edgecolors='black', linewidth=0.5)

    # Identity line
    min_val = min(pytorch_df['y_true'].min(), pytorch_df['y_pred'].min())
    max_val = max(pytorch_df['y_true'].max(), pytorch_df['y_pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Perfect Prediction')

    ax1.set_xlabel('Actual Duration (min)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Duration (min)', fontsize=12, fontweight='bold')
    ax1.set_title('PyTorch Single Neuron', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add metrics
    textstr = '\n'.join([
        f'R² = {pytorch_r2:.4f}',
        f'MAE = {pytorch_mae:.2f} min',
        f'RMSE = {pytorch_rmse:.2f} min'
    ])
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')

    # Subplot 2: Sklearn predictions
    ax2.scatter(
        sklearn_df['y_true'],
        sklearn_df['y_pred'],
        alpha=0.5,
        s=50,
        color='coral',
        edgecolors='black',
        linewidth=0.5)

    ax2.plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Perfect Prediction')

    ax2.set_xlabel('Actual Duration (min)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Duration (min)', fontsize=12, fontweight='bold')
    ax2.set_title('Sklearn Linear Regression', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add metrics
    textstr = '\n'.join([
        f'R² = {sklearn_r2:.4f}',
        f'MAE = {sklearn_mae:.2f} min',
        f'RMSE = {sklearn_rmse:.2f} min'
    ])
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()
    plt.savefig(
        Config.MODEL_COMPARISON_PNG,
        dpi=Config.DPI,
        bbox_inches='tight')
    plt.close()

    print(f"✓ Model comparison saved: {Config.MODEL_COMPARISON_PNG}")
    print()

    # Print comparison
    print("PERFORMANCE COMPARISON:")
    print()
    print(f"  PyTorch R²:   {pytorch_r2:.6f}")
    print(f"  Sklearn R²:   {sklearn_r2:.6f}")
    print(f"  Difference:   {abs(pytorch_r2 - sklearn_r2):.6f}")
    print()
    print(f"  PyTorch RMSE: {pytorch_rmse:.4f} min")
    print(f"  Sklearn RMSE: {sklearn_rmse:.4f} min")
    print(f"  Difference:   {abs(pytorch_rmse - sklearn_rmse):.4f} min")
    print()

    if abs(
            pytorch_r2 -
            sklearn_r2) < 0.001 and abs(
            pytorch_rmse -
            sklearn_rmse) < 0.1:
        print("  ✓ EXCELLENT: Performance metrics are nearly identical!")
        print("    This confirms both models learned the same relationship")
    elif abs(pytorch_r2 - sklearn_r2) < 0.01 and abs(pytorch_rmse - sklearn_rmse) < 0.5:
        print("  ✓ GOOD: Performance metrics are very similar")
    else:
        print("  ⚠ Performance differences detected")
        print("    Consider: more training epochs, hyperparameter tuning")
    print()


# ==============================================================================
# SECTION 6: MAIN EXECUTION
# ==============================================================================


def main():
    """Main visualization pipeline."""
    print("=" * 120)
    print("NEURAL NETWORK VISUALIZATIONS - FLIGHT TELEMETRY")
    print("=" * 120)
    print()

    try:
        # Create output directory
        Config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Images directory: {Config.IMAGES_DIR}")
        print()

        # 1. Load training history
        print("Loading training history...")
        history_df = safe_read_csv(Config.TRAINING_HISTORY_FILE)
        print(f"✓ Loaded {len(history_df)} epochs")
        print()

        # 2. Plot learning curve
        plot_learning_curve(history_df)

        # 3. Load weights comparison
        print("Loading weights comparison...")
        weights_df = safe_read_csv(Config.WEIGHTS_COMPARISON_FILE)
        print(f"✓ Loaded comparison for {len(weights_df)} features")
        print()

        # 4. Plot weights comparison
        plot_weights_comparison(weights_df, top_n=10)

        # 5. Load predictions
        print("Loading predictions...")
        pytorch_df = safe_read_csv(Config.PYTORCH_PREDS_FILE)
        sklearn_df = safe_read_csv(Config.SKLEARN_PREDS_FILE)
        print(f"✓ PyTorch predictions: {len(pytorch_df)} samples")
        print(f"✓ Sklearn predictions:  {len(sklearn_df)} samples")
        print()

        # 6. Plot model comparison
        plot_model_comparison(pytorch_df, sklearn_df)

        # Final summary
        print("\n" + "=" * 120)
        print("VISUALIZATION GENERATION COMPLETED")
        print("=" * 120)
        print()

        print("FILES CREATED:")
        print(f"  • Learning curve:      {Config.LEARNING_CURVE_PNG}")
        print(f"  • Weights comparison:  {Config.WEIGHTS_COMPARISON_PNG}")
        print(f"  • Model comparison:    {Config.MODEL_COMPARISON_PNG}")
        print()

        print("KEY TAKEAWAYS:")
        print()
        print("1. LEARNING CURVE:")
        print("   • Shows how loss decreases during training")
        print("   • Helps identify overfitting (train vs validation gap)")
        print("   • Indicates if model has converged")
        print()
        print("2. WEIGHTS COMPARISON:")
        print("   • Visualizes mathematical equivalence")
        print("   • PyTorch and sklearn should learn similar weights")
        print("   • Confirms single neuron = linear regression")
        print()
        print("3. PREDICTIONS COMPARISON:")
        print("   • Both models should perform similarly")
        print("   • Validates correct implementation")
        print("   • Shows prediction quality on test set")
        print()

        print("=" * 120)
        print()
        print("✅ Script completed successfully!")

    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nPlease run 03_train_neural_models.py first to generate required files.")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
