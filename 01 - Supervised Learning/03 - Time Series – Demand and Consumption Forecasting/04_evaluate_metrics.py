#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
TIME SERIES FORECASTING - MODEL EVALUATION WITH COMPREHENSIVE METRICS
==============================================================================

Purpose: Evaluate all trained forecasting models using standard time series metrics

This script:
1. Loads test data and all model predictions
2. Calculates four key metrics for each model:
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - MAPE (Mean Absolute Percentage Error)
3. Creates comparison tables
4. Identifies best performing model
5. Calculates improvement over naive baseline

CRITICAL CONCEPT: THE NAIVE BASELINE
====================================

A fundamental principle in time series forecasting:

    "A model that doesn't beat the naive baseline is NOT useful,
     regardless of its complexity or sophistication."

The naive baseline (persistence model) simply predicts that tomorrow's value
will equal today's value. It requires ZERO computation and ZERO training.

WHY IS THIS IMPORTANT?
- If your complex ML model can't beat "tomorrow = today", it adds no value
- The naive baseline establishes the MINIMUM acceptable performance
- Always compare against this baseline before deploying any model
- In production, you'd be better off using the naive baseline than a model
  that performs worse (Occam's razor: simpler is better if equally effective)

REAL-WORLD ANALOGY:
If you can't forecast better than saying "tomorrow's weather = today's weather",
your weather model is useless, even if it uses neural networks and GPUs.

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple

from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Evaluation configuration parameters."""
    PROCESSED_DATA_DIR = Path('outputs/data_processed')
    MODELS_DIR = Path('outputs/models')
    RESULTS_DIR = Path('outputs')

    # Metrics output files
    METRICS_CSV = 'metrics_comparison.csv'
    METRICS_MD = 'metrics_comparison.md'


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """
    Print formatted section header.

    Args:
        title: Section title to display
        char: Character to use for border (default: "=")
    """
    print("\n" + char * 80)
    print(title)
    print(char * 80)


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_test_data(dataset_name: str = 'voltage') -> pd.Series:
    """
    Load test target values from preprocessed data.

    Args:
        dataset_name: Name of dataset ('voltage' or 'missions')

    Returns:
        pd.Series: Test target values (actual values to compare against)

    Raises:
        SystemExit: If file not found
    """
    print_section("LOADING TEST DATA")

    data_dir = Config.PROCESSED_DATA_DIR / dataset_name
    y_test_path = data_dir / 'y_test.pkl'

    if not y_test_path.exists():
        print(f"✗ ERROR: Test data not found at {y_test_path}")
        print("\nPlease run the preprocessing script first:")
        print("  python 02_preprocessing.py")
        exit(1)

    with open(y_test_path, 'rb') as f:
        y_test = pickle.load(f)

    print(f"✓ Loaded y_test from: {y_test_path}")
    print(f"  Shape: {y_test.shape}")
    print(f"  Range: [{y_test.min():.4f}, {y_test.max():.4f}]")
    print(f"  Mean: {y_test.mean():.4f}")
    print(f"  Std: {y_test.std():.4f}")

    return y_test


def load_training_summary(dataset_name: str = 'voltage') -> Dict:
    """
    Load training summary containing all model predictions.

    The training summary is created by 03_train_models.py and contains:
    - Model names
    - Training times
    - Predictions dictionary (all model predictions on test set)

    Args:
        dataset_name: Name of dataset ('voltage' or 'missions')

    Returns:
        dict: Training summary with predictions for all models

    Raises:
        SystemExit: If file not found
    """
    print_section("LOADING MODEL PREDICTIONS")

    models_dir = Config.MODELS_DIR / dataset_name
    summary_path = models_dir / 'training_summary.pkl'

    if not summary_path.exists():
        print(f"✗ ERROR: Training summary not found at {summary_path}")
        print("\nPlease run the training script first:")
        print(f"  python 03_train_models.py {dataset_name}")
        exit(1)

    with open(summary_path, 'rb') as f:
        training_summary = pickle.load(f)

    print(f"✓ Loaded training summary from: {summary_path}")
    print(f"  Dataset: {training_summary['dataset_name']}")
    print(f"  Models: {len(training_summary['model_names'])}")

    print("\nModels found:")
    for i, (model_name, train_time) in enumerate(zip(
            training_summary['model_names'],
            training_summary['training_times']), 1):
        print(f"  {i}. {model_name:20s} (trained in {train_time:.4f}s)")

    return training_summary


# ==============================================================================
# SECTION 4: METRICS CALCULATION
# ==============================================================================


def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for time series forecasting.

    METRICS EXPLAINED:
    ==================

    1. MAE (Mean Absolute Error):
       - Average of absolute differences between predicted and actual
       - Formula: MAE = (1/n) * Σ|actual - predicted|
       - Units: Same as target variable (e.g., volts, missions)
       - Interpretation: Average error magnitude, treats all errors equally
       - Pro: Easy to interpret, robust to outliers
       - Con: Doesn't heavily penalize large errors
       - Use when: All errors should be weighted equally

    2. MSE (Mean Squared Error):
       - Average of squared differences
       - Formula: MSE = (1/n) * Σ(actual - predicted)²
       - Units: Squared units of target (e.g., volts², missions²)
       - Interpretation: Emphasizes larger errors due to squaring
       - Pro: Differentiable (good for optimization), penalizes large errors
       - Con: Hard to interpret due to squared units, sensitive to outliers
       - Use when: Large errors are particularly undesirable

    3. RMSE (Root Mean Squared Error):
       - Square root of MSE, brings back to original units
       - Formula: RMSE = √MSE = √[(1/n) * Σ(actual - predicted)²]
       - Units: Same as target variable (e.g., volts, missions)
       - Interpretation: "Typical" prediction error, penalizes large errors
       - Pro: Same units as target, penalizes large errors more than MAE
       - Con: Still sensitive to outliers
       - Use when: You want interpretable metric that penalizes large errors

       RMSE vs MAE:
       - If RMSE >> MAE: Model has some large errors (high variance)
       - If RMSE ≈ MAE: Errors are consistent in size
       - RMSE ≥ MAE always (equality only if all errors are identical)

    4. MAPE (Mean Absolute Percentage Error):
       - Average of absolute percentage errors
       - Formula: MAPE = (100/n) * Σ(|actual - predicted| / |actual|)
       - Units: Percentage (%)
       - Interpretation: Average percentage deviation from actual value
       - Pro: Scale-independent, easy to interpret ("off by X%")
       - Con: Undefined when actual = 0, biased toward under-predictions
       - Use when: Comparing models across different scales/datasets

       MAPE CONSIDERATIONS:
       - Skips samples where actual = 0 (division by zero)
       - If many zeros in data, MAPE may not be reliable
       - Alternative: sMAPE (symmetric MAPE) for zero-heavy data

    WHY WE NEED MULTIPLE METRICS:
    ==============================
    - MAE: Overall average error (equal weight to all mistakes)
    - RMSE: Emphasizes large errors (crucial if big mistakes are costly)
    - MAPE: Scale-independent comparison (useful across different datasets)
    - Together: Complete picture of model performance

    Args:
        y_true: Actual values (ground truth)
        y_pred: Predicted values

    Returns:
        dict: Dictionary with all calculated metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # MAE: Mean Absolute Error
    # Average magnitude of errors, treating all errors equally
    mae = mean_absolute_error(y_true, y_pred)

    # MSE: Mean Squared Error
    # Average of squared errors, penalizes large errors heavily
    mse = mean_squared_error(y_true, y_pred)

    # RMSE: Root Mean Squared Error
    # Square root of MSE, interpretable in original units
    # Penalizes large errors more than MAE
    rmse = np.sqrt(mse)

    # MAPE: Mean Absolute Percentage Error
    # Percentage error, scale-independent
    # CRITICAL: Handle division by zero (when actual value = 0)

    # Create mask for non-zero actual values
    non_zero_mask = y_true != 0

    if non_zero_mask.sum() == 0:
        # All actual values are zero - MAPE is undefined
        mape = np.nan
        print("  ⚠️  WARNING: All actual values are zero, MAPE cannot be calculated")
    else:
        # Calculate MAPE only for non-zero actual values
        # Formula: (100 / n) * Σ(|actual - predicted| / |actual|)
        absolute_percentage_errors = np.abs(
            (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        mape = 100.0 * np.mean(absolute_percentage_errors)

        # If we skipped some samples, report it
        n_skipped = len(y_true) - non_zero_mask.sum()
        if n_skipped > 0:
            print(f"  ⚠️  Note: MAPE calculated on {non_zero_mask.sum()} samples "
                  f"({n_skipped} zero values skipped)")

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }


# ==============================================================================
# SECTION 5: MODEL EVALUATION
# ==============================================================================


def evaluate_all_models(y_test: pd.Series,
                        predictions_dict: Dict[str,
                                               np.ndarray]) -> pd.DataFrame:
    """
    Evaluate all models and return results in a DataFrame.

    This function:
    1. Iterates through all models
    2. Calculates all metrics for each model
    3. Organizes results in a pandas DataFrame
    4. Identifies best model for each metric

    Args:
        y_test: Actual test values
        predictions_dict: Dictionary mapping model names to predictions

    Returns:
        pd.DataFrame: Results with models as rows, metrics as columns
    """
    print_section("EVALUATING ALL MODELS")

    results = []

    for model_name, y_pred in predictions_dict.items():
        print(f"\nEvaluating: {model_name}")
        print("-" * 40)

        # Calculate all metrics
        metrics = calculate_metrics(y_test.values, y_pred)

        # Print metrics for this model
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  MSE:  {metrics['MSE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.4f}%")

        # Store results
        results.append({
            'Model': model_name,
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE']
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Set model name as index
    results_df.set_index('Model', inplace=True)

    print("\n✓ Evaluation complete for all models")

    return results_df


# ==============================================================================
# SECTION 6: RESULTS FORMATTING AND SAVING
# ==============================================================================


def add_best_markers(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add star markers (★) to the best value in each metric column.

    For all metrics (MAE, MSE, RMSE, MAPE), lower is better.

    Args:
        results_df: DataFrame with evaluation results

    Returns:
        pd.DataFrame: Copy of results with best values marked
    """
    results_marked = results_df.copy()

    # For each metric column, find the minimum (best) value
    for col in ['MAE', 'MSE', 'RMSE', 'MAPE']:
        if col in results_marked.columns:
            # Find the minimum value in this column
            min_val = results_marked[col].min()

            # Add star to the best model(s)
            # Note: There could be ties, so we mark all rows with minimum value
            best_mask = results_marked[col] == min_val
            results_marked.loc[best_mask,
                               col] = results_marked.loc[best_mask,
                                                         col].apply(lambda x: f"{x:.4f} ★")
            # Format other values
            results_marked.loc[~best_mask,
                               col] = results_marked.loc[~best_mask,
                                                         col].apply(lambda x: f"{x:.4f}")

    return results_marked


def format_and_save_results(results_df: pd.DataFrame,
                            dataset_name: str) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Format results and save to both CSV and Markdown formats.

    This creates two output files:
    1. CSV: Machine-readable format for further analysis
    2. Markdown: Human-readable format with nice table formatting

    Args:
        results_df: DataFrame with evaluation metrics
        dataset_name: Name of dataset (for filenames)

    Returns:
        tuple: (marked_results_df, csv_path, md_path)
    """
    print_section("FORMATTING AND SAVING RESULTS")

    # Create output directory
    output_dir = Config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add dataset name to filenames
    csv_filename = f"{dataset_name}_{Config.METRICS_CSV}"
    md_filename = f"{dataset_name}_{Config.METRICS_MD}"

    csv_path = output_dir / csv_filename
    md_path = output_dir / md_filename

    # Add best markers for display
    results_marked = add_best_markers(results_df)

    # =========================================================================
    # SAVE AS CSV
    # =========================================================================
    print(f"\nSaving CSV to: {csv_path}")
    results_df.to_csv(csv_path, float_format='%.4f')
    print("  ✓ CSV saved (machine-readable format)")

    # =========================================================================
    # SAVE AS MARKDOWN
    # =========================================================================
    print(f"\nSaving Markdown to: {md_path}")

    with open(md_path, 'w', encoding='utf-8') as f:
        # Title
        f.write(
            f"# Model Evaluation Results - {dataset_name.upper()} Dataset\n\n")

        # Explanation
        f.write("## Metrics Explanation\n\n")
        f.write("- **MAE** (Mean Absolute Error): Average error magnitude\n")
        f.write("- **MSE** (Mean Squared Error): Average squared error\n")
        f.write("- **RMSE** (Root Mean Squared Error): Typical error size\n")
        f.write(
            "- **MAPE** (Mean Absolute Percentage Error): Average percentage error\n\n")
        f.write("**Lower is better for all metrics.** ★ marks the best model.\n\n")

        # Comparison table
        f.write("## Comparison Table\n\n")
        f.write(results_marked.to_markdown())
        f.write("\n\n")

        # Add interpretation guide
        f.write("## Interpretation Guide\n\n")
        f.write("### MAE vs RMSE\n")
        f.write("- If **RMSE >> MAE**: Model has some large errors (outliers)\n")
        f.write("- If **RMSE ≈ MAE**: Errors are consistent in size\n")
        f.write("- RMSE always ≥ MAE (equality only if all errors identical)\n\n")

        f.write("### MAPE Considerations\n")
        f.write("- Scale-independent metric (useful for comparing datasets)\n")
        f.write("- Undefined when actual values = 0\n")
        f.write("- Biased toward under-predictions\n\n")

    print("  ✓ Markdown saved (human-readable format)")

    return results_marked, csv_path, md_path


# ==============================================================================
# SECTION 7: COMPARISON ANALYSIS
# ==============================================================================


def analyze_baseline_improvement(results_df: pd.DataFrame) -> None:
    """
    Analyze improvement over naive baseline.

    THE BASELINE TEST:
    ==================
    This is the most important analysis in model evaluation.

    The naive baseline represents the absolute minimum performance:
    - Prediction strategy: "tomorrow = today"
    - Training time: 0 seconds
    - Complexity: Zero
    - Cost: Free

    If a model doesn't beat this baseline, it is USELESS because:
    1. You're better off using the simpler baseline (Occam's razor)
    2. The model adds complexity with no benefit
    3. In production, you'd deploy the baseline instead

    WHAT WE CALCULATE:
    - Best model by RMSE (most commonly used metric)
    - Percentage improvement over naive baseline
    - Whether the best model is actually worth deploying

    Args:
        results_df: DataFrame with evaluation metrics
    """
    print_section("BASELINE COMPARISON ANALYSIS")

    # Get naive baseline RMSE
    if 'naive_baseline' not in results_df.index:
        print("⚠️  WARNING: Naive baseline not found in results!")
        print("   Cannot calculate improvement over baseline.")
        return

    baseline_rmse = results_df.loc['naive_baseline', 'RMSE']
    print(f"\nNaive Baseline RMSE: {baseline_rmse:.4f}")
    print("This is our minimum acceptable performance threshold.")

    # Find best model by RMSE
    best_model_name = results_df['RMSE'].idxmin()
    best_model_rmse = results_df.loc[best_model_name, 'RMSE']

    print(f"\n{'=' * 80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'=' * 80}")
    print(f"RMSE: {best_model_rmse:.4f}")

    # Calculate improvement
    if best_model_name == 'naive_baseline':
        print("\n⚠️  RESULT: Naive baseline is the best model!")
        print("\n   INTERPRETATION:")
        print("   - None of the ML models beat the simple baseline")
        print("   - This suggests:")
        print("     1. The time series is very smooth (tomorrow ≈ today)")
        print("     2. Feature engineering didn't capture useful patterns")
        print("     3. Models may be overfitting to noise")
        print("\n   RECOMMENDATION:")
        print("   → Use the naive baseline in production (it's free and best!)")
        print("   → Try different features or model architectures")
        print("   → Consider if this data is predictable at all")
    else:
        # Calculate improvement
        improvement = ((baseline_rmse - best_model_rmse) / baseline_rmse) * 100

        print(f"\nImprovement over Naive Baseline: {improvement:.2f}%")

        if improvement > 0:
            print("\n✓ RESULT: Best model beats the baseline!")
            print(
                f"\n   The {best_model_name} is {
                    improvement:.2f}% more accurate.")

            # Contextualize the improvement
            if improvement < 5:
                print("\n   ⚠️  MARGINAL IMPROVEMENT (<5%)")
                print("   - The improvement is small")
                print("   - Consider if added complexity is worth it")
                print("   - Baseline might be sufficient for production")
            elif improvement < 15:
                print("\n   ✓ MODERATE IMPROVEMENT (5-15%)")
                print("   - Clear benefit over baseline")
                print("   - Worth deploying if complexity is acceptable")
            else:
                print("\n   ✓✓ SIGNIFICANT IMPROVEMENT (>15%)")
                print("   - Substantial benefit over baseline")
                print("   - Definitely worth deploying")
        else:
            print("\n✗ RESULT: Best model is WORSE than baseline!")
            print("\n   INTERPRETATION:")
            print("   - The ML model performs worse than simple persistence")
            print("   - This should NOT happen in a well-designed pipeline")
            print("\n   RECOMMENDATION:")
            print("   → Use the naive baseline instead")
            print("   → Debug the model/features/data")

    # Show full comparison table
    print(f"\n{'=' * 80}")
    print("FULL COMPARISON TABLE (sorted by RMSE)")
    print(f"{'=' * 80}")

    # Sort by RMSE and display
    sorted_results = results_df.sort_values('RMSE')
    print("\n" + sorted_results.to_string(float_format=lambda x: f"{x:.4f}"))


def print_detailed_summary(results_df: pd.DataFrame,
                           csv_path: Path,
                           md_path: Path,
                           dataset_name: str) -> None:
    """
    Print comprehensive summary of evaluation results.

    Args:
        results_df: DataFrame with evaluation metrics
        csv_path: Path to saved CSV file
        md_path: Path to saved Markdown file
        dataset_name: Name of dataset
    """
    print_section("EVALUATION SUMMARY")

    print(f"\nDataset: {dataset_name.upper()}")
    print(f"Models evaluated: {len(results_df)}")

    print("\n" + "=" * 80)
    print("BEST MODELS BY METRIC")
    print("=" * 80)

    for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
        best_model = results_df[metric].idxmin()
        best_value = results_df.loc[best_model, metric]
        print(f"\n{metric:5s}: {best_model:20s} ({best_value:.4f})")

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"  • CSV (machine-readable): {csv_path}")
    print(f"  • Markdown (human-readable): {md_path}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    # Check if any model significantly beats baseline
    if 'naive_baseline' in results_df.index:
        baseline_rmse = results_df.loc['naive_baseline', 'RMSE']
        best_rmse = results_df['RMSE'].min()
        improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100

        if improvement > 5:
            print(
                f"  ✓ ML models provide value (>{
                    improvement:.1f}% improvement)")
        elif improvement > 0:
            print(
                f"  ⚠️  Marginal improvement ({
                    improvement:.1f}%) - consider baseline")
        else:
            print("  ✗ Baseline is best - ML models not recommended")

    # Check consistency of errors
    for model in results_df.index:
        mae = results_df.loc[model, 'MAE']
        rmse = results_df.loc[model, 'RMSE']
        ratio = rmse / mae if mae > 0 else 0

        if ratio > 1.5:
            print(
                f"  ⚠️  {model}: Large outlier errors (RMSE/MAE = {ratio:.2f})")


# ==============================================================================
# SECTION 8: MAIN FUNCTION
# ==============================================================================


def main(dataset_name: str = 'voltage') -> None:
    """
    Main evaluation pipeline.

    Workflow:
    1. Load test data (actual values)
    2. Load model predictions
    3. Calculate metrics for all models
    4. Create comparison tables
    5. Analyze baseline improvement
    6. Save results
    7. Print summary

    Args:
        dataset_name: Which dataset to evaluate ('voltage' or 'missions')
    """
    print("=" * 80)
    print(
        f"TIME SERIES FORECASTING - MODEL EVALUATION: {dataset_name.upper()}")
    print("=" * 80)
    print("\nThis script evaluates all trained models using:")
    print("  • MAE  - Mean Absolute Error (average error magnitude)")
    print("  • MSE  - Mean Squared Error (penalizes large errors)")
    print("  • RMSE - Root Mean Squared Error (interpretable, penalizes outliers)")
    print("  • MAPE - Mean Absolute Percentage Error (scale-independent)")
    print("\nThe critical test: Does any model beat the naive baseline?")
    print("=" * 80)

    # 1. Load test data
    y_test = load_test_data(dataset_name)

    # 2. Load training summary with predictions
    training_summary = load_training_summary(dataset_name)
    predictions_dict = training_summary['predictions']

    # 3. Evaluate all models
    results_df = evaluate_all_models(y_test, predictions_dict)

    # 4. Format and save results
    results_marked, csv_path, md_path = format_and_save_results(
        results_df, dataset_name)

    # 5. Analyze baseline improvement
    analyze_baseline_improvement(results_df)

    # 6. Print detailed summary
    print_detailed_summary(results_df, csv_path, md_path, dataset_name)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review {md_path.name} for detailed comparison")
    print("  2. If models beat baseline: Deploy best model")
    print("  3. If baseline is best: Use baseline (it's free!)")
    print("  4. Consider hyperparameter tuning or feature engineering")
    print("=" * 80)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import sys

    # Allow user to specify dataset via command line argument
    # Usage: python 04_evaluate_metrics.py voltage
    #    or: python 04_evaluate_metrics.py missions
    #    or: python 04_evaluate_metrics.py both (evaluates both)

    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
        if dataset not in ['voltage', 'missions', 'both']:
            print(
                f"Error: Invalid dataset '{dataset}'. Choose 'voltage', 'missions', or 'both'")
            sys.exit(1)
    else:
        dataset = 'both'  # Default: evaluate both datasets

    # Evaluate datasets
    if dataset == 'both':
        datasets_to_evaluate = ['voltage', 'missions']
    else:
        datasets_to_evaluate = [dataset]

    # Evaluate each dataset
    for ds in datasets_to_evaluate:
        print(f"\n{'=' * 80}")
        print(f"EVALUATING DATASET: {ds.upper()}")
        print(f"{'=' * 80}")
        main(ds)
