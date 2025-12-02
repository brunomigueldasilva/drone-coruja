#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLIGHT TELEMETRY - MODEL EVALUATION METRICS
==============================================================================

Purpose: Evaluate and compare regression models using test set metrics

This script:
1. Loads test target (y_test) and predictions from all trained models
2. Calculates evaluation metrics for each model:
   - R² (R-squared / Coefficient of Determination)
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
3. Creates comparative table of all models ranked by RMSE
4. Identifies and saves the best performing model
5. Provides detailed discussion of metrics and their operational implications

Metrics Interpretation:
- R²: Proportion of variance explained (0 to 1, higher is better)
      Can be negative if model performs worse than horizontal line
- MAE: Average absolute prediction error (same units as target)
      Robust to outliers, treats all errors equally
- MSE: Average squared prediction error (squared units)
      Heavily penalizes large errors due to squaring
- RMSE: Square root of MSE (same units as target)
        More interpretable than MSE, penalizes large errors

Model Selection Criterion:
- Primary: RMSE (penalizes large errors, important for operational safety)
- RMSE is preferred over MAE when large prediction errors are particularly costly

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Evaluation configuration parameters."""
    # Directories
    OUTPUT_DIR = Path('outputs')
    ARTIFACTS_DIR = OUTPUT_DIR / 'models'
    TABLES_DIR = OUTPUT_DIR / 'results'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'

    # Input files
    Y_TEST_FILE = ARTIFACTS_DIR / 'y_test.pkl'

    # Output files
    RESULTS_CSV = TABLES_DIR / 'model_comparison.csv'
    RESULTS_MD = TABLES_DIR / 'model_comparison.md'
    BEST_MODEL_TXT = ARTIFACTS_DIR / 'best_model.txt'

    # Model information
    MODEL_INFO = {
        'Simple Linear Regression': 'preds_linear_simple.csv',
        'Multiple Linear Regression': 'preds_linear_multiple.csv',
        'Ridge Regression': 'preds_ridge.csv',
        'Lasso Regression': 'preds_lasso.csv',
        'Polynomial Regression (degree=2)': 'preds_polynomial_deg2.csv'
    }

    # Display options
    DECIMAL_PLACES = 4


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=", width: int = 120) -> None:
    """
    Print formatted section header.

    Args:
        title: Section title
        char: Character to use for separator
        width: Width of separator line
    """
    print("\n" + char * width)
    print(title)
    print(char * width)


def print_subsection(title: str, width: int = 120) -> None:
    """
    Print formatted subsection header.

    Args:
        title: Subsection title
        width: Width of separator line
    """
    print("\n" + title)
    print("-" * width)


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_test_target() -> np.ndarray:
    """
    Load ground truth test labels.

    Returns:
        np.ndarray: True test labels (y_test)

    Raises:
        SystemExit: If ground truth file not found
    """
    print_section("[SECTION 1] LOAD TEST TARGET AND PREDICTIONS")

    if not Config.Y_TEST_FILE.exists():
        print(f"✗ ERROR: Test target file not found: {Config.Y_TEST_FILE}")
        print("  Please run the training script first.")
        exit(1)

    print("Loading test target (ground truth)...")
    with open(Config.Y_TEST_FILE, 'rb') as f:
        y_test = pickle.load(f)

    print(f"✓ y_test loaded from: {Config.Y_TEST_FILE}")
    print(f"  Shape: {y_test.shape}")
    print(f"  Number of test samples: {len(y_test)}")

    return y_test


def load_all_predictions(y_test: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Load all model predictions and validate against ground truth.

    Args:
        y_test: Ground truth array (for validation)

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping model names to predictions

    Raises:
        SystemExit: If prediction files not found
    """
    print(f"\nLoading predictions for {len(Config.MODEL_INFO)} models...")
    print()

    predictions = {}

    for model_name, pred_file in Config.MODEL_INFO.items():
        pred_path = Config.PREDICTIONS_DIR / pred_file

        if not pred_path.exists():
            print(f"✗ ERROR: Prediction file not found: {pred_path}")
            continue

        print(f"Loading predictions for: {model_name}")
        pred_df = pd.read_csv(pred_path)

        # Verify that y_true matches y_test
        if not np.allclose(pred_df['y_true'].values, y_test):
            print(f"  WARNING: y_true in {pred_file} does not match y_test!")

        # Store predictions
        predictions[model_name] = pred_df['y_pred'].values

        print(f"  ✓ Loaded from: {pred_path}")
        print(f"    Predictions shape: {predictions[model_name].shape}")
        print()

    print("✓ All predictions loaded successfully")

    return predictions


# ==============================================================================
# SECTION 4: METRICS CALCULATION
# ==============================================================================


def print_metrics_overview() -> None:
    """Print detailed overview of evaluation metrics and their interpretation."""
    print_subsection("[SECTION 2] CALCULATE EVALUATION METRICS FOR TEST SET")

    print("Computing metrics for each model...")
    print()

    print("METRICS OVERVIEW:")
    print()

    print("R² (R-SQUARED / COEFFICIENT OF DETERMINATION):")
    print("  • Formula: R² = 1 - (SS_residual / SS_total)")
    print("  • Range: (-∞, 1], where 1 is perfect prediction")
    print("  • Interpretation: Proportion of variance in target explained by model")
    print("  • R² = 1.0: Perfect predictions (y_pred = y_true for all samples)")
    print("  • R² = 0.0: Model performs as well as predicting the mean of y_train")
    print("  • R² < 0.0: Model performs WORSE than predicting the mean (red flag!)")
    print("  • CAUTION: R² can be misleading if used alone; always check residuals")
    print()

    print("MAE (MEAN ABSOLUTE ERROR):")
    print("  • Formula: MAE = (1/n) × Σ|y_true - y_pred|")
    print("  • Units: Same as target variable (minutes in our case)")
    print("  • Interpretation: Average magnitude of prediction errors")
    print("  • ADVANTAGE: Robust to outliers (absolute value, not squared)")
    print("  • ADVANTAGE: Easy to interpret (e.g., 'off by 5 minutes on average')")
    print("  • LIMITATION: Treats all errors equally (10 min error = 10 × 1 min errors)")
    print("  • Lower is better (0 = perfect)")
    print()

    print("MSE (MEAN SQUARED ERROR):")
    print("  • Formula: MSE = (1/n) × Σ(y_true - y_pred)²")
    print("  • Units: Squared units of target (minutes² in our case)")
    print("  • Interpretation: Average squared prediction errors")
    print("  • ADVANTAGE: Differentiable everywhere (good for optimization)")
    print("  • ADVANTAGE: Heavily penalizes large errors (due to squaring)")
    print("  • LIMITATION: Squared units make it hard to interpret directly")
    print("  • LIMITATION: Very sensitive to outliers")
    print("  • Lower is better (0 = perfect)")
    print()

    print("RMSE (ROOT MEAN SQUARED ERROR):")
    print("  • Formula: RMSE = √MSE = √[(1/n) × Σ(y_true - y_pred)²]")
    print("  • Units: Same as target variable (minutes in our case)")
    print("  • Interpretation: Standard deviation of prediction errors")
    print("  • ADVANTAGE: Same units as MAE, easier to interpret than MSE")
    print("  • ADVANTAGE: Penalizes large errors more than MAE (due to squaring before averaging)")
    print("  • LIMITATION: Sensitive to outliers (like MSE)")
    print("  • Lower is better (0 = perfect)")
    print()

    print("MAE vs RMSE COMPARISON:")
    print("  • If RMSE ≈ MAE: Errors are relatively uniform in magnitude")
    print("  • If RMSE >> MAE: Some predictions have very large errors (outliers present)")
    print("  • RMSE will always be ≥ MAE (due to Jensen's inequality)")
    print()
    print("-" * 120)
    print()


def calculate_all_metrics(
        predictions: Dict[str, np.ndarray], y_test: np.ndarray) -> pd.DataFrame:
    """
    Calculate evaluation metrics for all models.

    Args:
        predictions: Dictionary of model predictions
        y_test: Ground truth values

    Returns:
        pd.DataFrame: DataFrame with metrics for all models
    """
    results = []

    for model_name, y_pred in predictions.items():
        print(f"Evaluating: {model_name}")

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Store results
        results.append({
            'Model': model_name,
            'R²': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        })

        print(f"  R²   = {r2:.4f}")
        print(f"  MAE  = {mae:.4f} minutes")
        print(f"  MSE  = {mse:.4f} minutes²")
        print(f"  RMSE = {rmse:.4f} minutes")
        print()

    print("✓ Metrics calculated for all models")

    return pd.DataFrame(results)


# ==============================================================================
# SECTION 5: MODEL RANKING AND COMPARISON
# ==============================================================================


def rank_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank models by RMSE and display comparison.

    Args:
        metrics_df: DataFrame with calculated metrics

    Returns:
        pd.DataFrame: Sorted DataFrame (best model first)
    """
    print_subsection(
        "[SECTION 3] CREATE COMPARATIVE TABLE AND IDENTIFY BEST MODEL")

    # Sort by RMSE (lower is better)
    metrics_sorted = metrics_df.sort_values(
        'RMSE', ascending=True).reset_index(
        drop=True)
    metrics_sorted.index = metrics_sorted.index + 1  # Start ranking from 1

    print("Ranking models by RMSE (lower is better)...")
    print()
    print("COMPARATIVE TABLE:")
    print()
    print(metrics_sorted.to_string(index=True))
    print()

    # Identify best model
    best_model = metrics_sorted.iloc[0]
    print("=" * 120)
    print("BEST MODEL IDENTIFIED")
    print("=" * 120)
    print(f"Model: {best_model['Model']}")
    print(f"RMSE:  {best_model['RMSE']:.4f} minutes")
    print(f"R²:    {best_model['R²']:.4f}")
    print(f"MAE:   {best_model['MAE']:.4f} minutes")
    print(f"MSE:   {best_model['MSE']:.4f} minutes²")
    print("=" * 120)

    return metrics_sorted


# ==============================================================================
# SECTION 6: RESULTS SAVING
# ==============================================================================


def save_results(metrics_df: pd.DataFrame) -> tuple:
    """
    Save evaluation results to CSV, Markdown, and best model text file.

    Args:
        metrics_df: Sorted DataFrame with metrics

    Returns:
        tuple: Paths to saved files (csv_path, md_path, txt_path)
    """
    print_subsection("[SECTION 4] SAVE RESULTS TO FILES")

    # Ensure output directories exist
    Config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    Config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    metrics_df.to_csv(Config.RESULTS_CSV, index=False)
    print(f"✓ Results saved to CSV: {Config.RESULTS_CSV}")

    # Save to Markdown
    best_model = metrics_df.iloc[0]

    with open(Config.RESULTS_MD, 'w') as f:
        f.write("# Model Evaluation Results - Flight Telemetry\n\n")
        f.write("## Comparative Metrics Table\n\n")
        f.write("Models ranked by RMSE (lower is better):\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Best Performing Model\n\n")
        f.write(f"**Model**: {best_model['Model']}\n\n")
        f.write("**Metrics**:\n")
        f.write(f"- RMSE: {best_model['RMSE']:.4f} minutes\n")
        f.write(f"- R²: {best_model['R²']:.4f}\n")
        f.write(f"- MAE: {best_model['MAE']:.4f} minutes\n")
        f.write(f"- MSE: {best_model['MSE']:.4f} minutes²\n")

    print(f"✓ Results saved to Markdown: {Config.RESULTS_MD}")

    # Save best model information with detailed format
    with open(Config.BEST_MODEL_TXT, 'w') as f:
        f.write("BEST MODEL - FLIGHT TELEMETRY REGRESSION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model Name: {best_model['Model']}\n\n")
        f.write("Test Set Performance:\n")
        f.write(f"  RMSE: {best_model['RMSE']:.4f} minutes\n")
        f.write(f"  MAE:  {best_model['MAE']:.4f} minutes\n")
        f.write(f"  MSE:  {best_model['MSE']:.4f} minutes²\n")
        f.write(f"  R²:   {best_model['R²']:.4f}\n\n")
        f.write("Selection Criterion: Lowest RMSE on test set\n")

    print(f"✓ Best model info saved to: {Config.BEST_MODEL_TXT}")

    print()
    print("Output files:")
    print(f"  • {Config.RESULTS_CSV.name} (spreadsheet format)")
    print(f"  • {Config.RESULTS_MD.name} (formatted report)")
    print(f"  • {Config.BEST_MODEL_TXT.name} (best model reference)")

    return Config.RESULTS_CSV, Config.RESULTS_MD, Config.BEST_MODEL_TXT


# ==============================================================================
# SECTION 7: DETAILED ANALYSIS AND INSIGHTS
# ==============================================================================


def print_detailed_analysis(metrics_df: pd.DataFrame) -> None:
    """
    Print detailed analysis and operational insights.

    Args:
        metrics_df: Sorted DataFrame with metrics
    """
    print_subsection(
        "[SECTION 5] DETAILED METRICS INTERPRETATION AND INSIGHTS")

    best_model = metrics_df.iloc[0]
    best_rmse = best_model['RMSE']
    best_mae = best_model['MAE']

    print("WHY TEST SET ONLY? (Training Metrics Would Be Misleading)")
    print()
    print("SCENARIO: Training a model that memorizes instead of learning patterns")
    print()
    print("If we evaluated on TRAINING SET:")
    print("  Model: Memorization Model (overfitting)")
    print("  Training RMSE: 0.001 minutes (nearly perfect!)")
    print("  Training R²: 0.999 (explains 99.9% of variance!)")
    print()
    print("  CONCLUSION: This model looks AMAZING... but it's lying!")
    print()
    print("If we evaluated on TEST SET (unseen data):")
    print("  Model: Memorization Model (overfitting)")
    print("  Test RMSE: 45.2 minutes (terrible!)")
    print("  Test R²: 0.12 (explains only 12% of variance)")
    print()
    print("  CONCLUSION: Model fails completely on new data!")
    print()
    print("KEY INSIGHT:")
    print("  • Training metrics measure how well model FITS the training data")
    print("  • Test metrics measure how well model GENERALIZES to new data")
    print("  • Only test metrics reveal true predictive performance")
    print("  • A model that memorizes training data will fail in production")
    print()
    print("ANALOGY:")
    print("  Training set = Study materials for an exam")
    print("  Test set = The actual exam (with different questions)")
    print("  • Memorizing answers to study questions ≠ understanding the subject")
    print("  • Only the exam score reveals true understanding")
    print()
    print("-" * 120)
    print()

    print("OPERATIONAL CONTEXT: UNDER-PREDICTION vs OVER-PREDICTION COSTS")
    print()
    print("In flight duration prediction, errors have asymmetric costs:")
    print()

    print("UNDER-PREDICTION (predicting shorter duration than actual):")
    print("  Operational Impact:")
    print("    • Aggressive scheduling: Aircraft assigned to next flight too soon")
    print("    • Crew scheduling conflicts: Crew might be assigned elsewhere")
    print("    • Passenger connections at risk: Tight connection times")
    print("    • Fuel planning issues: Might underestimate fuel requirements")
    print()
    print("  Consequences:")
    print("    • Flight delays cascade through network")
    print("    • Passenger dissatisfaction and missed connections")
    print("    • Increased operational costs (crew overtime, rebooking)")
    print("    • Potential safety concerns (rushed operations)")
    print()
    print("  Cost Profile: HIGH COST")
    print("    → Delays have significant financial and reputational impact")
    print()

    print("OVER-PREDICTION (predicting longer duration than actual):")
    print("  Operational Impact:")
    print("    • Conservative scheduling: More buffer time than needed")
    print("    • Aircraft idle time: Plane sits waiting for next assignment")
    print("    • Reduced fleet utilization: Fewer flights per aircraft per day")
    print("    • Higher connection buffer times: Passengers wait longer")
    print()
    print("  Consequences:")
    print("    • Reduced operational efficiency")
    print("    • Lower revenue potential (fewer flights)")
    print("    • Passenger inconvenience (longer layovers)")
    print("    • Competitive disadvantage (longer total trip times)")
    print()
    print("  Cost Profile: MODERATE COST")
    print("    → Inefficiency is costly but less disruptive than delays")
    print()

    print("METRIC SELECTION BASED ON COST ASYMMETRY:")
    print()
    print("  • MAE: Treats under-prediction and over-prediction equally")
    print("    → Suitable when costs are symmetric")
    print()
    print("  • RMSE: Penalizes large errors more heavily (squared before averaging)")
    print("    → Better when large errors are particularly problematic")
    print("    → Our choice: Large under-predictions cause cascading delays")
    print()
    print("  • Custom metrics (future consideration):")
    print("    → Asymmetric loss functions (penalize under-predictions more)")
    print("    → Quantile regression (predict 90th percentile for safety buffer)")
    print("    → Service level constraints (guarantee accuracy within X minutes)")
    print()

    print("-" * 120)
    print()

    print("MODEL COMPARISON INSIGHTS:")
    print()

    # Analyze model differences
    for idx, row in metrics_df.iterrows():
        rank = idx + 1
        model = row['Model']
        r2 = row['R²']
        mae = row['MAE']
        rmse = row['RMSE']

        print(f"Rank {rank}: {model}")
        print(f"  • R² = {r2:.4f}, MAE = {mae:.4f} min, RMSE = {rmse:.4f} min")

        if rank == 1:
            print("  • BEST MODEL: Achieves lowest RMSE")
        else:
            rmse_diff = rmse - best_rmse
            mae_diff = mae - best_mae
            print(
                f"  • RMSE is {
                    rmse_diff:.4f} min higher than best ({
                    (
                        rmse_diff /
                        best_rmse) *
                    100:.1f}% worse)")
            print(
                f"  • MAE is {
                    mae_diff:.4f} min higher than best ({
                    (
                        mae_diff /
                        best_mae) *
                    100:.1f}% worse)")
        print()

    print("-" * 120)


# ==============================================================================
# SECTION 8: RECOMMENDATIONS AND NEXT STEPS
# ==============================================================================


def print_recommendations(metrics_df: pd.DataFrame) -> None:
    """
    Print actionable recommendations based on evaluation results.

    Args:
        metrics_df: Sorted DataFrame with metrics
    """
    print_subsection("[SECTION 6] RECOMMENDATIONS AND NEXT STEPS")

    best_model = metrics_df.iloc[0]
    best_model_name = best_model['Model']
    best_rmse = best_model['RMSE']

    print("CURRENT STATUS:")
    print(f"  • Best model identified: {best_model_name}")
    print(f"  • Test set RMSE: {best_rmse:.4f} minutes")
    print("  • All models evaluated and compared")
    print()

    print("RECOMMENDATIONS FOR PRODUCTION DEPLOYMENT:")
    print()
    print("1. RESIDUAL ANALYSIS (CRITICAL - DO NEXT):")
    print("   • Plot residuals (y_true - y_pred) vs predicted values")
    print("   • Check for patterns: Should be randomly scattered around zero")
    print("   • Check for heteroscedasticity: Variance should be constant")
    print("   • Check for normality: Residuals should be approximately normal")
    print("   • Identify outliers: Investigate samples with large errors")
    print()

    print("2. FEATURE IMPORTANCE ANALYSIS:")
    print("   • Examine coefficients of best model")
    print("   • Understand which features drive predictions")
    print("   • Validate that importance aligns with domain knowledge")
    print()

    print("3. CROSS-VALIDATION:")
    print("   • Current evaluation uses single train/test split")
    print("   • K-fold cross-validation provides more robust estimate")
    print("   • Helps detect variance in model performance")
    print()

    print("4. HYPERPARAMETER TUNING:")
    print("   • Ridge/Lasso: Grid search over alpha values")
    print("   • Polynomial: Test degree=3 (with regularization)")
    print("   • Use cross-validation to select optimal hyperparameters")
    print()

    print("5. EXPLORE ADVANCED MODELS:")
    print("   • Random Forest: Handles non-linearity, interactions automatically")
    print("   • Gradient Boosting (XGBoost, LightGBM): Often best performance")
    print("   • Neural Networks: If dataset grows larger")
    print()

    print("6. OPERATIONAL CONSIDERATIONS:")
    print("   • Define acceptable error threshold (e.g., ±5 minutes)")
    print("   • Calculate percentage of predictions within threshold")
    print("   • Consider asymmetric loss if under-prediction is more costly")
    print("   • Implement prediction intervals (uncertainty quantification)")
    print()

    print("7. MONITORING AND MAINTENANCE:")
    print("   • Track model performance over time (concept drift)")
    print("   • Retrain periodically with new data")
    print("   • Alert if performance degrades below threshold")
    print("   • Version control for models and data pipelines")


# ==============================================================================
# SECTION 9: FINAL SUMMARY
# ==============================================================================


def print_final_summary(metrics_df: pd.DataFrame, saved_files: tuple) -> None:
    """
    Print final evaluation summary.

    Args:
        metrics_df: Sorted DataFrame with metrics
        saved_files: Tuple of saved file paths (csv_path, md_path, txt_path)
    """
    print_section("MODEL EVALUATION COMPLETED SUCCESSFULLY")

    best_model = metrics_df.iloc[0]
    best_model_name = best_model['Model']
    best_rmse = best_model['RMSE']
    results_csv_path, results_md_path, best_model_txt_path = saved_files

    print()
    print("SUMMARY:")
    print()
    print(
        f"  • Evaluated {len(Config.MODEL_INFO)} regression models on test set")
    print("  • Calculated 4 metrics: R², MAE, MSE, RMSE")
    print(f"  • Best model: {best_model_name}")
    print(f"  • Best RMSE: {best_rmse:.4f} minutes")
    print()

    print("FILES SAVED:")
    print(f"  • {results_csv_path}")
    print(f"  • {results_md_path}")
    print(f"  • {best_model_txt_path}")
    print()

    print("KEY INSIGHTS:")
    print()
    print("  METRICS INTERPRETATION:")
    print("    • R²: Proportion of variance explained by model")
    print("    • MAE: Average absolute error (robust to outliers)")
    print("    • RMSE: Standard deviation of errors (penalizes large errors)")
    print()
    print("  OPERATIONAL CONTEXT:")
    print("    • Under-prediction (shorter duration): HIGH COST (delays, safety)")
    print("    • Over-prediction (longer duration): MODERATE COST (inefficiency)")
    print("    • RMSE is appropriate metric (penalizes large under-predictions)")
    print()
    print("  WHY TEST SET ONLY:")
    print("    • Training metrics would be biased (overly optimistic)")
    print("    • Test set simulates real-world unseen data")
    print("    • Reveals true generalization capability")
    print()

    print("NEXT STEPS:")
    print("  1. Perform residual analysis to validate model assumptions")
    print("  2. Analyze prediction errors for outliers and patterns")
    print("  3. Consider cross-validation for robust performance estimate")
    print("  4. Tune hyperparameters of best model family")
    print("  5. Explore advanced models (Random Forest, Gradient Boosting)")
    print()

    print("=" * 120)
    print()
    print("✅ Script completed successfully!")


# ==============================================================================
# SECTION 10: MAIN EXECUTION
# ==============================================================================


def main():
    """Main evaluation pipeline execution."""
    print("=" * 120)
    print("MODEL EVALUATION - FLIGHT TELEMETRY")
    print("=" * 120)

    try:
        # 1. Load test target
        y_test = load_test_target()

        # 2. Load all predictions
        predictions = load_all_predictions(y_test)

        # 3. Print metrics overview and calculate metrics
        print_metrics_overview()
        metrics_df = calculate_all_metrics(predictions, y_test)

        # 4. Rank models
        metrics_sorted = rank_models(metrics_df)

        # 5. Save results
        saved_files = save_results(metrics_sorted)

        # 6. Print detailed analysis
        print_detailed_analysis(metrics_sorted)

        # 7. Print recommendations
        print_recommendations(metrics_sorted)

        # 8. Print final summary
        print_final_summary(metrics_sorted, saved_files)

    except Exception as e:
        print("\n✗ ERROR during evaluation:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
