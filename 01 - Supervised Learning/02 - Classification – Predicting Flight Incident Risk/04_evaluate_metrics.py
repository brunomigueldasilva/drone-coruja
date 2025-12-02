#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
AIRCRAFT INCIDENT CLASSIFICATION - MODEL EVALUATION
==============================================================================

Purpose: Evaluate and compare trained classification models using comprehensive metrics

This script:
1. Loads predictions from all trained models
2. Detects problem type (binary vs multiclass classification)
3. Calculates comprehensive evaluation metrics for each model
4. Performs threshold tuning for binary classification
5. Generates comparative tables with best model identification
6. Saves results in CSV and Markdown formats
7. Provides insights about metric interpretation and business impact

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Evaluation configuration parameters."""
    PROCESSED_DATA_DIR = Path('outputs/data_processed')
    PREDICTIONS_DIR = Path('outputs/predictions')
    TABLES_DIR = Path('outputs/results')

    # Threshold tuning parameters
    THRESHOLD_START = 0.05
    THRESHOLD_END = 0.96
    THRESHOLD_STEP = 0.01

    # Output files
    OUTPUT_CSV = Path('outputs/results/results_metrics.csv')
    OUTPUT_MD = Path('outputs/results/results_metrics.md')


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

    print("\n" + char * 80)
    print(title)
    print(char * 80)


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Specificity (True Negative Rate).

    Specificity = TN / (TN + FP)
    Measures: Of all actual negatives, how many did we correctly identify?

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        float: Specificity score
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_ground_truth() -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load ground truth labels and calculate class statistics.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: y_test array and statistics dict

    Raises:
        SystemExit: If ground truth file not found
    """
    print_section("1. LOADING GROUND TRUTH")

    y_test_path = Config.PROCESSED_DATA_DIR / 'y_test.csv'

    if not y_test_path.exists():
        print(f"✗ ERROR: File not found: {y_test_path}")
        print("  Please run 02_preprocessing.py first.")
        exit(1)

    y_test = pd.read_csv(y_test_path).values.ravel()
    print(f"✓ Ground truth loaded: {len(y_test)} samples")

    # Calculate statistics
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)

    print(f"\nNumber of unique classes: {n_classes}")
    print(f"Classes: {unique_classes}")

    # Class distribution
    print("\nClass distribution in test set:")
    test_counts = pd.Series(y_test).value_counts().sort_index()
    for cls in test_counts.index:
        count = test_counts[cls]
        pct = (count / len(y_test)) * 100
        print(f"  Class {cls}: {count:,} samples ({pct:.2f}%)")

    stats = {
        'n_classes': n_classes,
        'unique_classes': unique_classes,
        'test_counts': test_counts,
        'problem_type': 'BINARY' if n_classes == 2 else 'MULTICLASS'
    }

    # Imbalance ratio for binary
    if n_classes == 2:
        imbalance_ratio = test_counts.max() / test_counts.min()
        stats['imbalance_ratio'] = imbalance_ratio
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 1.5:
            print(
                "  ⚠️  Dataset is imbalanced - F1 and PR-AUC more informative than accuracy")

    return y_test, stats


def load_all_predictions() -> Dict[str, pd.DataFrame]:
    """
    Load predictions from all trained models.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping model names to prediction DataFrames

    Raises:
        SystemExit: If predictions directory not found
    """
    print_section("2. LOADING MODEL PREDICTIONS")

    if not Config.PREDICTIONS_DIR.exists():
        print(f"✗ ERROR: Directory not found: {Config.PREDICTIONS_DIR}")
        print("  Please run 03_train_models.py first.")
        exit(1)

    prediction_files = sorted(Config.PREDICTIONS_DIR.glob('predictions_*.csv'))

    if len(prediction_files) == 0:
        print(
            f"✗ ERROR: No prediction files found in {
                Config.PREDICTIONS_DIR}")
        print("  Please run 03_train_models.py first.")
        exit(1)

    print(f"Found {len(prediction_files)} prediction files:\n")

    predictions_dict = {}

    for pred_file in prediction_files:
        model_name = pred_file.stem.replace('predictions_', '')
        pred_df = pd.read_csv(pred_file)
        predictions_dict[model_name] = pred_df

        proba_cols = [
            col for col in pred_df.columns if col.startswith('proba_class_')]
        has_proba = len(proba_cols) > 0

        print(f"  ✓ {model_name:<25} {len(pred_df)} samples, probabilities: "
              f"{'Yes' if has_proba else 'No'}")

    return predictions_dict


# ==============================================================================
# SECTION 4: METRIC CALCULATION
# ==============================================================================


def calculate_binary_metrics(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_proba: Optional[np.ndarray] = None) -> Dict[str,
                                                                           float]:
    """
    Calculate all metrics for binary classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Probability estimates for positive class (optional)

    Returns:
        Dict[str, float]: Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': calculate_specificity(y_true, y_pred)
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    else:
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan

    return metrics


def calculate_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None,
                                 n_classes: int = 0) -> Dict[str, float]:
    """
    Calculate all metrics for multiclass classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Probability estimates for all classes (optional)
        n_classes: Number of classes

    Returns:
        Dict[str, float]: Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0)
    }

    if y_proba is not None and y_proba.shape[1] == n_classes:
        try:
            metrics['roc_auc_macro'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro')
        except Exception:
            metrics['roc_auc_macro'] = np.nan
    else:
        metrics['roc_auc_macro'] = np.nan

    return metrics


def evaluate_all_models(predictions_dict: Dict[str,
                                               pd.DataFrame],
                        y_test: np.ndarray,
                        stats: Dict[str,
                                    Any]) -> pd.DataFrame:
    """
    Evaluate all models and return metrics DataFrame.

    Args:
        predictions_dict: Dictionary with model predictions
        y_test: Ground truth labels
        stats: Statistics dictionary with problem type info

    Returns:
        pd.DataFrame: Metrics for all models
    """
    print_section("3. CALCULATING EVALUATION METRICS")

    problem_type = stats['problem_type']
    n_classes = stats['n_classes']

    print(f"\nProblem type: {problem_type} CLASSIFICATION")
    if problem_type == "BINARY":
        print("Metrics: Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC")
    else:
        print("Metrics: Accuracy, Precision/Recall/F1 (macro & micro), ROC-AUC (macro)")

    metrics_results = {}

    for model_name, pred_df in predictions_dict.items():
        print(f"\n{'-' * 80}")
        print(f"Evaluating: {model_name}")
        print(f"{'-' * 80}")

        y_true = pred_df['y_true'].values
        y_pred = pred_df['y_pred'].values

        # Extract probabilities
        proba_cols = [
            col for col in pred_df.columns if col.startswith('proba_class_')]
        y_proba = None

        if problem_type == "BINARY":
            if 'proba_class_1' in pred_df.columns:
                y_proba = pred_df['proba_class_1'].values
            elif 'proba_class_0' in pred_df.columns:
                y_proba = 1 - pred_df['proba_class_0'].values

            metrics = calculate_binary_metrics(y_true, y_pred, y_proba)

            # Display metrics
            print(f"  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  Recall:      {metrics['recall']:.4f}")
            print(f"  F1 Score:    {metrics['f1']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            if not np.isnan(metrics['roc_auc']):
                print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
                print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            print("\n  Confusion Matrix:")
            print(f"    TN={tn:>4}  FP={fp:>4}")
            print(f"    FN={fn:>4}  TP={tp:>4}")

        else:  # MULTICLASS
            if len(proba_cols) == n_classes:
                y_proba = pred_df[proba_cols].values

            metrics = calculate_multiclass_metrics(
                y_true, y_pred, y_proba, n_classes)

            # Display metrics
            print(f"  Accuracy:          {metrics['accuracy']:.4f}")
            print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
            print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
            print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
            print(f"  Precision (micro): {metrics['precision_micro']:.4f}")
            print(f"  Recall (micro):    {metrics['recall_micro']:.4f}")
            print(f"  F1 (micro):        {metrics['f1_micro']:.4f}")
            if not np.isnan(metrics['roc_auc_macro']):
                print(f"  ROC-AUC (macro):   {metrics['roc_auc_macro']:.4f}")

        metrics_results[model_name] = metrics

    # Create DataFrame
    results_df = pd.DataFrame(metrics_results).T

    # Sort by F1 score
    sort_column = 'f1' if problem_type == "BINARY" else 'f1_macro'
    results_df = results_df.sort_values(sort_column, ascending=False)

    return results_df


# ==============================================================================
# SECTION 5: THRESHOLD TUNING
# ==============================================================================


def tune_threshold(best_model_name: str,
                   predictions_dict: Dict[str,
                                          pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Perform threshold tuning for best binary classification model.

    Args:
        best_model_name: Name of best performing model
        predictions_dict: Dictionary with all predictions

    Returns:
        Optional[pd.DataFrame]: Threshold tuning results or None if not applicable
    """
    print_section("4. THRESHOLD TUNING (BINARY CLASSIFICATION)")

    best_pred_df = predictions_dict[best_model_name]

    if 'proba_class_1' not in best_pred_df.columns:
        print(
            f"\n⚠️  Model {best_model_name} does not have probability estimates")
        print("   Threshold tuning not possible")
        return None

    print("\nTHRESHOLD TUNING RATIONALE:")
    print("  • Default threshold: 0.5 (predict class 1 if P(class=1) > 0.5)")
    print("  • Lowering threshold → Higher Recall (fewer missed incidents)")
    print("  • Raising threshold → Higher Precision (fewer false alarms)")
    print("  • Business context should drive threshold selection")

    print(f"\nTuning threshold for: {best_model_name}\n")

    y_true = best_pred_df['y_true'].values
    y_proba = best_pred_df['proba_class_1'].values

    thresholds = np.arange(
        Config.THRESHOLD_START,
        Config.THRESHOLD_END,
        Config.THRESHOLD_STEP)
    threshold_results = []

    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)

        threshold_results.append({
            'threshold': threshold,
            'precision': precision_score(y_true, y_pred_thresh, zero_division=0),
            'recall': recall_score(y_true, y_pred_thresh, zero_division=0),
            'f1': f1_score(y_true, y_pred_thresh, zero_division=0)
        })

    threshold_df = pd.DataFrame(threshold_results)

    # Find optimal thresholds
    best_f1_row = threshold_df.loc[threshold_df['f1'].idxmax()]
    best_recall_row = threshold_df.loc[threshold_df['recall'].idxmax()]

    print("OPTIMAL THRESHOLDS:")
    print("\n📊 Threshold maximizing F1 Score:")
    print(f"    Threshold: {best_f1_row['threshold']:.2f}")
    print(f"    F1:        {best_f1_row['f1']:.4f}")
    print("    (Balanced precision-recall trade-off)")

    print("\n🎯 Threshold maximizing Recall (safety-focused):")
    print(f"    Threshold:  {best_recall_row['threshold']:.2f}")
    print(f"    Recall:     {best_recall_row['recall']:.4f}")
    print(f"    Precision:  {best_recall_row['precision']:.4f}")
    print("    (Minimizes missed incidents, increases false alarms)")

    # Default threshold comparison
    default_row = threshold_df[threshold_df['threshold'] == 0.50]
    if not default_row.empty:
        print("\n📌 Default threshold (0.50) for comparison:")
        print(f"    Precision: {default_row['precision'].values[0]:.4f}")
        print(f"    Recall:    {default_row['recall'].values[0]:.4f}")
        print(f"    F1:        {default_row['f1'].values[0]:.4f}")

    print("\nRECOMMENDATION:")
    print(
        f"  • Balanced performance: Use threshold = {
            best_f1_row['threshold']:.2f}")
    print(
        f"  • Safety priority:      Use threshold = {
            best_recall_row['threshold']:.2f}")

    return threshold_df


# ==============================================================================
# SECTION 6: ANALYSIS AND INTERPRETATION
# ==============================================================================


def print_metric_interpretation() -> None:
    """Print detailed metric interpretation for imbalanced datasets."""
    print_section("5. METRIC INTERPRETATION")

    interpretation = """
UNDERSTANDING METRICS IN IMBALANCED DATASETS:

⚠️  ACCURACY CAN BE MISLEADING:
  • In imbalanced datasets, predicting only the majority class can achieve high accuracy
  • Example: 90% no-incident flights → Always predict "no incident" = 90% accuracy
  • But this model misses ALL actual incidents (useless for safety!)
  • Accuracy treats all errors equally, ignoring class importance

✓ F1 SCORE IS MORE INFORMATIVE:
  • Harmonic mean of Precision and Recall
  • Balances trade-off between false positives and false negatives
  • More robust to class imbalance than accuracy
  • Low F1 indicates model struggles with minority class

🎯 RECALL (SENSITIVITY) IS CRITICAL FOR SAFETY:
  • Recall = TP / (TP + FN) - proportion of actual incidents detected
  • High Recall means few False Negatives (missed incidents)
  • In aviation safety, missing an incident (FN) is COSTLY:
    → Risk to aircraft, crew, and passengers
    → Potential catastrophic outcomes
    → Regulatory and legal consequences
  • Business priority: Maximize Recall to minimize missed incidents

🛡️  SPECIFICITY CONTROLS FALSE ALARMS:
  • Specificity = TN / (TN + FP) - proportion of non-incidents correctly identified
  • High Specificity means few False Positives (false alarms)
  • False alarms have operational costs:
    → Unnecessary maintenance checks
    → Flight delays and cancellations
    → Resource waste and lost revenue
  • Trade-off: High Recall often reduces Specificity

📊 PR-AUC FOR IMBALANCED DATA:
  • Precision-Recall AUC more informative than ROC-AUC for imbalanced data
  • ROC-AUC can be overly optimistic when negatives dominate
  • PR-AUC focuses on performance on minority (positive) class
  • Better reflects real-world performance in incident detection
"""
    print(interpretation)


# ==============================================================================
# SECTION 7: SAVE RESULTS
# ==============================================================================


def save_results(results_df: pd.DataFrame, stats: Dict[str, Any]) -> None:
    """
    Save evaluation results to CSV and Markdown.

    Args:
        results_df: DataFrame with metrics for all models
        stats: Statistics dictionary with problem type info
    """
    print_section("6. SAVING RESULTS")

    problem_type = stats['problem_type']
    best_model = results_df.index[0]
    sort_column = 'f1' if problem_type == "BINARY" else 'f1_macro'
    best_f1_value = results_df.loc[best_model, sort_column]

    # Mark best model
    results_df_display = results_df.copy()
    results_df_display.index = [f"{idx}*" if idx == best_model else idx
                                for idx in results_df_display.index]

    # Save CSV
    Config.OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df_display.to_csv(Config.OUTPUT_CSV, float_format='%.4f')
    print(f"✓ CSV saved: {Config.OUTPUT_CSV.name}")

    # Save Markdown
    with open(Config.OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Model Evaluation Results\n\n")
        f.write(f"**Problem Type:** {problem_type} Classification\n\n")
        f.write(f"**Best Model:** {best_model} (F1={best_f1_value:.4f})\n\n")
        f.write(
            "*Note: Model marked with * is the best performer based on F1 score.*\n\n")
        f.write("## Metrics Table\n\n")
        f.write(results_df_display.to_markdown(floatfmt='.4f'))
        f.write("\n\n")

        f.write("## Metric Definitions\n\n")
        if problem_type == "BINARY":
            f.write("- **Accuracy**: Overall correctness (TP+TN)/(TP+TN+FP+FN)\n")
            f.write("- **Precision**: Positive predictive value TP/(TP+FP)\n")
            f.write("- **Recall**: Sensitivity, true positive rate TP/(TP+FN)\n")
            f.write("- **F1**: Harmonic mean of precision and recall\n")
            f.write("- **Specificity**: True negative rate TN/(TN+FP)\n")
            f.write("- **ROC-AUC**: Area under ROC curve\n")
            f.write(
                "- **PR-AUC**: Area under Precision-Recall curve (better for imbalanced data)\n")
        else:
            f.write("- **Accuracy**: Overall correctness\n")
            f.write(
                "- **Precision/Recall/F1 (macro)**: Unweighted average across all classes\n")
            f.write(
                "- **Precision/Recall/F1 (micro)**: Aggregate contributions of all classes\n")
            f.write("- **ROC-AUC (macro)**: One-vs-rest macro-averaged ROC-AUC\n")

    print(f"✓ Markdown saved: {Config.OUTPUT_MD.name}")


def print_final_summary(results_df: pd.DataFrame,
                        stats: Dict[str, Any]) -> None:
    """
    Print final evaluation summary.

    Args:
        results_df: DataFrame with metrics
        stats: Statistics dictionary
    """
    print_section("EVALUATION COMPLETED SUCCESSFULLY!", "=")

    problem_type = stats['problem_type']
    best_model = results_df.index[0]
    sort_column = 'f1' if problem_type == "BINARY" else 'f1_macro'
    best_f1_value = results_df.loc[best_model, sort_column]

    print("\nSummary:")
    print(f"  ✓ Evaluated {len(results_df)} models")
    print(f"  ✓ Problem type: {problem_type} classification")
    print(f"  ✓ Best model: {best_model}")
    print(f"  ✓ Best F1: {best_f1_value:.4f}")

    print("\nOutput files:")
    print(f"  • {Config.OUTPUT_CSV.name}")
    print(f"  • {Config.OUTPUT_MD.name}")

    if problem_type == "BINARY":
        best_recall = results_df.loc[best_model, 'recall']
        best_specificity = results_df.loc[best_model, 'specificity']
        print("\nKey insights:")
        print(
            f"  • Best model achieves {
                best_recall:.1%} recall (incident detection rate)")
        print(
            f"  • Specificity of {
                best_specificity:.1%} (false alarm rate: {
                (
                    1 -
                    best_specificity):.1%})")
        print("  • Consider threshold tuning based on business priorities")

    print("\nNext steps:")
    print("  → Analyze confusion matrices for detailed error patterns")
    print("  → Consider ensemble methods or hyperparameter tuning")
    print("  → Deploy best model with appropriate threshold")


# ==============================================================================
# SECTION 8: MAIN EXECUTION
# ==============================================================================


def main() -> None:
    """Main execution function."""
    print_section("AIRCRAFT INCIDENT CLASSIFICATION - MODEL EVALUATION", "=")

    # Load data
    y_test, stats = load_ground_truth()
    predictions_dict = load_all_predictions()

    # Evaluate all models
    results_df = evaluate_all_models(predictions_dict, y_test, stats)

    # Print comparative results
    print_section("COMPARATIVE RESULTS")
    print("\nModels ranked by F1 score:\n")
    print(results_df.to_string(float_format='%.4f'))

    best_model = results_df.index[0]
    sort_column = 'f1' if stats['problem_type'] == "BINARY" else 'f1_macro'
    print(
        f"\n🏆 Best model: {best_model} (F1={results_df.loc[best_model, sort_column]:.4f})")

    # Threshold tuning for binary classification
    if stats['problem_type'] == "BINARY":
        tune_threshold(best_model, predictions_dict)

    # Print interpretations
    print_metric_interpretation()

    # Save results
    save_results(results_df, stats)

    # Final summary
    print_final_summary(results_df, stats)


if __name__ == '__main__':
    main()
