#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
AIRCRAFT INCIDENT CLASSIFICATION - CONFUSION MATRIX ANALYSIS
==============================================================================

Purpose: Visualize and interpret confusion matrix for the best performing model

This script:
1. Automatically identifies the best model based on F1 score
2. Loads predictions for the best model
3. Generates confusion matrix visualization with percentages
4. Calculates detailed error metrics (FPR, FNR, Specificity, Recall)
5. Provides contextual interpretation for aviation safety
6. Saves plots (PNG/PDF) and summary report (Markdown)

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')


class Config:
    """Confusion matrix analysis configuration parameters."""
    # Directories
    TABLES_DIR = Path('outputs/results')
    GRAPHICS_DIR = Path('outputs/graphics')
    PREDICTIONS_DIR = Path('outputs/predictions')

    # Input files
    METRICS_FILE = TABLES_DIR / 'results_metrics.csv'

    # Output files
    OUTPUT_PNG = GRAPHICS_DIR / 'confusion_matrix.png'
    OUTPUT_PDF = GRAPHICS_DIR / 'confusion_matrix.pdf'
    OUTPUT_SUMMARY_MD = TABLES_DIR / 'confusion_matrix_summary.md'

    # Visualization settings
    DPI = 300
    FIGSIZE_BINARY = (10, 8)
    FIGSIZE_MULTICLASS = (12, 10)


# Visualization configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "-") -> None:
    """Print formatted section header."""
    print("\n" + char * 80)
    print(title)
    print(char * 80)
    print()


def save_plot(fig: plt.Figure, filename_base: str) -> None:
    """
    Save plot in PNG and PDF formats.

    Args:
        fig: Matplotlib figure to save
        filename_base: Base filename (e.g., 'confusion_matrix')
    """
    png_path = Config.GRAPHICS_DIR / f'{filename_base}.png'
    pdf_path = Config.GRAPHICS_DIR / f'{filename_base}.pdf'

    fig.savefig(png_path, dpi=Config.DPI, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    print(f"✓ Saved: {png_path.name}")
    print(f"✓ Saved: {pdf_path.name}")


# ==============================================================================
# SECTION 3: MODEL SELECTION FUNCTIONS
# ==============================================================================


def load_metrics() -> pd.DataFrame:
    """
    Load model metrics from CSV file.

    Returns:
        DataFrame with model metrics

    Raises:
        SystemExit: If metrics file not found
    """
    if not Config.METRICS_FILE.exists():
        print(f"✗ ERROR: Metrics file not found: {Config.METRICS_FILE}")
        print("  Please run 04_evaluate_metrics.py first.")
        exit(1)

    metrics_df = pd.read_csv(Config.METRICS_FILE, index_col=0)
    # Remove asterisk from model names if present
    metrics_df.index = metrics_df.index.str.replace('*', '', regex=False)

    print(f"✓ Loaded metrics for {len(metrics_df)} models")
    return metrics_df


def detect_problem_type(metrics_df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Detect if problem is binary or multiclass classification.

    Args:
        metrics_df: DataFrame with model metrics

    Returns:
        Tuple of (problem_type, f1_column, roc_column)

    Raises:
        SystemExit: If problem type cannot be determined
    """
    if 'f1' in metrics_df.columns:
        problem_type = 'BINARY'
        f1_column = 'f1'
        roc_column = 'roc_auc'
        print("Problem type: BINARY CLASSIFICATION")
    elif 'f1_macro' in metrics_df.columns:
        problem_type = 'MULTICLASS'
        f1_column = 'f1_macro'
        roc_column = 'roc_auc_macro'
        print("Problem type: MULTICLASS CLASSIFICATION")
    else:
        print("✗ ERROR: Cannot determine problem type from metrics")
        exit(1)

    print(
        f"Selection criteria: {f1_column} (primary), {roc_column} (tiebreaker)")
    return problem_type, f1_column, roc_column


def identify_best_model(metrics_df: pd.DataFrame, f1_column: str,
                        roc_column: str) -> str:
    """
    Identify best model based on F1 score (and ROC-AUC as tiebreaker).

    Args:
        metrics_df: DataFrame with model metrics
        f1_column: Name of F1 score column
        roc_column: Name of ROC-AUC column

    Returns:
        Name of best model
    """
    # Find best model based on F1 score
    best_f1 = metrics_df[f1_column].max()
    best_models = metrics_df[metrics_df[f1_column] == best_f1]

    if len(best_models) > 1:
        # Tiebreaker: highest ROC-AUC
        print(f"⚠️  Multiple models with same F1 score ({best_f1:.4f})")
        print("   Using ROC-AUC as tiebreaker...")
        best_model_name = best_models[roc_column].idxmax()
    else:
        best_model_name = best_models.index[0]

    print(f"🏆 Best model identified: {best_model_name}")
    print(f"   F1 Score: {metrics_df.loc[best_model_name, f1_column]:.4f}")
    if not pd.isna(metrics_df.loc[best_model_name, roc_column]):
        print(
            f"   ROC-AUC:  {metrics_df.loc[best_model_name, roc_column]:.4f}")

    return best_model_name


# ==============================================================================
# SECTION 4: DATA LOADING FUNCTIONS
# ==============================================================================


def load_predictions(
        model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load predictions for specified model.

    Args:
        model_name: Name of the model

    Returns:
        Tuple of (y_true, y_pred, unique_classes)

    Raises:
        SystemExit: If prediction file not found
    """
    prediction_file = Config.PREDICTIONS_DIR / f'predictions_{model_name}.csv'

    if not prediction_file.exists():
        print(f"✗ ERROR: Prediction file not found: {prediction_file}")
        exit(1)

    pred_df = pd.read_csv(prediction_file)
    print(f"✓ Loaded predictions from: {prediction_file.name}")
    print(f"  Columns: {list(pred_df.columns)}")
    print(f"  Samples: {len(pred_df)}")

    y_true = pred_df['y_true'].values
    y_pred = pred_df['y_pred'].values
    unique_classes = np.unique(y_true)

    print(f"Number of classes: {len(unique_classes)}")
    print(f"Class labels: {unique_classes}")

    return y_true, y_pred, unique_classes


# ==============================================================================
# SECTION 5: CONFUSION MATRIX COMPUTATION
# ==============================================================================


def compute_confusion_matrix(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             unique_classes: np.ndarray) -> Tuple[np.ndarray,
                                                                  np.ndarray]:
    """
    Compute confusion matrix with counts and percentages.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        unique_classes: Array of unique class labels

    Returns:
        Tuple of (confusion_matrix_counts, confusion_matrix_percentages)
    """
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    print("Confusion Matrix (raw counts):")
    print(cm)

    # Calculate percentages (normalize by row = by true class)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print("\nConfusion Matrix (percentages by row):")
    print(cm_percent)

    return cm, cm_percent


# ==============================================================================
# SECTION 6: VISUALIZATION FUNCTIONS
# ==============================================================================


def plot_binary_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    """
    Create confusion matrix plot for binary classification.

    Args:
        cm: Confusion matrix (2x2)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=Config.FIGSIZE_BINARY)

    # Extract values
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    # Create labels with counts and percentages
    labels = [
        f'TN\n{tn}\n({tn / total * 100:.1f}%)',
        f'FP\n{fp}\n({fp / total * 100:.1f}%)',
        f'FN\n{fn}\n({fn / total * 100:.1f}%)',
        f'TP\n{tp}\n({tp / total * 100:.1f}%)'
    ]

    # Create annotation matrix
    annotations = np.array(labels).reshape(2, 2)

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=annotations,
        fmt='',
        cmap='Blues',
        cbar=True,
        square=True,
        linewidths=2,
        linecolor='black',
        ax=ax,
        vmin=0,
        vmax=cm.max(),
        cbar_kws={'label': 'Number of Samples'}
    )

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix - Binary Classification',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticklabels(['No Incident (0)', 'Incident (1)'], fontsize=11)
    ax.set_yticklabels(['No Incident (0)', 'Incident (1)'],
                       fontsize=11, rotation=0)

    plt.tight_layout()
    return fig


def plot_multiclass_confusion_matrix(cm: np.ndarray,
                                     unique_classes: np.ndarray) -> plt.Figure:
    """
    Create confusion matrix plot for multiclass classification.

    Args:
        cm: Confusion matrix (NxN)
        unique_classes: Array of unique class labels

    Returns:
        Matplotlib figure
    """
    n_classes = len(unique_classes)
    fig, ax = plt.subplots(figsize=Config.FIGSIZE_MULTICLASS)

    # Create annotations with counts and percentages
    annotations = np.empty_like(cm, dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            total_row = cm[i, :].sum()
            pct = (count / total_row * 100) if total_row > 0 else 0
            annotations[i, j] = f'{count}\n({pct:.1f}%)'

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=annotations,
        fmt='',
        cmap='Blues',
        cbar=True,
        square=True,
        linewidths=1,
        linecolor='gray',
        ax=ax,
        vmin=0,
        vmax=cm.max(),
        cbar_kws={'label': 'Number of Samples'}
    )

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix - Multiclass Classification',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticklabels([f'Class {c}' for c in unique_classes], fontsize=10)
    ax.set_yticklabels(
        [f'Class {c}' for c in unique_classes], fontsize=10, rotation=0)

    plt.tight_layout()
    return fig


# ==============================================================================
# SECTION 7: METRICS ANALYSIS FUNCTIONS
# ==============================================================================


def calculate_binary_metrics(cm: np.ndarray) -> dict:
    """
    Calculate detailed metrics for binary classification.

    Args:
        cm: Confusion matrix (2x2)

    Returns:
        Dictionary with calculated metrics
    """
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'accuracy': (tp + tn) / (tp + tn + fp + fn)
    }

    return metrics


def print_binary_metrics(metrics: dict) -> None:
    """
    Print detailed binary classification metrics.

    Args:
        metrics: Dictionary with calculated metrics
    """
    print_section("5. DETAILED ERROR METRICS (BINARY CLASSIFICATION)")

    print("Error Metrics:")
    print(
        f"  • False Positive Rate (FPR): {
            metrics['fpr']:.4f} ({
            metrics['fpr'] *
            100:.2f}%)")
    print(f"    Formula: FP / (FP + TN) = {metrics['fp']} / "
          f"({metrics['fp']} + {metrics['tn']}) = {metrics['fpr']:.4f}")
    print()
    print(
        f"  • False Negative Rate (FNR): {
            metrics['fnr']:.4f} ({
            metrics['fnr'] *
            100:.2f}%)")
    print(f"    Formula: FN / (FN + TP) = {metrics['fn']} / "
          f"({metrics['fn']} + {metrics['tp']}) = {metrics['fnr']:.4f}")
    print()

    print("Performance Metrics:")
    print(f"  • Specificity (TNR): {metrics['specificity']:.4f} "
          f"({metrics['specificity'] * 100:.2f}%)")
    print(f"    Formula: TN / (TN + FP) = {metrics['tn']} / "
          f"({metrics['tn']} + {metrics['fp']}) = {metrics['specificity']:.4f}")
    print()
    print(f"  • Recall/Sensitivity (TPR): {metrics['recall']:.4f} "
          f"({metrics['recall'] * 100:.2f}%)")
    print(f"    Formula: TP / (TP + FN) = {metrics['tp']} / "
          f"({metrics['tp']} + {metrics['fn']}) = {metrics['recall']:.4f}")
    print()
    print(f"  • Precision (PPV): {metrics['precision']:.4f} "
          f"({metrics['precision'] * 100:.2f}%)")
    print(f"    Formula: TP / (TP + FP) = {metrics['tp']} / "
          f"({metrics['tp']} + {metrics['fp']}) = {metrics['precision']:.4f}")
    print()
    print(
        f"  • Accuracy: {
            metrics['accuracy']:.4f} ({
            metrics['accuracy'] *
            100:.2f}%)")
    print()


def print_aviation_interpretation(metrics: dict) -> None:
    """
    Print aviation safety interpretation of metrics.

    Args:
        metrics: Dictionary with calculated metrics
    """
    print_section("6. AVIATION SAFETY INTERPRETATION")

    fn = metrics['fn']
    fp = metrics['fp']
    fnr = metrics['fnr']
    fpr = metrics['fpr']

    print("🚨 FALSE NEGATIVES (FN) - CRITICAL SAFETY CONCERN:")
    print()
    print(f"   Count: {fn} incidents MISSED by the model")
    print()
    print("   Business Impact:")
    print("   • Undetected incidents pose DIRECT SAFETY RISKS")
    print("   • Potential for aircraft damage, crew/passenger injury")
    print("   • Regulatory violations and legal liability")
    print("   • Catastrophic outcomes if critical incidents are missed")
    print()
    print(f"   Current FNR: {fnr * 100:.2f}%")

    if fnr == 0:
        print("   ✓ EXCELLENT: Zero missed incidents!")
    elif fnr < 0.05:
        print("   ✓ VERY GOOD: Minimal safety risk (<5%)")
    elif fnr < 0.10:
        print("   ⚠️  ACCEPTABLE: Some risk, but manageable (5-10%)")
    else:
        print("   ⚠️  CONCERNING: High safety risk (>10%)")
    print()

    print("⚠️  FALSE POSITIVES (FP) - OPERATIONAL COST CONCERN:")
    print()
    print(f"   Count: {fp} false alarms")
    print()
    print("   Business Impact:")
    print("   • Triggers unnecessary maintenance inspections")
    print("   • Causes flight delays and potential cancellations")
    print("   • Wastes resources (personnel, time, money)")
    print("   • May lead to 'alarm fatigue' if too frequent")
    print()
    print(f"   Current FPR: {fpr * 100:.2f}%")

    if fpr < 0.05:
        print("   ✓ EXCELLENT: Minimal false alarms (<5%)")
    elif fpr < 0.10:
        print("   ✓ GOOD: Low operational impact (<10%)")
    elif fpr < 0.20:
        print("   ⚠️  MODERATE: Noticeable operational impact (10-20%)")
    else:
        print("   ⚠️  HIGH: Significant operational burden (>20%)")
    print()

    print("⚖️  TRADE-OFF ANALYSIS:")
    print()
    print("   In aviation safety, the priority hierarchy is:")
    print("   1. SAFETY FIRST: Minimize FN (don't miss incidents)")
    print("   2. EFFICIENCY SECOND: Minimize FP (reduce false alarms)")
    print()
    print("   Current model balance:")

    if fnr == 0 and fpr < 0.05:
        print("   ✓ OPTIMAL: Perfect safety with minimal false alarms")
    elif fnr < 0.05 and fpr < 0.10:
        print("   ✓ EXCELLENT: Strong safety with acceptable false alarms")
    elif fnr < 0.10 and fpr < 0.20:
        print("   ✓ GOOD: Good safety-efficiency balance")
    elif fnr < 0.10:
        print("   ⚠️  ACCEPTABLE: Good safety, but high false alarm rate")
    elif fpr < 0.10:
        print("   ⚠️  CONCERNING: Good efficiency, but missing too many incidents")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT: Both metrics could be better")
    print()

    print("💡 RECOMMENDATIONS:")
    if fnr > 0.05:
        print(f"   • Priority: Reduce FNR from {fnr * 100:.2f}% to <5%")
        print("   • Consider lowering classification threshold")
        print("   • Investigate characteristics of missed incidents")
        print("   • May need to collect more training data for minority class")
    if fpr > 0.10:
        print(f"   • Consider: Reduce FPR from {fpr * 100:.2f}% to <10%")
        print("   • Fine-tune model hyperparameters")
        print("   • Review threshold tuning results")
        print("   • Balance with safety requirements (FN reduction is priority)")
    if fnr <= 0.05 and fpr <= 0.10:
        print("   ✓ Model performance is excellent!")
        print("   • Monitor performance on new data")
        print("   • Consider deployment with current configuration")
        print("   • Implement real-time monitoring for drift detection")
    print()


def print_multiclass_analysis(
        cm: np.ndarray,
        unique_classes: np.ndarray) -> None:
    """
    Print per-class accuracy analysis for multiclass problems.

    Args:
        cm: Confusion matrix
        unique_classes: Array of unique class labels
    """
    print_section("5. PER-CLASS ACCURACY (MULTICLASS)")

    print("Classification accuracy by class:")
    for i, class_label in enumerate(unique_classes):
        correct = cm[i, i]
        total = cm[i, :].sum()
        accuracy_class = correct / total if total > 0 else 0
        print(
            f"  Class {class_label}: {correct}/{total} correct ({accuracy_class * 100:.2f}%)")

    print()
    print("Most common misclassifications:")

    # Find top 3 off-diagonal elements
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)

    flat_indices = np.argsort(cm_copy.ravel())[-3:][::-1]
    top_errors = np.unravel_index(flat_indices, cm_copy.shape)

    for true_idx, pred_idx in zip(top_errors[0], top_errors[1]):
        count = cm[true_idx, pred_idx]
        if count > 0:
            print(f"  • Class {unique_classes[true_idx]} misclassified as "
                  f"Class {unique_classes[pred_idx]}: {count} times")
    print()


# ==============================================================================
# SECTION 8: EXPORT FUNCTIONS
# ==============================================================================


def save_markdown_summary(
        model_name: str,
        problem_type: str,
        metrics_df: pd.DataFrame,
        f1_column: str,
        cm: np.ndarray,
        unique_classes: np.ndarray,
        metrics: Optional[dict] = None) -> None:
    """
    Save summary report to Markdown file.

    Args:
        model_name: Name of best model
        problem_type: 'BINARY' or 'MULTICLASS'
        metrics_df: DataFrame with all model metrics
        f1_column: Name of F1 column
        cm: Confusion matrix
        unique_classes: Array of unique class labels
        metrics: Dictionary with binary metrics (optional)
    """
    with open(Config.OUTPUT_SUMMARY_MD, 'w', encoding='utf-8') as f:
        f.write("# Confusion Matrix Analysis\n\n")
        f.write(f"**Model:** {model_name}\n\n")
        f.write(f"**Problem Type:** {problem_type} Classification\n\n")
        f.write(
            f"**F1 Score:** {metrics_df.loc[model_name, f1_column]:.4f}\n\n")

        f.write("## Confusion Matrix\n\n")
        f.write("### Raw Counts\n\n")
        f.write("```\n")
        f.write(str(cm))
        f.write("\n```\n\n")

        if problem_type == 'BINARY' and metrics:
            tn, fp, fn, tp = cm.ravel()
            fpr = metrics['fpr']
            fnr = metrics['fnr']
            specificity = metrics['specificity']
            recall = metrics['recall']
            precision = metrics['precision']
            accuracy = metrics['accuracy']

            f.write("### Matrix Breakdown\n\n")
            f.write(f"- **True Negatives (TN):** {tn} - "
                    "Correctly identified no-incident cases\n")
            f.write(f"- **False Positives (FP):** {fp} - "
                    "False alarms (predicted incident, was none)\n")
            f.write(f"- **False Negatives (FN):** {fn} - "
                    "Missed incidents (CRITICAL) ⚠️\n")
            f.write(f"- **True Positives (TP):** {tp} - "
                    "Correctly detected incidents\n\n")

            f.write("## Error Metrics\n\n")
            f.write("| Metric | Formula | Value | Percentage |\n")
            f.write("|--------|---------|-------|------------|\n")
            f.write(f"| False Positive Rate (FPR) | FP / (FP + TN) | "
                    f"{fpr:.4f} | {fpr * 100:.2f}% |\n")
            f.write(f"| False Negative Rate (FNR) | FN / (FN + TP) | "
                    f"{fnr:.4f} | {fnr * 100:.2f}% |\n")
            f.write(f"| Specificity (TNR) | TN / (TN + FP) | "
                    f"{specificity:.4f} | {specificity * 100:.2f}% |\n")
            f.write(f"| Recall (Sensitivity) | TP / (TP + FN) | "
                    f"{recall:.4f} | {recall * 100:.2f}% |\n")
            f.write(f"| Precision (PPV) | TP / (TP + FP) | "
                    f"{precision:.4f} | {precision * 100:.2f}% |\n")
            f.write(f"| Accuracy | (TP + TN) / Total | "
                    f"{accuracy:.4f} | {accuracy * 100:.2f}% |\n\n")

            f.write("## Interpretation for Aviation Safety\n\n")
            f.write("### False Negatives (FN) - Critical Safety Concern\n\n")
            f.write(f"**Count:** {fn} incidents missed by the model\n\n")
            f.write("**Business Impact:**\n")
            f.write("- Undetected incidents pose DIRECT SAFETY RISKS\n")
            f.write("- Potential for aircraft damage, crew/passenger injury\n")
            f.write("- Regulatory violations and legal liability\n")
            f.write("- Catastrophic outcomes if critical incidents are missed\n\n")
            f.write(f"**Status:** FNR = {fnr * 100:.2f}%\n\n")

            f.write("### False Positives (FP) - Operational Cost Concern\n\n")
            f.write(f"**Count:** {fp} false alarms\n\n")
            f.write("**Business Impact:**\n")
            f.write("- Triggers unnecessary maintenance inspections\n")
            f.write("- Causes flight delays and potential cancellations\n")
            f.write("- Wastes resources (personnel, time, money)\n")
            f.write("- May lead to 'alarm fatigue' if too frequent\n\n")
            f.write(f"**Status:** FPR = {fpr * 100:.2f}%\n\n")

            f.write("### Recommendations\n\n")
            if fnr > 0.05:
                f.write(
                    f"- **Priority:** Reduce FNR from {fnr * 100:.2f}% to <5%\n")
                f.write("- Consider lowering classification threshold\n")
                f.write("- Investigate characteristics of missed incidents\n\n")
            if fpr > 0.10:
                f.write(
                    f"- **Consider:** Reduce FPR from {fpr * 100:.2f}% to <10%\n")
                f.write("- Fine-tune model hyperparameters\n")
                f.write("- Balance with safety requirements\n\n")
            if fnr <= 0.05 and fpr <= 0.10:
                f.write("- ✓ Model performance is excellent!\n")
                f.write("- Ready for deployment with current configuration\n")
                f.write("- Implement real-time monitoring for drift detection\n\n")

        else:
            f.write("### Per-Class Accuracy\n\n")
            for i, class_label in enumerate(unique_classes):
                correct = cm[i, i]
                total = cm[i, :].sum()
                accuracy_class = correct / total if total > 0 else 0
                f.write(f"- **Class {class_label}:** {correct}/{total} "
                        f"({accuracy_class * 100:.2f}%)\n")
            f.write("\n")

        f.write("## Visualization\n\n")
        f.write("Confusion matrix plots saved to:\n")
        f.write(f"- `{Config.OUTPUT_PNG}`\n")
        f.write(f"- `{Config.OUTPUT_PDF}`\n\n")

        f.write("---\n\n")
        f.write("*Generated automatically by 05_confusion_matrix.py*\n")

    print(f"✓ Summary report saved: {Config.OUTPUT_SUMMARY_MD.name}")


def print_final_summary(model_name: str, problem_type: str,
                        metrics: Optional[dict] = None) -> None:
    """
    Print final summary of analysis.

    Args:
        model_name: Name of best model
        problem_type: 'BINARY' or 'MULTICLASS'
        metrics: Dictionary with binary metrics (optional)
    """
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX ANALYSIS COMPLETED!")
    print("=" * 80)
    print()

    print("Summary:")
    print(f"  ✓ Best model: {model_name}")
    print(f"  ✓ Problem type: {problem_type} classification")
    print("  ✓ Confusion matrix generated and analyzed")
    print()

    print("Output files created:")
    print(f"  • {Config.OUTPUT_PNG.name} (high-resolution PNG)")
    print(f"  • {Config.OUTPUT_PDF.name} (publication-quality PDF)")
    print(f"  • {Config.OUTPUT_SUMMARY_MD.name} (detailed summary report)")
    print()

    if problem_type == 'BINARY' and metrics:
        fn = metrics['fn']
        fp = metrics['fp']
        fnr = metrics['fnr']
        fpr = metrics['fpr']
        recall = metrics['recall']
        specificity = metrics['specificity']

        print("Key findings:")
        print(
            f"  • False Negative Rate: {
                fnr *
                100:.2f}% ({fn} incidents missed)")
        print(f"  • False Positive Rate: {fpr * 100:.2f}% ({fp} false alarms)")
        print(f"  • Recall (Sensitivity): {recall * 100:.2f}%")
        print(f"  • Specificity: {specificity * 100:.2f}%")
        print()

    print("Next steps:")
    print("  → Review confusion matrix visualization")
    print("  → Read detailed analysis in confusion_matrix_summary.md")
    print("  → Consider threshold adjustments if needed")
    print("  → Proceed with model deployment if metrics are acceptable")
    print()


# ==============================================================================
# SECTION 9: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the confusion matrix analysis."""
    # Create output directories
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONFUSION MATRIX ANALYSIS - BEST MODEL")
    print("=" * 80)
    print()

    try:
        # 1. Identify best model
        print_section("1. IDENTIFYING BEST MODEL")
        metrics_df = load_metrics()
        problem_type, f1_column, roc_column = detect_problem_type(metrics_df)
        best_model_name = identify_best_model(
            metrics_df, f1_column, roc_column)

        # 2. Load predictions
        print_section("2. LOADING PREDICTIONS FOR BEST MODEL")
        y_true, y_pred, unique_classes = load_predictions(best_model_name)

        # 3. Compute confusion matrix
        print_section("3. COMPUTING CONFUSION MATRIX")
        cm, cm_percent = compute_confusion_matrix(
            y_true, y_pred, unique_classes)

        # 4. Generate visualization
        print_section("4. GENERATING CONFUSION MATRIX VISUALIZATION")
        if problem_type == 'BINARY':
            fig = plot_binary_confusion_matrix(cm)
        else:
            fig = plot_multiclass_confusion_matrix(cm, unique_classes)

        save_plot(fig, 'confusion_matrix')
        plt.close(fig)

        # 5. Analyze metrics
        if problem_type == 'BINARY':
            metrics = calculate_binary_metrics(cm)
            print_binary_metrics(metrics)
            print_aviation_interpretation(metrics)
        else:
            metrics = None
            print_multiclass_analysis(cm, unique_classes)

        # 6. Save summary
        print_section("7. SAVING SUMMARY REPORT")
        save_markdown_summary(best_model_name, problem_type, metrics_df,
                              f1_column, cm, unique_classes, metrics)

        # 7. Print final summary
        print_final_summary(best_model_name, problem_type, metrics)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
