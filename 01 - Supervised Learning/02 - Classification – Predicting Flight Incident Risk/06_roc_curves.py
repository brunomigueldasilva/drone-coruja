#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
AIRCRAFT INCIDENT CLASSIFICATION - ROC AND PRECISION-RECALL CURVES
==============================================================================

Purpose: Generate and compare ROC and Precision-Recall curves for all models

This script:
1. Automatically detects binary vs multiclass classification
2. Loads predictions and probabilities from all trained models
3. For binary classification:
   - Plots comparative ROC curves with AUC scores
   - Plots comparative Precision-Recall curves with AP scores
   - Explains why PR curves are important for imbalanced data
4. For multiclass classification:
   - Binarizes labels for one-vs-rest approach
   - Plots macro-averaged ROC curves
   - Optionally plots per-class ROC curves
5. Saves high-quality visualizations (PNG/PDF)

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')


# Matplotlib Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Configuration Constants
class Config:
    """ROC and Precision-Recall curves analysis configuration parameters."""
    PROCESSED_DATA_DIR = Path('outputs/data_processed')
    PREDICTIONS_DIR = Path('outputs/predictions')
    GRAPHICS_DIR = Path('outputs/graphics')

    # Output files
    OUTPUT_ROC_PNG = GRAPHICS_DIR / 'roc_comparative.png'
    OUTPUT_ROC_PDF = GRAPHICS_DIR / 'roc_comparative.pdf'
    OUTPUT_PR_PNG = GRAPHICS_DIR / 'pr_comparative.png'
    OUTPUT_PR_PDF = GRAPHICS_DIR / 'pr_comparative.pdf'
    OUTPUT_ROC_MULTI_PNG = GRAPHICS_DIR / 'roc_multiclass.png'
    OUTPUT_ROC_MULTI_PDF = GRAPHICS_DIR / 'roc_multiclass.pdf'
    OUTPUT_ROC_PER_CLASS_PNG = GRAPHICS_DIR / 'roc_per_class.png'
    OUTPUT_ROC_PER_CLASS_PDF = GRAPHICS_DIR / 'roc_per_class.pdf'

    # Plotting configuration
    DPI = 250
    FIGSIZE_SINGLE = (10, 8)
    FIGSIZE_MULTI = (14, 10)


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


def load_ground_truth() -> Tuple[np.ndarray, str, int, np.ndarray]:
    """
    Load ground truth labels and detect problem type.

    Returns:
        Tuple containing:
            - y_test: Ground truth labels
            - problem_type: 'BINARY' or 'MULTICLASS'
            - n_classes: Number of unique classes
            - unique_classes: Array of unique class labels

    Raises:
        SystemExit: If ground truth file not found
    """
    print_section("1. LOADING GROUND TRUTH AND DETECTING PROBLEM TYPE", "-")

    y_test_path = Config.PROCESSED_DATA_DIR / 'y_test.csv'

    if not y_test_path.exists():
        print(f"✗ ERROR: Ground truth file not found: {y_test_path}")
        print("  Please run 02_preprocessing.py first.")
        exit(1)

    y_test = pd.read_csv(y_test_path).values.ravel()
    print(f"✓ Loaded ground truth: {len(y_test)} samples")

    # Detect problem type
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)

    print(f"\nNumber of unique classes: {n_classes}")
    print(f"Class labels: {unique_classes}")

    if n_classes == 2:
        problem_type = 'BINARY'
        print("\n✓ Problem type: BINARY CLASSIFICATION")
        print("  → Will generate ROC and Precision-Recall curves")
        print("  → All models will be compared on the same plot")
    elif n_classes > 2:
        problem_type = 'MULTICLASS'
        print(
            f"\n✓ Problem type: MULTICLASS CLASSIFICATION ({n_classes} classes)")
        print("  → Will generate one-vs-rest ROC curves")
        print("  → Macro and micro averaging will be used")
    else:
        print("\n✗ ERROR: Invalid number of classes")
        exit(1)

    return y_test, problem_type, n_classes, unique_classes


def load_predictions() -> Dict[str, Dict[str, Any]]:
    """
    Load predictions and probabilities from all trained models.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping model names to their prediction data

    Raises:
        SystemExit: If no prediction files found
    """
    print_section("2. LOADING PREDICTIONS FROM ALL MODELS", "-")

    prediction_files = sorted(Config.PREDICTIONS_DIR.glob('predictions_*.csv'))

    if len(prediction_files) == 0:
        print(
            f"✗ ERROR: No prediction files found in {
                Config.PREDICTIONS_DIR}")
        print("  Please run 03_train_models.py first.")
        exit(1)

    print(f"Found {len(prediction_files)} prediction files\n")

    models_data = {}

    for pred_file in prediction_files:
        model_name = pred_file.stem.replace('predictions_', '')

        pred_df = pd.read_csv(pred_file)

        proba_cols = [
            col for col in pred_df.columns if col.startswith('proba_class_')]

        if len(proba_cols) == 0:
            print(
                f"  ⚠️  {
                    model_name:<25} - No probability estimates (skipping)")
            continue

        models_data[model_name] = {
            'y_true': pred_df['y_true'].values,
            'y_pred': pred_df['y_pred'].values,
            'proba_cols': proba_cols,
            'pred_df': pred_df
        }

        print(f"  ✓ {model_name:<25} - {len(proba_cols)} probability columns")

    print(
        f"\n✓ Loaded predictions for {
            len(models_data)} models with probabilities")

    if len(models_data) == 0:
        print("\n✗ ERROR: No models with probability estimates found")
        print("  ROC and PR curves require probability scores")
        exit(1)

    return models_data


# ==============================================================================
# SECTION 4: BINARY CLASSIFICATION - ROC CURVES
# ==============================================================================


def plot_binary_roc_curves(
        models_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Generate ROC curves for binary classification.

    Args:
        models_data: Dictionary containing model predictions and probabilities

    Returns:
        Dict[str, Dict[str, Any]]: ROC data for each model (fpr, tpr, auc)
    """
    print_section("3. BINARY CLASSIFICATION - ROC CURVES", "-")

    print("\nROC (Receiver Operating Characteristic) Curve:")
    print("  • Plots True Positive Rate (Recall) vs False Positive Rate")
    print("  • Shows classifier performance across all classification thresholds")
    print("  • AUC (Area Under Curve) = overall measure of performance")
    print("    - AUC = 1.0: Perfect classifier")
    print("    - AUC = 0.5: Random guessing (diagonal line)")
    print("    - AUC < 0.5: Worse than random (inverted predictions)")
    print()

    fig_roc, ax_roc = plt.subplots(figsize=Config.FIGSIZE_SINGLE)

    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=2,
                label='Random Classifier (AUC = 0.50)', alpha=0.7)

    roc_data = {}
    colors = sns.color_palette("husl", len(models_data))

    for idx, (model_name, data) in enumerate(models_data.items()):
        y_true = data['y_true']
        pred_df = data['pred_df']

        y_proba = pred_df['proba_class_1'].values

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        roc_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }

        ax_roc.plot(
            fpr,
            tpr,
            color=colors[idx],
            linewidth=2.5,
            label=f'{model_name} (AUC = {
                roc_auc:.4f})')

        print(f"  ✓ {model_name:<25} ROC-AUC = {roc_auc:.4f}")

    print()

    ax_roc.set_xlabel(
        'False Positive Rate (FPR)',
        fontsize=12,
        fontweight='bold')
    ax_roc.set_ylabel(
        'True Positive Rate (TPR)',
        fontsize=12,
        fontweight='bold')
    ax_roc.set_title(
        'ROC Curves - Model Comparison\nBinary Classification',
        fontsize=14,
        fontweight='bold',
        pad=15)
    ax_roc.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])

    plt.tight_layout()

    fig_roc.savefig(Config.OUTPUT_ROC_PNG, dpi=Config.DPI, bbox_inches='tight')
    print(f"✓ Saved ROC curve: {Config.OUTPUT_ROC_PNG.name}")

    fig_roc.savefig(Config.OUTPUT_ROC_PDF, dpi=Config.DPI, bbox_inches='tight')
    print(f"✓ Saved ROC curve: {Config.OUTPUT_ROC_PDF.name}")

    plt.close(fig_roc)

    return roc_data


# ==============================================================================
# SECTION 5: BINARY CLASSIFICATION - PRECISION-RECALL CURVES
# ==============================================================================


def plot_binary_pr_curves(
        models_data: Dict[str, Dict[str, Any]], y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Generate Precision-Recall curves for binary classification.

    Args:
        models_data: Dictionary containing model predictions and probabilities
        y_test: Ground truth labels

    Returns:
        Dict[str, Dict[str, Any]]: PR data for each model (precision, recall, ap)
    """
    print_section("4. BINARY CLASSIFICATION - PRECISION-RECALL CURVES", "-")

    class_counts = np.bincount(y_test.astype(int))
    majority_class = class_counts[0]
    minority_class = class_counts[1]
    imbalance_ratio = majority_class / minority_class

    print("\nPrecision-Recall (PR) Curve:")
    print("  • Plots Precision vs Recall across all classification thresholds")
    print("  • MORE INFORMATIVE than ROC for IMBALANCED datasets")
    print("  • Why? ROC can be overly optimistic when negative class dominates")
    print("  • AP (Average Precision) = weighted mean of precisions at each threshold")
    print()
    print("Dataset imbalance:")
    print(f"  Class 0: {majority_class:,} samples")
    print(f"  Class 1: {minority_class:,} samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    print()

    if imbalance_ratio > 2.0:
        print("⚠️  IMBALANCED DATASET DETECTED!")
        print("  → Precision-Recall curve is MORE RELIABLE than ROC curve")
        print("  → Focus on AP (Average Precision) for model comparison")
        print()

    fig_pr, ax_pr = plt.subplots(figsize=Config.FIGSIZE_SINGLE)

    baseline_precision = minority_class / len(y_test)
    ax_pr.axhline(
        y=baseline_precision,
        color='k',
        linestyle='--',
        linewidth=2,
        label=f'Baseline (No Skill) = {
            baseline_precision:.4f}',
        alpha=0.7)

    pr_data = {}
    colors = sns.color_palette("husl", len(models_data))

    for idx, (model_name, data) in enumerate(models_data.items()):
        y_true = data['y_true']
        pred_df = data['pred_df']

        y_proba = pred_df['proba_class_1'].values

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)

        pr_data[model_name] = {
            'precision': precision,
            'recall': recall,
            'ap': ap
        }

        ax_pr.plot(
            recall,
            precision,
            color=colors[idx],
            linewidth=2.5,
            label=f'{model_name} (AP = {
                ap:.4f})')

        print(f"  ✓ {model_name:<25} Average Precision = {ap:.4f}")

    print()

    ax_pr.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax_pr.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax_pr.set_title(
        'Precision-Recall Curves - Model Comparison\nBinary Classification',
        fontsize=14,
        fontweight='bold',
        pad=15)
    ax_pr.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([0.0, 1.05])

    note_text = f'Dataset imbalance: {
        imbalance_ratio:.1f}:1 → PR curve more informative than ROC'
    ax_pr.text(
        0.5,
        0.05,
        note_text,
        transform=ax_pr.transAxes,
        fontsize=9,
        ha='center',
        style='italic',
        alpha=0.7)

    plt.tight_layout()

    fig_pr.savefig(Config.OUTPUT_PR_PNG, dpi=Config.DPI, bbox_inches='tight')
    print(f"✓ Saved Precision-Recall curve: {Config.OUTPUT_PR_PNG.name}")

    fig_pr.savefig(Config.OUTPUT_PR_PDF, dpi=Config.DPI, bbox_inches='tight')
    print(f"✓ Saved Precision-Recall curve: {Config.OUTPUT_PR_PDF.name}")

    plt.close(fig_pr)

    return pr_data, imbalance_ratio


# ==============================================================================
# SECTION 6: BINARY CLASSIFICATION - SUMMARY
# ==============================================================================


def print_binary_summary(roc_data: Dict[str,
                                        Dict[str,
                                             Any]],
                         pr_data: Dict[str,
                                       Dict[str,
                                            Any]],
                         imbalance_ratio: float) -> Tuple[str,
                                                          float,
                                                          float]:
    """
    Print summary for binary classification.

    Args:
        roc_data: ROC curve data for all models
        pr_data: Precision-Recall curve data for all models
        imbalance_ratio: Ratio of majority to minority class

    Returns:
        Tuple containing best model name and its ROC-AUC and AP scores
    """
    print_section("5. BINARY CLASSIFICATION - SUMMARY", "-")

    best_roc_model = max(roc_data.items(), key=lambda x: x[1]['auc'])
    best_pr_model = max(pr_data.items(), key=lambda x: x[1]['ap'])

    print("\nBEST PERFORMING MODELS:")
    print()
    print(f"🏆 By ROC-AUC: {best_roc_model[0]}")
    print(f"   ROC-AUC = {best_roc_model[1]['auc']:.4f}")
    print()
    print(f"🏆 By Average Precision: {best_pr_model[0]}")
    print(f"   AP = {best_pr_model[1]['ap']:.4f}")
    print()

    print("ALL MODELS COMPARISON:")
    print()
    print(f"{'Model':<25} {'ROC-AUC':<10} {'Avg Precision':<15} {'Status':<15}")
    print("-" * 80)

    for model_name in roc_data.keys():
        roc_auc = roc_data[model_name]['auc']
        ap = pr_data[model_name]['ap']

        if roc_auc >= 0.95:
            status = "Excellent"
        elif roc_auc >= 0.90:
            status = "Very Good"
        elif roc_auc >= 0.80:
            status = "Good"
        elif roc_auc >= 0.70:
            status = "Fair"
        else:
            status = "Poor"

        print(f"{model_name:<25} {roc_auc:<10.4f} {ap:<15.4f} {status:<15}")

    print()

    if imbalance_ratio > 2.0:
        print("RECOMMENDATION:")
        print(f"  → Dataset is imbalanced ({imbalance_ratio:.1f}:1)")
        print("  → Prioritize Average Precision (AP) over ROC-AUC")
        print(
            f"  → Best model: {
                best_pr_model[0]} (AP = {
                best_pr_model[1]['ap']:.4f})")
    else:
        print("RECOMMENDATION:")
        print("  → Dataset is relatively balanced")
        print("  → Both ROC-AUC and AP are reliable metrics")
        print(
            f"  → Best model: {best_roc_model[0]} (ROC-AUC = {best_roc_model[1]['auc']:.4f})")

    return best_pr_model[0], best_roc_model[1]['auc'], best_pr_model[1]['ap']


# ==============================================================================
# SECTION 7: MULTICLASS CLASSIFICATION - ROC CURVES
# ==============================================================================


def plot_multiclass_roc_curves(models_data: Dict[str,
                                                 Dict[str,
                                                      Any]],
                               y_test: np.ndarray,
                               n_classes: int,
                               unique_classes: np.ndarray) -> Dict[str,
                                                                   Dict[str,
                                                                        Any]]:
    """
    Generate macro-averaged ROC curves for multiclass classification.

    Args:
        models_data: Dictionary containing model predictions and probabilities
        y_test: Ground truth labels
        n_classes: Number of classes
        unique_classes: Array of unique class labels

    Returns:
        Dict[str, Dict[str, Any]]: Macro-averaged ROC data for each model
    """
    print_section(
        "3. MULTICLASS CLASSIFICATION - MACRO-AVERAGED ROC CURVES",
        "-")

    print("\nMulticlass ROC Curve Strategy:")
    print("  • Binarize labels using one-vs-rest approach")
    print("  • Calculate ROC curve for each class separately")
    print("  • Compute macro-average: unweighted mean across all classes")
    print("  • Macro-AUC treats all classes equally (useful for imbalanced datasets)")
    print()

    y_test_bin = label_binarize(y_test, classes=unique_classes)

    fig_roc_macro, ax_roc_macro = plt.subplots(figsize=Config.FIGSIZE_MULTI)

    ax_roc_macro.plot([0, 1], [0, 1], 'k--', linewidth=2,
                      label='Random Classifier', alpha=0.7)

    roc_macro_data = {}
    colors = sns.color_palette("husl", len(models_data))

    for idx, (model_name, data) in enumerate(models_data.items()):
        pred_df = data['pred_df']
        proba_cols = data['proba_cols']

        y_proba = pred_df[proba_cols].values

        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}

        for i, class_label in enumerate(unique_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc_dict[class_label] = auc(fpr, tpr)
            fpr_dict[class_label] = fpr
            tpr_dict[class_label] = tpr

        all_fpr = np.unique(np.concatenate(
            [fpr_dict[c] for c in unique_classes]))
        mean_tpr = np.zeros_like(all_fpr)

        for class_label in unique_classes:
            mean_tpr += np.interp(all_fpr,
                                  fpr_dict[class_label],
                                  tpr_dict[class_label])

        mean_tpr /= n_classes

        macro_auc = np.mean(list(roc_auc_dict.values()))

        roc_macro_data[model_name] = {
            'fpr': all_fpr,
            'tpr': mean_tpr,
            'auc': macro_auc,
            'per_class_auc': roc_auc_dict
        }

        ax_roc_macro.plot(all_fpr, mean_tpr, color=colors[idx], linewidth=2.5,
                          label=f'{model_name} (Macro AUC = {macro_auc:.4f})')

        print(f"  ✓ {model_name:<25} Macro ROC-AUC = {macro_auc:.4f}")
        for class_label, auc_val in roc_auc_dict.items():
            print(f"      Class {class_label}: AUC = {auc_val:.4f}")

    print()

    ax_roc_macro.set_xlabel(
        'False Positive Rate (FPR)',
        fontsize=12,
        fontweight='bold')
    ax_roc_macro.set_ylabel(
        'True Positive Rate (TPR)',
        fontsize=12,
        fontweight='bold')
    ax_roc_macro.set_title(
        f'ROC Curves (Macro-Averaged) - Model Comparison\n' f'Multiclass Classification ({n_classes} classes)',
        fontsize=14,
        fontweight='bold',
        pad=15)
    ax_roc_macro.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_roc_macro.grid(True, alpha=0.3)
    ax_roc_macro.set_xlim([-0.02, 1.02])
    ax_roc_macro.set_ylim([-0.02, 1.02])

    note_text = 'Macro-averaging: Unweighted mean across all classes'
    ax_roc_macro.text(
        0.5,
        0.05,
        note_text,
        transform=ax_roc_macro.transAxes,
        fontsize=9,
        ha='center',
        style='italic',
        alpha=0.7)

    plt.tight_layout()

    fig_roc_macro.savefig(
        Config.OUTPUT_ROC_MULTI_PNG,
        dpi=Config.DPI,
        bbox_inches='tight')
    print(
        f"✓ Saved macro-averaged ROC curve: {Config.OUTPUT_ROC_MULTI_PNG.name}")

    fig_roc_macro.savefig(
        Config.OUTPUT_ROC_MULTI_PDF,
        dpi=Config.DPI,
        bbox_inches='tight')
    print(
        f"✓ Saved macro-averaged ROC curve: {Config.OUTPUT_ROC_MULTI_PDF.name}")

    plt.close(fig_roc_macro)

    return roc_macro_data


# ==============================================================================
# SECTION 8: MULTICLASS CLASSIFICATION - PER-CLASS ROC CURVES
# ==============================================================================


def plot_per_class_roc_curves(models_data: Dict[str,
                                                Dict[str,
                                                     Any]],
                              roc_macro_data: Dict[str,
                                                   Dict[str,
                                                        Any]],
                              y_test: np.ndarray,
                              n_classes: int,
                              unique_classes: np.ndarray) -> None:
    """
    Generate per-class ROC curves for best model.

    Args:
        models_data: Dictionary containing model predictions and probabilities
        roc_macro_data: Macro-averaged ROC data
        y_test: Ground truth labels
        n_classes: Number of classes
        unique_classes: Array of unique class labels
    """
    if n_classes > 10 or len(roc_macro_data) == 0:
        return

    print_section("4. PER-CLASS ROC CURVES", "-")

    best_model_name = max(roc_macro_data.items(), key=lambda x: x[1]['auc'])[0]
    print(f"Plotting per-class ROC curves for best model: {best_model_name}")
    print()

    y_test_bin = label_binarize(y_test, classes=unique_classes)

    best_model_data = models_data[best_model_name]
    pred_df = best_model_data['pred_df']
    proba_cols = best_model_data['proba_cols']
    y_proba = pred_df[proba_cols].values

    fig_per_class, ax_per_class = plt.subplots(figsize=Config.FIGSIZE_MULTI)

    ax_per_class.plot([0, 1], [0, 1], 'k--', linewidth=2,
                      label='Random Classifier', alpha=0.7)

    class_colors = sns.color_palette("husl", n_classes)

    for i, class_label in enumerate(unique_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        ax_per_class.plot(fpr, tpr, color=class_colors[i], linewidth=2,
                          label=f'Class {class_label} (AUC = {roc_auc:.4f})')

        print(f"  • Class {class_label}: AUC = {roc_auc:.4f}")

    print()

    ax_per_class.set_xlabel(
        'False Positive Rate (FPR)',
        fontsize=12,
        fontweight='bold')
    ax_per_class.set_ylabel(
        'True Positive Rate (TPR)',
        fontsize=12,
        fontweight='bold')
    ax_per_class.set_title(
        f'Per-Class ROC Curves - {best_model_name}\nOne-vs-Rest Classification',
        fontsize=14,
        fontweight='bold',
        pad=15)
    ax_per_class.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_per_class.grid(True, alpha=0.3)
    ax_per_class.set_xlim([-0.02, 1.02])
    ax_per_class.set_ylim([-0.02, 1.02])

    plt.tight_layout()

    fig_per_class.savefig(
        Config.OUTPUT_ROC_PER_CLASS_PNG,
        dpi=Config.DPI,
        bbox_inches='tight')
    print(
        f"✓ Saved per-class ROC curve: {Config.OUTPUT_ROC_PER_CLASS_PNG.name}")

    fig_per_class.savefig(
        Config.OUTPUT_ROC_PER_CLASS_PDF,
        dpi=Config.DPI,
        bbox_inches='tight')
    print(
        f"✓ Saved per-class ROC curve: {Config.OUTPUT_ROC_PER_CLASS_PDF.name}")

    plt.close(fig_per_class)


# ==============================================================================
# SECTION 9: MULTICLASS CLASSIFICATION - SUMMARY
# ==============================================================================


def print_multiclass_summary(
        roc_macro_data: Dict[str, Dict[str, Any]]) -> Tuple[str, float]:
    """
    Print summary for multiclass classification.

    Args:
        roc_macro_data: Macro-averaged ROC data for all models

    Returns:
        Tuple containing best model name and its macro AUC score
    """
    print_section("5. MULTICLASS CLASSIFICATION - SUMMARY", "-")

    if len(roc_macro_data) == 0:
        return "", 0.0

    best_model = max(roc_macro_data.items(), key=lambda x: x[1]['auc'])

    print("\nBEST PERFORMING MODEL:")
    print()
    print(f"🏆 {best_model[0]}")
    print(f"   Macro-averaged ROC-AUC = {best_model[1]['auc']:.4f}")
    print()
    print("   Per-class AUC:")
    for class_label, auc_val in best_model[1]['per_class_auc'].items():
        print(f"     Class {class_label}: {auc_val:.4f}")
    print()

    print("ALL MODELS COMPARISON:")
    print()
    for model_name, data in sorted(
            roc_macro_data.items(), key=lambda x: x[1]['auc'], reverse=True):
        auc_val = data['auc']
        if auc_val >= 0.95:
            status = "Excellent"
        elif auc_val >= 0.90:
            status = "Very Good"
        elif auc_val >= 0.80:
            status = "Good"
        elif auc_val >= 0.70:
            status = "Fair"
        else:
            status = "Poor"
        print(f"  • {model_name:<25} Macro AUC = {auc_val:.4f} ({status})")

    return best_model[0], best_model[1]['auc']


# ==============================================================================
# SECTION 10: FINAL SUMMARY
# ==============================================================================


def print_final_summary(problem_type: str,
                        models_data: Dict[str,
                                          Dict[str,
                                               Any]],
                        n_classes: int,
                        best_model_info: Tuple) -> None:
    """
    Print final summary of ROC analysis.

    Args:
        problem_type: 'BINARY' or 'MULTICLASS'
        models_data: Dictionary containing model predictions
        n_classes: Number of classes
        best_model_info: Tuple with best model information
    """
    print_section("ROC AND PRECISION-RECALL ANALYSIS COMPLETED!")

    print("\nSummary:")
    print(f"  ✓ Problem type: {problem_type} classification")
    print(f"  ✓ Models analyzed: {len(models_data)}")
    print()

    print("Output files created:")
    if problem_type == 'BINARY':
        best_pr_model, best_roc_auc, best_ap, imbalance_ratio = best_model_info

        print(f"  • {Config.OUTPUT_ROC_PNG.name} (ROC curves comparison)")
        print(f"  • {Config.OUTPUT_ROC_PDF.name} (ROC curves comparison)")
        print(f"  • {Config.OUTPUT_PR_PNG.name} (Precision-Recall curves)")
        print(f"  • {Config.OUTPUT_PR_PDF.name} (Precision-Recall curves)")
        print()
        print("Key takeaway:")
        print("  → For imbalanced data, Precision-Recall curves are MORE INFORMATIVE")
        print(f"  → Imbalance ratio: {imbalance_ratio:.1f}:1")
        if best_pr_model:
            print(
                f"  → Best model by PR-AUC: {best_pr_model} (AP = {best_ap:.4f})")
    else:
        best_model_name, best_auc = best_model_info

        print(f"  • {Config.OUTPUT_ROC_MULTI_PNG.name} (Macro-averaged ROC)")
        print(f"  • {Config.OUTPUT_ROC_MULTI_PDF.name} (Macro-averaged ROC)")
        if n_classes <= 10:
            print(
                f"  • {
                    Config.OUTPUT_ROC_PER_CLASS_PNG.name} (Per-class ROC)")
            print(
                f"  • {
                    Config.OUTPUT_ROC_PER_CLASS_PDF.name} (Per-class ROC)")
        print()
        print("Key takeaway:")
        print("  → Macro-averaging treats all classes equally")
        if best_model_name:
            print(
                f"  → Best model: {best_model_name} (Macro AUC = {
                    best_auc:.4f})")

    print()
    print("Next steps:")
    print("  → Review ROC/PR curves to understand model discrimination ability")
    print("  → Compare with confusion matrix and other metrics")
    print("  → Select final model based on comprehensive evaluation")
    print()


# ==============================================================================
# SECTION 11: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main function that orchestrates ROC and Precision-Recall curve analysis.

    This function executes the complete analysis workflow:
    1. Load ground truth and detect problem type
    2. Load predictions from all models
    3. Generate ROC curves (binary or multiclass)
    4. Generate Precision-Recall curves (binary only)
    5. Print comprehensive summary
    """
    print("=" * 80)
    print("ROC AND PRECISION-RECALL CURVES ANALYSIS")
    print("=" * 80)

    # Create graphics directory if needed
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load ground truth and detect problem type
    y_test, problem_type, n_classes, unique_classes = load_ground_truth()

    # 2. Load predictions from all models
    models_data = load_predictions()

    # 3. Generate curves based on problem type
    if problem_type == 'BINARY':
        # Binary classification
        roc_data = plot_binary_roc_curves(models_data)
        pr_data, imbalance_ratio = plot_binary_pr_curves(models_data, y_test)
        best_model_info = print_binary_summary(
            roc_data, pr_data, imbalance_ratio)
        best_model_info = (*best_model_info, imbalance_ratio)

    else:
        # Multiclass classification
        roc_macro_data = plot_multiclass_roc_curves(
            models_data, y_test, n_classes, unique_classes)
        plot_per_class_roc_curves(
            models_data,
            roc_macro_data,
            y_test,
            n_classes,
            unique_classes)
        best_model_info = print_multiclass_summary(roc_macro_data)

    # 4. Print final summary
    print_final_summary(problem_type, models_data, n_classes, best_model_info)

    print("=" * 80)
    print("ANALYSIS COMPLETED!")
    print("=" * 80)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
