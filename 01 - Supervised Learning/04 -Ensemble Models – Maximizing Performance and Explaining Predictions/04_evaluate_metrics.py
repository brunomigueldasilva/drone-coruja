#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FLIGHT INCIDENT PREDICTION - MODEL EVALUATION (IMPROVED VERSION)
================================================================================

Purpose: Comprehensive evaluation of ensemble models with overfitting detection

This script evaluates 4 classification models focusing on metrics appropriate
for imbalanced datasets where the minority class (incidents) is critical.

NEW FEATURES IN THIS VERSION:
- Overfitting detection (train vs test comparison)
- Warning system for suspiciously perfect scores
- Improved model recommendations considering generalization
- Better handling of imbalanced data metrics

Key Learning Objectives:
- Understand why accuracy is misleading for imbalanced data
- Learn appropriate metrics: F1-Score, ROC-AUC, Precision, Recall
- Detect overfitting through train/test comparison
- Interpret confusion matrices
- Compare model performance objectively
- Make data-driven model selection decisions with production readiness in mind

Models evaluated:
1. Decision Tree (baseline)
2. Random Forest (bagging)
3. XGBoost (boosting)
4. Gradient Boosting (boosting alternative)

Author: Bruno Silva (Improved by Claude)
Date: 2025
================================================================================
"""

# ================================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ================================================================================

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)

warnings.filterwarnings('ignore')


# Configuration
class Config:
    """Configuration parameters for model evaluation pipeline."""
    # Input paths
    MODELS_DIR = Path('outputs') / 'models'
    PREDICTIONS_DIR = Path('outputs') / 'predictions'
    DATA_DIR = Path('outputs') / 'data_processed'
    Y_TEST_FILE = DATA_DIR / 'y_test.csv'
    Y_TRAIN_FILE = DATA_DIR / 'y_train.csv'

    # Output paths
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'

    # Model names (must match saved files)
    MODEL_NAMES = [
        'DecisionTree',
        'RandomForest',
        'XGBoost',
        'GradientBoosting'
    ]

    # Visualization settings
    FIGSIZE = (16, 12)
    DPI = 300
    CMAP = 'Blues'

    # Overfitting thresholds
    OVERFITTING_THRESHOLD = 0.05  # 5% gap between train and test
    PERFECT_SCORE_THRESHOLD = 0.995  # Scores above this are suspicious


# ================================================================================
# SECTION 2: UTILITY FUNCTIONS
# ================================================================================


def print_header(text: str, char: str = "=") -> None:
    """
    Print formatted section header.

    Args:
        text: Header text
        char: Border character
    """
    print(f"\n{char * 100}")
    print(text)
    print(char * 100)


def load_pickle(filepath: Path) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ================================================================================
# SECTION 3: DATA LOADING
# ================================================================================


def load_test_labels() -> pd.Series:
    """
    Load true test labels.

    Returns:
        Series with true test labels

    Raises:
        FileNotFoundError: If y_test.csv doesn't exist
    """
    print_header("STEP 1: LOADING TEST LABELS")

    if not Config.Y_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test labels not found at {Config.Y_TEST_FILE}\n"
            f"Please run 02_preprocessing.py first!"
        )

    y_test = pd.read_csv(Config.Y_TEST_FILE).squeeze()

    print("✓ Test labels loaded successfully")
    print(f"  Total samples: {len(y_test):,}")

    # Show class distribution
    counts = y_test.value_counts().sort_index()
    percentages = y_test.value_counts(normalize=True).sort_index() * 100

    print("\n  Class distribution:")
    print(f"    Class 0 (No incident):  {counts[0]:,} ({percentages[0]:.2f}%)")
    print(
        f"    Class 1 (With incident): {
            counts[1]:,} ({
            percentages[1]:.2f}%)")
    print(f"    Imbalance ratio: {counts[0] / counts[1]:.2f}:1")

    return y_test


def load_predictions() -> Dict[str, pd.DataFrame]:
    """
    Load predictions from all models.

    Returns:
        Dictionary mapping model names to prediction DataFrames

    Each DataFrame contains:
    - true_label
    - predicted_label
    - probability_class_0
    - probability_class_1
    """
    print_header("STEP 2: LOADING MODEL PREDICTIONS (TEST SET)")

    predictions = {}

    for model_name in Config.MODEL_NAMES:
        filepath = Config.PREDICTIONS_DIR / \
            f'{model_name.lower()}_predictions.csv'

        if not filepath.exists():
            print(f"  ⚠️  Warning: {filepath.name} not found, skipping...")
            continue

        df = pd.read_csv(filepath)
        predictions[model_name] = df
        print(f"  ✓ Loaded: {model_name} predictions ({len(df):,} samples)")

    print(f"\n✓ Loaded predictions from {len(predictions)} models")

    return predictions


def load_train_predictions() -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load training predictions for overfitting detection.

    Returns:
        Dictionary mapping model names to training prediction DataFrames,
        or None if training predictions are not available
    """
    print_header(
        "STEP 2.5: LOADING TRAINING PREDICTIONS (FOR OVERFITTING CHECK)")

    train_predictions = {}

    for model_name in Config.MODEL_NAMES:
        filepath = Config.PREDICTIONS_DIR / \
            f'{model_name.lower()}_train_predictions.csv'

        if not filepath.exists():
            print(f"  ⚠️  {filepath.name} not found")
            continue

        df = pd.read_csv(filepath)
        train_predictions[model_name] = df
        print(
            f"  ✓ Loaded: {model_name} training predictions ({
                len(df):,} samples)")

    if not train_predictions:
        print("\n⚠️  No training predictions found!")
        print("    Overfitting detection will be limited.")
        print("    Tip: Save training predictions during model training for better diagnostics.")
        return None

    print(
        f"\n✓ Loaded training predictions from {
            len(train_predictions)} models")
    return train_predictions


# ================================================================================
# SECTION 4: CALCULATE METRICS
# ================================================================================


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for a single model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for class 1

    Returns:
        Dictionary with all calculated metrics

    CRITICAL CONCEPT: METRICS FOR IMBALANCED DATA
    ----------------------------------------------
    When dealing with imbalanced data (e.g., 90% class 0, 10% class 1),
    different metrics tell different stories:

    1. ACCURACY: Total correct / Total samples
       - Problem: Misleading for imbalanced data
       - Example: Always predict class 0 → 90% accuracy!
       - This is called the "Accuracy Paradox"
       - Lesson: High accuracy ≠ Good model for imbalanced data

    2. PRECISION: True Positives / (True Positives + False Positives)
       - "Of predicted incidents, how many were real?"
       - Answers: "When I say incident, how often am I right?"
       - High precision = Low false alarm rate
       - Cost: Missing some real incidents (low recall)
       - Use case: When false alarms are expensive

    3. RECALL (Sensitivity): True Positives / (True Positives + False Negatives)
       - "Of real incidents, how many did we catch?"
       - Answers: "Am I finding all the incidents?"
       - High recall = Catch most incidents
       - Cost: More false alarms (low precision)
       - Use case: When missing incidents is dangerous

    4. F1-SCORE: Harmonic mean of Precision and Recall
       - Formula: 2 × (Precision × Recall) / (Precision + Recall)
       - Balances precision and recall
       - Better than accuracy for imbalanced data
       - Single metric that considers both false positives and false negatives
       - Range: 0 to 1 (higher is better)

    5. ROC-AUC: Area Under Receiver Operating Characteristic Curve
       - Measures model's ability to distinguish between classes
       - Threshold-independent metric
       - Range: 0.5 (random) to 1.0 (perfect)
       - Good for comparing models overall

    6. COHEN'S KAPPA: Agreement beyond chance
       - Accounts for class imbalance
       - Range: -1 to 1
       - < 0: Worse than random
       - 0-0.20: Slight agreement
       - 0.21-0.40: Fair
       - 0.41-0.60: Moderate
       - 0.61-0.80: Substantial
       - 0.81-1.00: Almost perfect
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1_Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_true, y_proba),
        'Cohen_Kappa': cohen_kappa_score(y_true, y_pred)
    }


def calculate_metrics_all_models(
    predictions: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate metrics for all models.

    Args:
        predictions: Dictionary of model predictions

    Returns:
        DataFrame with metrics for all models
    """
    print_header("STEP 3: CALCULATING EVALUATION METRICS (TEST SET)")

    print("\nCalculating metrics for each model...")

    metrics_list = []

    for model_name, pred_df in predictions.items():
        print(f"\n  Processing: {model_name}")

        # Extract data
        y_true = pred_df['true_label'].values
        y_pred = pred_df['predicted_label'].values
        y_proba = pred_df['probability_class_1'].values

        # Calculate metrics
        metrics = calculate_all_metrics(y_true, y_pred, y_proba)
        metrics['Model'] = model_name

        # Print key metrics
        print(f"    Accuracy: {metrics['Accuracy']:.4f}")
        print(f"    F1-Score: {metrics['F1_Score']:.4f}")
        print(f"    ROC-AUC:  {metrics['ROC_AUC']:.4f}")

        metrics_list.append(metrics)

    df = pd.DataFrame(metrics_list)
    df = df[['Model', 'Accuracy', 'Precision',
             'Recall', 'F1_Score', 'ROC_AUC', 'Cohen_Kappa']]

    print(f"\n✓ Metrics calculated for {len(df)} models")

    return df


# ================================================================================
# SECTION 4.5: OVERFITTING DETECTION (NEW)
# ================================================================================


def diagnose_overfitting(
    train_predictions: Optional[Dict[str, pd.DataFrame]],
    test_predictions: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Diagnose overfitting by comparing train vs test performance.

    Args:
        train_predictions: Dictionary of training predictions (can be None)
        test_predictions: Dictionary of test predictions

    Returns:
        DataFrame with overfitting analysis

    CRITICAL CONCEPT: OVERFITTING DETECTION
    ---------------------------------------
    Overfitting occurs when a model memorizes training data instead of
    learning generalizable patterns.

    Signs of overfitting:
    1. Perfect or near-perfect training accuracy (>99.5%)
    2. Large gap between train and test performance (>5%)
    3. Decision trees with no pruning/constraints
    4. Complex models on small datasets

    Why it matters:
    - Overfitted models fail on new data
    - They've memorized noise, not learned patterns
    - Perfect test scores are suspicious (especially for Decision Trees)

    What to do:
    - Use regularization (max_depth, min_samples_split)
    - Prefer ensemble methods (Random Forest, Gradient Boosting)
    - Cross-validation during training
    - Choose models with slight imperfections over "perfect" ones
    """
    print_header("STEP 3.5: OVERFITTING DETECTION")

    if train_predictions is None:
        print("\n⚠️  Training predictions not available - using test-only heuristics")
        print("\nChecking for suspiciously perfect scores...")

        overfitting_analysis = []

        for model_name, test_df in test_predictions.items():
            y_true = test_df['true_label'].values
            y_pred = test_df['predicted_label'].values
            y_proba = test_df['probability_class_1'].values

            test_metrics = calculate_all_metrics(y_true, y_pred, y_proba)

            # Check for perfect or near-perfect scores
            is_suspicious = (
                test_metrics['F1_Score'] >= Config.PERFECT_SCORE_THRESHOLD or
                test_metrics['Accuracy'] >= Config.PERFECT_SCORE_THRESHOLD
            )

            # Decision trees are especially prone to overfitting
            is_decision_tree = 'DecisionTree' in model_name or 'Tree' in model_name

            overfitting_risk = "HIGH" if (
                is_suspicious and is_decision_tree) else "MODERATE" if is_suspicious else "LOW"

            print(f"\n  {model_name}:")
            print(f"    Test F1-Score:   {test_metrics['F1_Score']:.4f}")
            print(f"    Test Accuracy:   {test_metrics['Accuracy']:.4f}")
            print(f"    Overfitting Risk: {overfitting_risk}")

            if is_suspicious:
                print("    ⚠️  WARNING: Near-perfect scores detected!")
                if is_decision_tree:
                    print(
                        "    🚩 Decision Tree with perfect scores = HIGH overfitting risk!")

            overfitting_analysis.append({
                'Model': model_name,
                'Test_F1': test_metrics['F1_Score'],
                'Test_Accuracy': test_metrics['Accuracy'],
                'Train_Test_Gap': np.nan,
                'Overfitting_Risk': overfitting_risk,
                'Has_Perfect_Score': is_suspicious
            })

        return pd.DataFrame(overfitting_analysis)

    # Full overfitting analysis with train/test comparison
    print("\n✓ Comparing training vs test performance...")

    overfitting_analysis = []

    for model_name in test_predictions.keys():
        if model_name not in train_predictions:
            print(
                f"\n  ⚠️  {model_name}: Training predictions missing, skipping...")
            continue

        print(f"\n  {model_name}:")

        # Calculate train metrics
        train_df = train_predictions[model_name]
        y_train_true = train_df['true_label'].values
        y_train_pred = train_df['predicted_label'].values
        y_train_proba = train_df['probability_class_1'].values
        train_metrics = calculate_all_metrics(
            y_train_true, y_train_pred, y_train_proba)

        # Calculate test metrics
        test_df = test_predictions[model_name]
        y_test_true = test_df['true_label'].values
        y_test_pred = test_df['predicted_label'].values
        y_test_proba = test_df['probability_class_1'].values
        test_metrics = calculate_all_metrics(
            y_test_true, y_test_pred, y_test_proba)

        # Calculate gaps
        f1_gap = train_metrics['F1_Score'] - test_metrics['F1_Score']
        train_metrics['Accuracy'] - test_metrics['Accuracy']

        print(f"    Train F1:        {train_metrics['F1_Score']:.4f}")
        print(f"    Test F1:         {test_metrics['F1_Score']:.4f}")
        print(f"    Gap:             {f1_gap:+.4f}")

        # Determine overfitting risk
        if f1_gap > Config.OVERFITTING_THRESHOLD:
            overfitting_risk = "HIGH"
            print(
                f"    🚩 WARNING: Significant overfitting detected! (gap > {
                    Config.OVERFITTING_THRESHOLD})")
        elif test_metrics['F1_Score'] >= Config.PERFECT_SCORE_THRESHOLD:
            overfitting_risk = "MODERATE"
            print("    ⚠️  Near-perfect test score - may indicate overfitting")
        else:
            overfitting_risk = "LOW"
            print("    ✓ Healthy generalization")

        overfitting_analysis.append({
            'Model': model_name,
            'Train_F1': train_metrics['F1_Score'],
            'Test_F1': test_metrics['F1_Score'],
            'Train_Test_Gap': f1_gap,
            'Overfitting_Risk': overfitting_risk,
            'Has_Perfect_Score': test_metrics['F1_Score'] >= Config.PERFECT_SCORE_THRESHOLD
        })

    return pd.DataFrame(overfitting_analysis)


# ================================================================================
# SECTION 5: CREATE COMPARISON TABLE
# ================================================================================


def create_metrics_comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create formatted comparison table with best values highlighted.

    Args:
        metrics_df: DataFrame with calculated metrics

    Returns:
        Formatted DataFrame ready for display
    """
    print_header("STEP 4: CREATING COMPARISON TABLE")

    # Create formatted version
    formatted_df = metrics_df.copy()

    # Format numeric columns to 4 decimal places
    numeric_cols = ['Accuracy', 'Precision',
                    'Recall', 'F1_Score', 'ROC_AUC', 'Cohen_Kappa']
    for col in numeric_cols:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")

    # Mark best values with asterisk
    print("\nBest performing model per metric:")
    for col in numeric_cols:
        best_idx = metrics_df[col].idxmax()
        best_model = metrics_df.loc[best_idx, 'Model']
        best_value = metrics_df.loc[best_idx, col]

        formatted_df.loc[best_idx, col] = f"{best_value:.4f}*"

        print(f"  {col:15}: {best_model:20} ({best_value:.4f})")

    print("\n✓ Comparison table created")
    print("  (* indicates best value for each metric)")

    return formatted_df


# ================================================================================
# SECTION 6: SAVE RESULTS
# ================================================================================


def save_metrics_tables(
    metrics_df: pd.DataFrame,
    formatted_df: pd.DataFrame
) -> None:
    """
    Save metrics tables to CSV and Markdown formats.

    Args:
        metrics_df: Raw metrics DataFrame
        formatted_df: Formatted metrics DataFrame
    """
    print_header("STEP 5: SAVING METRICS TABLES")

    # Ensure output directory exists
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSV (raw values for further analysis)
    csv_path = Config.OUTPUT_DIR / 'metrics_comparison.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path.name}")

    # Save Markdown (formatted for documentation)
    md_path = Config.OUTPUT_DIR / 'metrics_comparison.md'
    with open(md_path, 'w') as f:
        f.write("# Model Performance Comparison\n\n")
        f.write("## Metrics Summary\n\n")
        f.write(formatted_df.to_markdown(index=False))
        f.write("\n\n*Asterisk (*) indicates best value for each metric*\n")

    print(f"✓ Saved Markdown: {md_path.name}")


# ================================================================================
# SECTION 7: VISUALIZATIONS
# ================================================================================


def plot_confusion_matrices(predictions: Dict[str, pd.DataFrame]) -> None:
    """
    Generate confusion matrix plots for all models.

    Args:
        predictions: Dictionary of model predictions

    CONFUSION MATRIX INTERPRETATION
    --------------------------------
    For binary classification (No incident = 0, Incident = 1):

                    Predicted
                    0       1
    Actual  0    [TN]    [FP]    True Negatives & False Positives
            1    [FN]    [TP]    False Negatives & True Positives

    Where:
    - TN (True Negative): Correctly predicted no incident
    - FP (False Positive): Incorrectly predicted incident (false alarm)
    - FN (False Negative): Incorrectly predicted no incident (DANGEROUS!)
    - TP (True Positive): Correctly predicted incident

    For flight safety:
    - Minimize FN (missing incidents is dangerous)
    - FP are acceptable (false alarms are inconvenient but safe)
    - Goal: High TP, Low FN
    """
    print_header("STEP 6: GENERATING CONFUSION MATRICES")

    # Ensure graphics directory exists
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating confusion matrices...")

    _, axes = plt.subplots(2, 2, figsize=Config.FIGSIZE)
    axes = axes.ravel()

    for idx, (model_name, pred_df) in enumerate(predictions.items()):
        print(f"  Processing: {model_name}")

        # Extract data
        y_true = pred_df['true_label'].values
        y_pred = pred_df['predicted_label'].values

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=Config.CMAP,
            ax=axes[idx],
            cbar=True,
            square=True,
            xticklabels=['No Incident', 'Incident'],
            yticklabels=['No Incident', 'Incident']
        )

        axes[idx].set_title(f'{model_name}', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=12)
        axes[idx].set_xlabel('Predicted Label', fontsize=12)

        # Add interpretation text
        tn, fp, fn, tp = cm.ravel()
        info_text = (
            f"TN={tn} | FP={fp}\n"
            f"FN={fn} | TP={tp}\n"
            f"Focus: Minimize FN!"
        )
        axes[idx].text(
            0.5, -0.15, info_text,
            transform=axes[idx].transAxes,
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

    plt.tight_layout()

    # Save
    output_path = Config.GRAPHICS_DIR / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Confusion matrices saved: {output_path.name}")


# ================================================================================
# SECTION 8: CLASSIFICATION REPORTS
# ================================================================================


def print_classification_reports(predictions: Dict[str, pd.DataFrame]) -> None:
    """
    Print detailed classification reports for all models.

    Args:
        predictions: Dictionary of model predictions

    CLASSIFICATION REPORT EXPLANATION
    ---------------------------------
    The classification report shows precision, recall, and F1-score for each class.

    For imbalanced data, focus on:
    - Metrics for minority class (class 1)
    - Macro average (fair to both classes)
    - NOT weighted average (dominated by majority class)
    """
    print_header("STEP 7: DETAILED CLASSIFICATION REPORTS")

    for model_name, pred_df in predictions.items():
        print(f"\n{'=' * 100}")
        print(f"MODEL: {model_name}")
        print('=' * 100)

        # Extract data
        y_true = pred_df['true_label'].values
        y_pred = pred_df['predicted_label'].values

        # Generate and print report
        report = classification_report(
            y_true,
            y_pred,
            target_names=['No Incident (0)', 'Incident (1)'],
            digits=4
        )

        print(report)

        # Add interpretation
        print("\nInterpretation:")
        print("  • Focus on 'Incident (1)' metrics (minority class, most important)")
        print("  • High recall for class 1 = Catching most incidents")
        print("  • High precision for class 1 = Few false alarms")
        print("  • Macro avg = Fair comparison across classes (not dominated by majority)")


# ================================================================================
# SECTION 9: ANALYSIS AND INSIGHTS (IMPROVED)
# ================================================================================


def analyze_results(
    metrics_df: pd.DataFrame,
    overfitting_df: Optional[pd.DataFrame]
) -> None:
    """
    Analyze results and provide production-ready insights.

    Args:
        metrics_df: DataFrame with all test metrics
        overfitting_df: DataFrame with overfitting analysis (can be None)

    KEY QUESTIONS TO ANSWER
    -----------------------
    1. Which model is best overall?
       → Consider F1-Score, ROC-AUC, AND overfitting risk

    2. How much better than baseline (Decision Tree)?
       → Calculate improvement percentages

    3. Are there overfitting concerns?
       → Check train/test gaps and perfect scores

    4. What trade-offs exist?
       → Compare precision vs recall

    5. Is accuracy misleading?
       → Compare to class distribution

    6. What's the production recommendation?
       → Balance performance with generalization
    """
    print_header("STEP 8: RESULTS ANALYSIS AND INSIGHTS", "=")

    # ========================================
    # OVERFITTING ANALYSIS (NEW)
    # ========================================
    if overfitting_df is not None:
        print("\n🔍 OVERFITTING ANALYSIS")
        print("-" * 100)

        high_risk_models = overfitting_df[
            overfitting_df['Overfitting_Risk'] == 'HIGH'
        ]['Model'].tolist()

        perfect_score_models = overfitting_df[overfitting_df['Has_Perfect_Score']]['Model'].tolist(
        )

        if high_risk_models:
            print("⚠️  Models with HIGH overfitting risk:")
            for model in high_risk_models:
                print(f"    🚩 {model}")
                row = overfitting_df[overfitting_df['Model'] == model].iloc[0]
                if 'Train_Test_Gap' in row and not pd.isna(
                        row['Train_Test_Gap']):
                    print(
                        f"       Train-Test Gap: {row['Train_Test_Gap']:+.4f}")

        if perfect_score_models and not high_risk_models:
            print("⚠️  Models with suspiciously perfect scores:")
            for model in perfect_score_models:
                print(
                    f"    🟡 {model} - Perfect scores may indicate overfitting")

        if not high_risk_models and not perfect_score_models:
            print("✓ No significant overfitting detected across models")

        print("\n💡 Remember: Slight imperfections (97-99%) often indicate better generalization!")

    # ========================================
    # BEST MODELS PER METRIC
    # ========================================
    best_f1_idx = metrics_df['F1_Score'].idxmax()
    best_auc_idx = metrics_df['ROC_AUC'].idxmax()
    best_recall_idx = metrics_df['Recall'].idxmax()

    best_f1_model = metrics_df.loc[best_f1_idx, 'Model']
    best_auc_model = metrics_df.loc[best_auc_idx, 'Model']
    best_recall_model = metrics_df.loc[best_recall_idx, 'Model']

    print("\n🏆 BEST MODELS PER METRIC (TEST SET)")
    print("-" * 100)
    print(f"Best F1-Score:  {best_f1_model:<20} "
          f"({metrics_df.loc[best_f1_idx, 'F1_Score']:.4f})")
    print(f"Best ROC-AUC:   {best_auc_model:<20} "
          f"({metrics_df.loc[best_auc_idx, 'ROC_AUC']:.4f})")
    print(f"Best Recall:    {best_recall_model:<20} "
          f"({metrics_df.loc[best_recall_idx, 'Recall']:.4f})")

    # ========================================
    # COMPARISON TO BASELINE
    # ========================================
    baseline_idx = metrics_df[metrics_df['Model'] == 'DecisionTree'].index[0]

    print("\n📊 IMPROVEMENT OVER BASELINE (Decision Tree)")
    print("-" * 100)

    baseline_f1 = metrics_df.loc[baseline_idx, 'F1_Score']
    baseline_auc = metrics_df.loc[baseline_idx, 'ROC_AUC']

    for idx, row in metrics_df.iterrows():
        if row['Model'] == 'DecisionTree':
            continue

        f1_improvement = ((row['F1_Score'] - baseline_f1) / baseline_f1) * 100
        auc_improvement = (
            (row['ROC_AUC'] - baseline_auc) / baseline_auc) * 100

        print(f"{row['Model']:<20}:")
        print(f"  F1-Score improvement:  {f1_improvement:+6.2f}%")
        print(f"  ROC-AUC improvement:   {auc_improvement:+6.2f}%")

    # ========================================
    # ACCURACY PARADOX DEMONSTRATION
    # ========================================
    print("\n⚠️  ACCURACY PARADOX DEMONSTRATION")
    print("-" * 100)
    print("Reminder: Test set has ~90% class 0, ~10% class 1")
    print("\nA 'dumb' classifier that ALWAYS predicts class 0 would get:")
    print("  • Accuracy: ~90% (looks good!)")
    print("  • F1-Score: 0.00 (reveals it's useless)")
    print("  • Recall: 0.00 (catches ZERO incidents)")
    print("\nThis is why accuracy is misleading for imbalanced data!")

    print("\nOur models:")
    for idx, row in metrics_df.iterrows():
        print(f"  {row['Model']:<20}: Accuracy={row['Accuracy']:.4f}, "
              f"F1={row['F1_Score']:.4f}, Recall={row['Recall']:.4f}")

    print("\n→ Even if accuracy is high, check F1 and Recall to ensure")
    print("  the model actually catches incidents (not just predicting majority class)")

    # ========================================
    # PRECISION-RECALL TRADE-OFF
    # ========================================
    print("\n⚖️  PRECISION-RECALL TRADE-OFF")
    print("-" * 100)

    high_precision_model = metrics_df.loc[metrics_df['Precision'].idxmax(
    ), 'Model']
    high_recall_model = metrics_df.loc[metrics_df['Recall'].idxmax(), 'Model']

    if high_precision_model != high_recall_model:
        print("Different models excel at different aspects:")
        print(
            f"  • {high_precision_model}: Best precision (fewer false alarms)")
        print(f"  • {high_recall_model}: Best recall (catches more incidents)")
        print("\nFor flight safety, prioritize RECALL (don't miss incidents)")
        print("False alarms are inconvenient, but missed incidents are dangerous")
    else:
        print(
            f"{high_precision_model} achieves best balance of precision and recall")

    # ========================================
    # PRODUCTION RECOMMENDATION (IMPROVED)
    # ========================================
    print("\n🎯 PRODUCTION RECOMMENDATION")
    print("-" * 100)

    # Filter out high-risk overfitted models
    if overfitting_df is not None:
        safe_models = overfitting_df[
            overfitting_df['Overfitting_Risk'] != 'HIGH'
        ]['Model'].tolist()

        # Get metrics only for safe models
        safe_metrics = metrics_df[metrics_df['Model'].isin(safe_models)].copy()

        if len(safe_metrics) == 0:
            print("⚠️  WARNING: All models show overfitting concerns!")
            print("    Recommendation: Retrain with proper regularization")
            safe_metrics = metrics_df  # Fallback to all models
        elif len(safe_metrics) < len(metrics_df):
            print("✓ Excluding overfitted models from recommendation:")
            excluded = set(metrics_df['Model']) - set(safe_models)
            for model in excluded:
                print(f"    ✗ {model} (overfitting risk)")
            print()
    else:
        safe_metrics = metrics_df

    # Find best model among safe choices
    best_safe_recall_idx = safe_metrics['Recall'].idxmax()

    recommended_model = safe_metrics.loc[best_safe_recall_idx, 'Model']
    recommended_f1 = safe_metrics.loc[best_safe_recall_idx, 'F1_Score']
    recommended_recall = safe_metrics.loc[best_safe_recall_idx, 'Recall']

    print(f"🥇 RECOMMENDED MODEL: {recommended_model}")
    print(f"   F1-Score: {recommended_f1:.4f}")
    print(f"   Recall:   {recommended_recall:.4f}")
    print("\n   Why this model?")

    if recommended_recall >= 0.99:
        print(
            f"   ✓ Excellent recall ({
                recommended_recall:.1%}) - catches nearly all incidents")
    else:
        print(
            f"   ✓ Strong recall ({
                recommended_recall:.1%}) - catches most incidents")

    if recommended_f1 >= 0.97:
        print(f"   ✓ High F1-Score ({recommended_f1:.4f}) - excellent balance")

    if overfitting_df is not None:
        risk = overfitting_df[overfitting_df['Model'] ==
                              recommended_model]['Overfitting_Risk'].values[0]
        if risk == 'LOW':
            print("   ✓ Low overfitting risk - better generalization expected")
        elif risk == 'MODERATE':
            print("   ⚠️  Moderate overfitting risk - monitor performance on new data")

    # Alternative recommendations
    print("\n🥈 Alternative options:")
    top_3_models = safe_metrics.nlargest(3, 'F1_Score')
    for idx, row in top_3_models.iterrows():
        if row['Model'] != recommended_model:
            print(
                f"   • {
                    row['Model']}: F1={
                    row['F1_Score']:.4f}, Recall={
                    row['Recall']:.4f}")

    print("\n📋 For production deployment:")
    print(f"  1. Deploy {recommended_model} as primary model")
    print("  2. Consider ensemble of top 2-3 models for robustness")
    print("  3. Tune threshold to optimize recall if needed (currently at 0.5)")
    print("  4. Implement monitoring for:")
    print("     - Prediction distribution (watch for drift)")
    print("     - False negative rate (critical for safety)")
    print("     - Model performance over time")
    print("  5. Retrain quarterly or when performance degrades")
    print("  6. Keep validation set for ongoing evaluation")


# ================================================================================
# SECTION 10: MAIN FUNCTION
# ================================================================================


def main() -> None:
    """
    Main evaluation pipeline with overfitting detection.

    Workflow:
    1. Load test labels
    2. Load test predictions
    3. Load training predictions (for overfitting check)
    4. Diagnose overfitting
    5. Calculate test metrics
    6. Create comparison table
    7. Save tables
    8. Generate visualizations
    9. Print classification reports
    10. Analyze results with production recommendations
    """
    print("=" * 100)
    print("FLIGHT INCIDENT PREDICTION - MODEL EVALUATION (IMPROVED)")
    print("=" * 100)
    print("\nEvaluating 4 ensemble models on imbalanced test data")
    print("Focus: Metrics + Overfitting detection + Production readiness")

    # Step 1: Load test labels
    load_test_labels()

    # Step 2: Load predictions
    predictions = load_predictions()

    if not predictions:
        print("\n✗ ERROR: No predictions found!")
        print("  Please run 03_train_models.py first!")
        return

    # Step 2.5: Load training predictions for overfitting check
    train_predictions = load_train_predictions()

    # Step 3: Diagnose overfitting
    overfitting_df = diagnose_overfitting(train_predictions, predictions)

    # Step 4: Calculate metrics
    metrics_df = calculate_metrics_all_models(predictions)

    # Step 5: Create comparison table
    formatted_df = create_metrics_comparison_table(metrics_df)

    # Step 6: Save tables
    save_metrics_tables(metrics_df, formatted_df)

    # Step 7: Plot confusion matrices
    plot_confusion_matrices(predictions)

    # Step 8: Print classification reports
    print_classification_reports(predictions)

    # Step 9: Analyze results (with overfitting awareness)
    analyze_results(metrics_df, overfitting_df)

    print("\n" + "=" * 100)
    print("✅ MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 100)
    print("\nOutputs saved:")
    print(
        f"  • Metrics table (CSV): {
            Config.OUTPUT_DIR /
            'metrics_comparison.csv'}")
    print(
        f"  • Metrics table (MD):  {
            Config.OUTPUT_DIR /
            'metrics_comparison.md'}")
    print(
        f"  • Confusion matrices:  {
            Config.GRAPHICS_DIR /
            'confusion_matrices.png'}")
    print("\n💡 Key Insights:")
    print("  1. Review overfitting analysis above")
    print("  2. Perfect scores (100%) may indicate overfitting")
    print("  3. For production, prefer models with:")
    print("     - High recall (catch all incidents)")
    print("     - Moderate F1 (97-99% is great!)")
    print("     - Low overfitting risk")
    print("  4. Monitor performance on new data continuously")


# ================================================================================
# EXECUTION
# ================================================================================

if __name__ == "__main__":
    main()
