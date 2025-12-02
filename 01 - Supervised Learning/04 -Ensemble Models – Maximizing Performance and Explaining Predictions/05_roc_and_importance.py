#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FLIGHT INCIDENT PREDICTION - ROC CURVES AND FEATURE IMPORTANCE
================================================================================

Purpose: Visualize model discrimination and identify important features

This script provides two critical analyses:
1. ROC Curves: Compare discrimination ability across all models
2. Feature Importance: Understand which features drive predictions

Key Learning Objectives:
- Interpret ROC curves and AUC scores
- Understand feature importance in tree-based models
- Compare feature importance across different algorithms
- Connect technical metrics to business insights

Author: Bruno Silva
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
from typing import Dict, Any, List

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('ignore')


# Configuration
class Config:
    """Configuration parameters for ROC and importance analysis."""
    # Input paths
    MODELS_DIR = Path('outputs') / 'models'
    PREDICTIONS_DIR = Path('outputs') / 'predictions'
    DATA_DIR = Path('outputs') / 'data_processed'
    FEATURE_NAMES_FILE = DATA_DIR / 'feature_names.pkl'

    # Output paths
    GRAPHICS_DIR = Path('outputs') / 'graphics'

    # Model names
    MODEL_NAMES = [
        'DecisionTree',
        'RandomForest',
        'XGBoost',
        'GradientBoosting'
    ]

    # Visualization settings
    FIGSIZE_ROC = (10, 8)
    FIGSIZE_IMPORTANCE = (10, 8)
    FIGSIZE_COMPARISON = (16, 6)
    DPI = 300

    # Colors for ROC curves
    ROC_COLORS = {
        'DecisionTree': '#FF6B6B',      # Red
        'RandomForest': '#4ECDC4',      # Teal
        'XGBoost': '#45B7D1',           # Blue
        'GradientBoosting': '#96CEB4'   # Green
    }


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
# SECTION 3: LOAD DATA
# ================================================================================


def load_predictions() -> Dict[str, pd.DataFrame]:
    """
    Load predictions from all models.

    Returns:
        Dictionary mapping model names to prediction DataFrames
    """
    print_header("STEP 1: LOADING PREDICTIONS")

    predictions = {}

    for model_name in Config.MODEL_NAMES:
        filepath = Config.PREDICTIONS_DIR / \
            f'{model_name.lower()}_predictions.csv'

        if not filepath.exists():
            print(f"  ⚠️  Warning: {filepath.name} not found, skipping...")
            continue

        df = pd.read_csv(filepath)
        predictions[model_name] = df
        print(f"  ✓ Loaded: {model_name} predictions")

    print(f"\n✓ Loaded predictions from {len(predictions)} models")

    return predictions


def load_models() -> Dict[str, Any]:
    """
    Load trained models.

    Returns:
        Dictionary mapping model names to model objects
    """
    print_header("STEP 2: LOADING TRAINED MODELS")

    models = {}

    # Load Random Forest
    rf_path = Config.MODELS_DIR / 'randomforest_model.pkl'
    if rf_path.exists():
        models['RandomForest'] = load_pickle(rf_path)
        print("  ✓ Loaded: RandomForest model")

    # Load XGBoost
    xgb_path = Config.MODELS_DIR / 'xgboost_model.pkl'
    if xgb_path.exists():
        models['XGBoost'] = load_pickle(xgb_path)
        print("  ✓ Loaded: XGBoost model")

    print(f"\n✓ Loaded {len(models)} models for feature importance analysis")

    return models


def load_feature_names() -> List[str]:
    """
    Load feature names.

    Returns:
        List of feature names
    """
    if not Config.FEATURE_NAMES_FILE.exists():
        raise FileNotFoundError(
            f"Feature names not found at {Config.FEATURE_NAMES_FILE}\n"
            f"Please run 02_preprocessing.py first!"
        )

    feature_names = load_pickle(Config.FEATURE_NAMES_FILE)
    print(f"\n  ✓ Loaded {len(feature_names)} feature names")

    return feature_names


# ================================================================================
# SECTION 4: ROC CURVES
# ================================================================================


def calculate_roc_curves(
    predictions: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculate ROC curves for all models.

    Args:
        predictions: Dictionary of prediction DataFrames

    Returns:
        Dictionary with ROC data for each model

    CRITICAL CONCEPT: ROC CURVE (RECEIVER OPERATING CHARACTERISTIC)
    ---------------------------------------------------------------
    ROC curve shows model's discrimination ability across all thresholds.

    What is a threshold?
    - By default, we predict class 1 if probability > 0.5
    - But we can adjust this threshold (e.g., 0.3, 0.7, etc.)
    - Different thresholds give different precision-recall trade-offs

    ROC Curve Components:
    - X-axis: False Positive Rate (FPR) = FP / (FP + TN)
      → "Of actual negatives, how many did we wrongly call positive?"
      → Also called "Fall-out" or "1 - Specificity"

    - Y-axis: True Positive Rate (TPR) = TP / (TP + FN)
      → "Of actual positives, how many did we correctly identify?"
      → Same as Recall/Sensitivity

    - Each point on curve: One threshold value
      → Lower threshold → More predictions → Higher FPR and TPR
      → Higher threshold → Fewer predictions → Lower FPR and TPR

    Interpretation:

    1. Perfect Classifier (AUC = 1.0):
       - Curve hugs top-left corner
       - Can achieve TPR=1.0 with FPR=0.0
       - Completely separates classes

    2. Random Classifier (AUC = 0.5):
       - Diagonal line from (0,0) to (1,1)
       - TPR = FPR at all thresholds
       - No discrimination ability

    3. Good Classifier (AUC > 0.7):
       - Curve above diagonal
       - Higher TPR than FPR
       - Useful separation

    4. Excellent Classifier (AUC > 0.8):
       - Curve close to top-left
       - Much higher TPR than FPR
       - Strong discrimination

    AUC (Area Under Curve):
    - Summary metric: Single number for entire curve
    - Range: 0.0 to 1.0
    - Interpretation:
      → AUC = 0.5: No better than random guessing
      → AUC = 0.7: Acceptable discrimination
      → AUC = 0.8: Excellent discrimination
      → AUC = 0.9: Outstanding discrimination
      → AUC = 1.0: Perfect discrimination (rare in practice)

    - Probability interpretation:
      AUC = 0.85 means:
      "85% chance that model ranks a random positive sample
       higher than a random negative sample"

    Why ROC-AUC is great for imbalanced data:
    - Threshold-independent (considers ALL thresholds)
    - Not affected by class imbalance
    - Robust metric for comparison
    - Tells us about discrimination, not calibration

    Business application:
    - Use ROC curve to choose optimal threshold
    - Balance between catching incidents (TPR) and false alarms (FPR)
    - Different business contexts need different thresholds
    """
    print_header("STEP 3: CALCULATING ROC CURVES")

    roc_data = {}

    print("\nCalculating ROC curves for each model...")

    for model_name, pred_df in predictions.items():
        # Extract data
        y_true = pred_df['true_label'].values
        y_proba = pred_df['probability_class_1'].values

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        # Calculate AUC
        roc_auc = auc(fpr, tpr)

        roc_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }

        print(f"  {model_name:<20}: AUC = {roc_auc:.4f}")

    print(f"\n✓ ROC curves calculated for {len(roc_data)} models")

    return roc_data


def plot_roc_curves(roc_data: Dict[str, Dict[str, np.ndarray]]) -> None:
    """
    Create ROC curves comparison plot.

    Args:
        roc_data: Dictionary with ROC data for each model
    """
    print_header("STEP 4: PLOTTING ROC CURVES")

    # Create output directory
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=Config.FIGSIZE_ROC)

    # Plot diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2,
            label='Random Classifier (AUC = 0.50)', alpha=0.5)

    # Plot ROC curve for each model
    for model_name, data in roc_data.items():
        color = Config.ROC_COLORS.get(model_name, '#333333')
        ax.plot(
            data['fpr'],
            data['tpr'],
            color=color,
            linewidth=2.5,
            label=f"{model_name} (AUC = {data['auc']:.4f})",
            alpha=0.8
        )

    # Formatting
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curves Comparison - Flight Incident Prediction',
                 fontsize=14, fontweight='bold', pad=20)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

    # Add interpretation text
    ax.text(
        0.5, 0.05,
        'Better models: Curve closer to top-left corner (higher AUC)',
        horizontalalignment='center',
        fontsize=10,
        style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    # Save
    output_path = Config.GRAPHICS_DIR / 'roc_curves_comparison.png'
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()

    print(f"\n✓ ROC curves plot saved: {output_path.name}")


def analyze_roc_results(roc_data: Dict[str, Dict[str, np.ndarray]]) -> None:
    """
    Analyze and interpret ROC results.

    Args:
        roc_data: Dictionary with ROC data for each model
    """
    print_header("STEP 5: ROC CURVES INTERPRETATION")

    # Find best model
    best_model = max(roc_data.items(), key=lambda x: x[1]['auc'])
    best_model_name = best_model[0]
    best_auc = best_model[1]['auc']

    # Get baseline (Decision Tree)
    baseline_auc = roc_data.get('DecisionTree', {}).get('auc', 0.5)

    print("\n📊 ROC-AUC SCORES RANKING")
    print("-" * 100)

    # Sort by AUC
    sorted_models = sorted(
        roc_data.items(),
        key=lambda x: x[1]['auc'],
        reverse=True)

    for rank, (model_name, data) in enumerate(sorted_models, 1):
        auc_score = data['auc']

        # Interpret AUC
        if auc_score >= 0.9:
            interpretation = "Outstanding"
        elif auc_score >= 0.8:
            interpretation = "Excellent"
        elif auc_score >= 0.7:
            interpretation = "Acceptable"
        elif auc_score >= 0.6:
            interpretation = "Poor"
        else:
            interpretation = "Fail (no better than random)"

        # Calculate improvement over baseline
        if model_name != 'DecisionTree':
            improvement = ((auc_score - baseline_auc) / baseline_auc) * 100
            improvement_str = f" (+{improvement:5.2f}% over baseline)"
        else:
            improvement_str = " (baseline)"

        print(
            f"  {rank}. {
                model_name:<20}: AUC = {
                auc_score:.4f} ({interpretation}){improvement_str}")

    print("\n🏆 BEST MODEL")
    print("-" * 100)
    print(f"  Model: {best_model_name}")
    print(f"  AUC Score: {best_auc:.4f}")

    if best_auc >= 0.9:
        print("  → Outstanding discrimination ability!")
        print("  → Model can separate classes very well")
    elif best_auc >= 0.8:
        print("  → Excellent discrimination ability")
        print("  → Strong separation between classes")
    elif best_auc >= 0.7:
        print("  → Acceptable discrimination ability")
        print("  → Reasonable separation between classes")

    # Compare to baseline
    if best_model_name != 'DecisionTree':
        improvement = ((best_auc - baseline_auc) / baseline_auc) * 100
        print("\n  Improvement over Decision Tree baseline:")
        print(f"  • Baseline AUC: {baseline_auc:.4f}")
        print(f"  • Best AUC: {best_auc:.4f}")
        print(f"  • Improvement: +{improvement:.2f}%")

    print("\n💡 INTERPRETATION")
    print("-" * 100)
    print("  ROC-AUC measures model's ability to discriminate between classes:")
    print("  • 0.5 = Random guessing (no discrimination)")
    print("  • 0.7 = Acceptable (moderate discrimination)")
    print("  • 0.8 = Excellent (strong discrimination)")
    print("  • 0.9 = Outstanding (very strong discrimination)")
    print("  • 1.0 = Perfect (complete separation)")
    print("\n  For imbalanced data, ROC-AUC is more reliable than accuracy")
    print("  because it's not affected by class distribution.")


# ================================================================================
# SECTION 5: FEATURE IMPORTANCE - RANDOM FOREST
# ================================================================================


def plot_feature_importance_rf(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Plot feature importance for Random Forest.

    Args:
        model: Trained RandomForest model
        feature_names: List of feature names

    Returns:
        DataFrame with feature importances

    CRITICAL CONCEPT: FEATURE IMPORTANCE (GINI IMPORTANCE)
    ------------------------------------------------------
    Random Forest calculates feature importance using "Mean Decrease in Impurity"
    (also called Gini Importance):

    How it works:
    1. For each tree in the forest:
       - Track how much each feature decreases node impurity
       - Impurity = How mixed the classes are in a node
       - Good splits decrease impurity more

    2. Sum importance across all trees
    3. Normalize to percentages (sum to 100%)

    Interpretation:
    - High importance = Feature frequently used for important splits
    - Low importance = Feature rarely used or makes weak splits

    Example:
    - horas_voo_desde_ultima_manutencao: 35%
      → Most important feature
      → Used for many key splits
      → Strong predictor of incidents

    - previsao_turbulencia: 25%
      → Second most important
      → Also frequently used

    - missao_Vigilância: 2%
      → Rarely used
      → Weak predictor

    Caveats:
    1. Biased towards high-cardinality features
       - Features with many unique values favored
       - Not a problem for our scaled features

    2. Correlated features split importance
       - If two features correlated, importance shared
       - Both might be important but appear less so

    3. Doesn't show direction
       - Only importance, not if feature increases/decreases risk

    For business insights:
    - Focus on top 3-5 features
    - Connect to domain knowledge
    - Validate with stakeholders
    """
    print_header("STEP 6: FEATURE IMPORTANCE - RANDOM FOREST")

    # Extract feature importances
    importances = model.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Calculate percentages
    importance_df['Percentage'] = importance_df['Importance'] * 100

    print("\nTop 3 most important features (Random Forest):")
    for idx, row in importance_df.head(3).iterrows():
        print(f"  {row['Feature']:<40}: {row['Percentage']:6.2f}%")

    # Create plot
    fig, ax = plt.subplots(figsize=Config.FIGSIZE_IMPORTANCE)

    # Sort for plotting (ascending for horizontal bars)
    plot_df = importance_df.sort_values('Importance', ascending=True)

    # Create horizontal bar plot
    bars = ax.barh(
        plot_df['Feature'],
        plot_df['Percentage'],
        color='#4ECDC4',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )

    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, plot_df['Percentage'])):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%',
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    # Formatting
    ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance - Random Forest',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    output_path = Config.GRAPHICS_DIR / 'feature_importance_rf.png'
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()

    print(
        f"\n✓ Random Forest feature importance plot saved: {
            output_path.name}")

    return importance_df


# ================================================================================
# SECTION 6: FEATURE IMPORTANCE - XGBOOST
# ================================================================================


def plot_feature_importance_xgb(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Plot feature importance for XGBoost.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names

    Returns:
        DataFrame with feature importances

    CRITICAL CONCEPT: XGBOOST FEATURE IMPORTANCE (GAIN)
    ---------------------------------------------------
    XGBoost can calculate importance in three ways:

    1. GAIN (default, what we use):
       - Average gain of splits using the feature
       - Gain = Improvement in loss function
       - Higher gain = More useful for reducing error
       - Most similar to Random Forest's Gini importance

    2. WEIGHT (frequency):
       - Number of times feature appears in trees
       - Higher weight = Used more often
       - Can be misleading (frequent but weak splits)

    3. COVER (coverage):
       - Average coverage of splits using feature
       - Coverage = Number of samples affected
       - Considers how many samples the split influences

    We use GAIN because:
    - Most interpretable (actual improvement)
    - Similar to Random Forest (easier comparison)
    - Reflects true contribution to predictions

    Comparison with Random Forest:
    - Both measure importance via splits
    - XGBoost: Sequential (each tree corrects previous)
    - RF: Parallel (independent trees)
    - XGBoost might identify different features
    - Agreement between models = Strong signal
    - Disagreement = Worth investigating

    Business insights from importance:
    - Identifies key risk factors
    - Guides data collection priorities
    - Informs preventive maintenance strategies
    - Helps explain predictions to stakeholders
    """
    print_header("STEP 7: FEATURE IMPORTANCE - XGBOOST")

    # Extract feature importances (gain-based)
    importances = model.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Calculate percentages
    total = importance_df['Importance'].sum()
    importance_df['Percentage'] = (importance_df['Importance'] / total) * 100

    print("\nTop 3 most important features (XGBoost):")
    for idx, row in importance_df.head(3).iterrows():
        print(f"  {row['Feature']:<40}: {row['Percentage']:6.2f}%")

    # Create plot
    fig, ax = plt.subplots(figsize=Config.FIGSIZE_IMPORTANCE)

    # Sort for plotting (ascending for horizontal bars)
    plot_df = importance_df.sort_values('Importance', ascending=True)

    # Create horizontal bar plot
    bars = ax.barh(
        plot_df['Feature'],
        plot_df['Percentage'],
        color='#45B7D1',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )

    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, plot_df['Percentage'])):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%',
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    # Formatting
    ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance - XGBoost (Gain-based)',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    output_path = Config.GRAPHICS_DIR / 'feature_importance_xgb.png'
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()

    print(f"\n✓ XGBoost feature importance plot saved: {output_path.name}")

    return importance_df


# ================================================================================
# SECTION 7: FEATURE IMPORTANCE COMPARISON
# ================================================================================


def plot_importance_comparison(
    rf_importance: pd.DataFrame,
    xgb_importance: pd.DataFrame
) -> None:
    """
    Create side-by-side comparison of feature importance.

    Args:
        rf_importance: Random Forest importance DataFrame
        xgb_importance: XGBoost importance DataFrame
    """
    print_header("STEP 8: FEATURE IMPORTANCE COMPARISON")

    # Merge data
    comparison_df = pd.merge(
        rf_importance[['Feature', 'Percentage']].rename(columns={'Percentage': 'RF'}),
        xgb_importance[['Feature', 'Percentage']].rename(columns={'Percentage': 'XGB'}),
        on='Feature'
    )

    # Calculate average importance
    comparison_df['Average'] = (comparison_df['RF'] + comparison_df['XGB']) / 2

    # Sort by average
    comparison_df = comparison_df.sort_values('Average', ascending=False)

    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Config.FIGSIZE_COMPARISON)

    # Sort for plotting (ascending for horizontal bars)
    plot_df = comparison_df.sort_values('Average', ascending=True)

    y_pos = np.arange(len(plot_df))

    # Random Forest (left)
    ax1.barh(y_pos, plot_df['RF'], color='#4ECDC4', edgecolor='black',
             linewidth=1.5, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(plot_df['Feature'], fontsize=10)
    ax1.set_xlabel('Importance (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Random Forest', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add values
    for i, (y, val) in enumerate(zip(y_pos, plot_df['RF'])):
        ax1.text(val + 0.5, y, f'{val:.1f}%', va='center', fontsize=9)

    # XGBoost (right)
    ax2.barh(y_pos, plot_df['XGB'], color='#45B7D1', edgecolor='black',
             linewidth=1.5, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(plot_df['Feature'], fontsize=10)
    ax2.set_xlabel('Importance (%)', fontsize=11, fontweight='bold')
    ax2.set_title('XGBoost', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add values
    for i, (y, val) in enumerate(zip(y_pos, plot_df['XGB'])):
        ax2.text(val + 0.5, y, f'{val:.1f}%', va='center', fontsize=9)

    # Overall title
    fig.suptitle('Feature Importance Comparison: Random Forest vs XGBoost',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = Config.GRAPHICS_DIR / 'feature_importance_comparison.png'
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Feature importance comparison plot saved: {output_path.name}")


def analyze_importance_comparison(
    rf_importance: pd.DataFrame,
    xgb_importance: pd.DataFrame
) -> None:
    """
    Analyze and interpret feature importance comparison.

    Args:
        rf_importance: Random Forest importance DataFrame
        xgb_importance: XGBoost importance DataFrame
    """
    print_header("STEP 9: IMPORTANCE ANALYSIS AND INTERPRETATION")

    # Get top 3 from each model
    rf_top3 = set(rf_importance.head(3)['Feature'].tolist())
    xgb_top3 = set(xgb_importance.head(3)['Feature'].tolist())

    # Find agreement
    agreement = rf_top3.intersection(xgb_top3)
    disagreement = rf_top3.symmetric_difference(xgb_top3)

    print("\n🔍 TOP 3 FEATURES COMPARISON")
    print("-" * 100)

    print("\nRandom Forest top 3:")
    for feat in rf_importance.head(3)['Feature']:
        pct = rf_importance[rf_importance['Feature']
                            == feat]['Percentage'].values[0]
        print(f"  • {feat:<40} ({pct:5.2f}%)")

    print("\nXGBoost top 3:")
    for feat in xgb_importance.head(3)['Feature']:
        pct = xgb_importance[xgb_importance['Feature']
                             == feat]['Percentage'].values[0]
        print(f"  • {feat:<40} ({pct:5.2f}%)")

    print("\n📊 AGREEMENT ANALYSIS")
    print("-" * 100)

    if len(agreement) == 3:
        print("✓ PERFECT AGREEMENT: Both models agree on all top 3 features")
        print("  → High confidence in these features")
        print("  → Strong signal, not algorithm-specific")
        print("\n  Agreed features:")
        for feat in agreement:
            print(f"    • {feat}")

    elif len(agreement) >= 2:
        print(f"✓ STRONG AGREEMENT: {len(agreement)}/3 features in common")
        print("\n  Agreed features:")
        for feat in agreement:
            print(f"    • {feat}")

        if disagreement:
            print("\n  Disagreement on:")
            for feat in disagreement:
                if feat in rf_top3:
                    print(f"    • {feat:<40} (RF only)")
                else:
                    print(f"    • {feat:<40} (XGB only)")

    else:
        print(
            f"⚠️  LIMITED AGREEMENT: Only {
                len(agreement)}/3 features in common")
        print("  → Models identify different patterns")
        print("  → Consider ensemble approach")
        print("  → Investigate disagreements")

    # Correlation analysis
    print("\n📈 IMPORTANCE CORRELATION")
    print("-" * 100)

    # Merge and calculate correlation
    merged = pd.merge(
        rf_importance[['Feature', 'Percentage']].rename(columns={'Percentage': 'RF'}),
        xgb_importance[['Feature', 'Percentage']].rename(columns={'Percentage': 'XGB'}),
        on='Feature'
    )

    correlation = merged['RF'].corr(merged['XGB'])

    print(f"Correlation between RF and XGB importance: {correlation:.4f}")

    if correlation > 0.8:
        print("  → Very high correlation")
        print("  → Models largely agree on feature importance")
    elif correlation > 0.6:
        print("  → High correlation")
        print("  → Models generally agree")
    elif correlation > 0.4:
        print("  → Moderate correlation")
        print("  → Some agreement, some differences")
    else:
        print("  → Low correlation")
        print("  → Significant differences in importance rankings")

    # Business insights
    print("\n💡 BUSINESS INSIGHTS")
    print("-" * 100)

    # Get consistently important features (high in both models)
    merged['Average'] = (merged['RF'] + merged['XGB']) / 2
    top_overall = merged.nlargest(3, 'Average')

    print("\nMost important features (averaged across models):")
    for idx, row in top_overall.iterrows():
        print(f"\n  {idx + 1}. {row['Feature']}")
        print(
            f"     RF: {
                row['RF']:.2f}%, XGB: {
                row['XGB']:.2f}%, Avg: {
                row['Average']:.2f}%")

        # Add domain interpretation
        feat = row['Feature']
        if 'horas_voo' in feat.lower() or 'manutencao' in feat.lower():
            print("     💡 Insight: Maintenance timing is critical for safety")
        elif 'turb' in feat.lower():
            print("     💡 Insight: Weather conditions significantly impact incident risk")
        elif 'idade' in feat.lower():
            print("     💡 Insight: Aircraft age affects incident probability")
        elif 'experiencia' in feat.lower():
            print("     💡 Insight: Pilot experience is a key safety factor")
        elif 'missao' in feat.lower():
            print("     💡 Insight: Mission type influences incident patterns")

    print("\n🎯 RECOMMENDATIONS")
    print("-" * 100)
    print("Based on feature importance analysis:")
    print("\n1. Data Collection Priority:")
    print("   • Focus on top 3 features for quality and completeness")
    print("   • Ensure accurate measurement of high-importance features")

    print("\n2. Operational Focus:")
    print("   • Monitor high-importance features closely")
    print("   • Implement alerts for unusual values in key features")

    print("\n3. Model Interpretation:")
    print("   • Use SHAP values for individual prediction explanations")
    print("   • Features with consistent high importance across models are most reliable")

    print("\n4. Risk Mitigation:")
    print("   • Target interventions on modifiable high-importance features")
    print("   • Example: If maintenance timing is key, optimize schedules")


# ================================================================================
# SECTION 8: MAIN FUNCTION
# ================================================================================


def main() -> None:
    """
    Main pipeline for ROC curves and feature importance analysis.

    Workflow:
    1. Load predictions from all models
    2. Calculate and plot ROC curves
    3. Analyze ROC results
    4. Load trained models (RF, XGBoost)
    5. Load feature names
    6. Plot Random Forest feature importance
    7. Plot XGBoost feature importance
    8. Create comparison plot
    9. Analyze and interpret results
    """
    print("=" * 100)
    print("FLIGHT INCIDENT PREDICTION - ROC CURVES AND FEATURE IMPORTANCE")
    print("=" * 100)
    print("\nThis analysis provides:")
    print("  • ROC curves: Compare discrimination ability across models")
    print("  • Feature importance: Identify key predictors in ensemble models")
    print("  • Comparison: Validate important features across algorithms")

    # Part 1: ROC Curves
    predictions = load_predictions()

    if not predictions:
        print("\n✗ ERROR: No predictions found!")
        print("  Please run 03_train_models.py first!")
        return

    roc_data = calculate_roc_curves(predictions)
    plot_roc_curves(roc_data)
    analyze_roc_results(roc_data)

    # Part 2 & 3: Feature Importance
    models = load_models()
    feature_names = load_feature_names()

    rf_importance = None
    xgb_importance = None

    if 'RandomForest' in models:
        rf_importance = plot_feature_importance_rf(
            models['RandomForest'], feature_names)

    if 'XGBoost' in models:
        xgb_importance = plot_feature_importance_xgb(
            models['XGBoost'], feature_names)

    # Part 4: Comparison
    if rf_importance is not None and xgb_importance is not None:
        plot_importance_comparison(rf_importance, xgb_importance)
        analyze_importance_comparison(rf_importance, xgb_importance)
    else:
        print("\n⚠️  Warning: Could not create comparison (missing models)")

    print("\n" + "=" * 100)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 100)
    print("\nOutputs saved:")
    print(
        f"  • ROC curves: {
            Config.GRAPHICS_DIR /
            'roc_curves_comparison.png'}")
    print(
        f"  • RF importance: {
            Config.GRAPHICS_DIR /
            'feature_importance_rf.png'}")
    print(
        f"  • XGB importance: {
            Config.GRAPHICS_DIR /
            'feature_importance_xgb.png'}")
    print(
        f"  • Comparison: {
            Config.GRAPHICS_DIR /
            'feature_importance_comparison.png'}")
    print("\nKey takeaways:")
    print("  1. Best model identified by ROC-AUC")
    print("  2. Most important features validated across algorithms")
    print("  3. Business insights connected to technical findings")


# ================================================================================
# EXECUTION
# ================================================================================

if __name__ == "__main__":
    main()
