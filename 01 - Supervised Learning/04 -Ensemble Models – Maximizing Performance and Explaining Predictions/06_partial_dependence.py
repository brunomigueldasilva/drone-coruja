#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FLIGHT INCIDENT PREDICTION - PARTIAL DEPENDENCE PLOTS (XAI)
================================================================================

Purpose: Explain model predictions through Partial Dependence analysis

This script uses Partial Dependence Plots (PDP) to understand how top features
influence incident probability. PDPs are a key Explainable AI (XAI) technique
that helps "open the black box" of complex ensemble models.

UPDATED: Now includes production-aware model selection to avoid overfitted models

Key Learning Objectives:
- Understand marginal effects of features on predictions
- Identify thresholds and non-linear relationships
- Generate actionable business insights
- Validate model behavior makes sense
- Debug potential data or model issues

XAI Techniques Covered:
- Partial Dependence Plots (PDP) - 1D and 2D
- Feature interaction analysis
- Model interpretation framework

Author: Bruno Silva (Updated)
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
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay, partial_dependence

warnings.filterwarnings('ignore')


# Configuration
class Config:
    """Configuration parameters for Partial Dependence analysis."""
    # Input paths
    MODELS_DIR = Path('outputs') / 'models'
    DATA_DIR = Path('outputs') / 'data_processed'
    X_TEST_FILE = DATA_DIR / 'X_test_scaled.csv'
    FEATURE_NAMES_FILE = DATA_DIR / 'feature_names.pkl'
    METRICS_FILE = Path('outputs') / 'metrics_comparison.csv'

    # Output paths
    GRAPHICS_DIR = Path('outputs') / 'graphics'

    # Visualization settings
    FIGSIZE_1D = (18, 6)
    FIGSIZE_2D = (10, 8)
    DPI = 300

    # PDP settings
    N_JOBS = -1  # Use all CPU cores
    GRID_RESOLUTION = 50  # Number of points in PDP grid

    # Model selection
    PERFECT_THRESHOLD = 0.995  # Scores above this are suspicious


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
# SECTION 3: PRODUCTION-AWARE MODEL SELECTION
# ================================================================================


def identify_best_model() -> str:
    """
    Identify best model for production considering overfitting risks.

    Returns:
        Name of best model

    UPDATED BEHAVIOR:
    -----------------
    This function now implements intelligent model selection that:
    1. Detects models with suspiciously perfect scores (>99.5%)
    2. Prioritizes models with high recall (safety-critical)
    3. Excludes likely overfitted models from consideration
    4. Provides clear warnings about overfitting concerns

    Why this matters:
    - Perfect scores (100%) often indicate overfitting
    - Overfitted models don't generalize to new data
    - Production needs robust models, not memorizers
    - Slight imperfections (97-99%) are actually healthier
    """
    print_header("STEP 1: IDENTIFYING BEST PRODUCTION MODEL")

    if not Config.METRICS_FILE.exists():
        print(f"⚠️  Warning: Metrics file not found at {Config.METRICS_FILE}")
        print("  Defaulting to RandomForest (typically performs well)")
        return 'RandomForest'

    # Load metrics
    metrics_df = pd.read_csv(Config.METRICS_FILE)

    # Detect perfect or near-perfect scores
    perfect_models = metrics_df[
        metrics_df['F1_Score'] >= Config.PERFECT_THRESHOLD
    ]['Model'].tolist()

    if perfect_models:
        print("\n" + "=" * 100)
        print("⚠️  OVERFITTING DETECTION")
        print("=" * 100)
        print(
            f"\n⚠️  Models with suspiciously perfect scores: {
                ', '.join(perfect_models)}")
        print(
            f"   Perfect scores (≥{
                Config.PERFECT_THRESHOLD:.1%}) often indicate overfitting.")
        print("   These models may not generalize well to new data.")

        # Filter out perfect-scoring models
        safe_metrics = metrics_df[
            metrics_df['F1_Score'] < Config.PERFECT_THRESHOLD
        ].copy()

        if safe_metrics.empty:
            print("\n🚨 CRITICAL: ALL models show perfect scores!")
            print("   This strongly suggests overfitting.")
            print("   Proceeding with 'best' model but HIGHLY recommend retraining.")
            safe_metrics = metrics_df
    else:
        print("\n✓ No suspicious perfect scores detected")
        safe_metrics = metrics_df

    # Selection strategy: Prioritize recall for safety-critical application
    # Among models with high F1 (within 2% of best), pick highest recall
    top_f1_threshold = safe_metrics['F1_Score'].max() * 0.98
    candidates = safe_metrics[safe_metrics['F1_Score']
                              >= top_f1_threshold].copy()

    if candidates.empty:
        candidates = safe_metrics

    best_idx = candidates['Recall'].idxmax()
    best_model = candidates.loc[best_idx, 'Model']
    best_f1 = candidates.loc[best_idx, 'F1_Score']
    best_recall = candidates.loc[best_idx, 'Recall']

    print("\n✓ Production model selected (Recall-prioritized):")
    print(f"  Model:     {best_model}")
    print(f"  F1-Score:  {best_f1:.4f}")
    print(f"  Recall:    {best_recall:.4f}")

    if best_recall >= 0.99:
        print(
            f"  ✓ Excellent recall ({
                best_recall:.1%}) - catches nearly all incidents!")

    print(f"\n  Why {best_model}?")

    if best_f1 < 1.0:
        print(f"    • F1-Score of {best_f1:.4f} suggests good generalization")
        print("    • Slight imperfection is healthy (97-99% is excellent!)")

    if best_recall >= 0.99:
        print(
            f"    • {
                best_recall:.1%} recall - critical for safety applications")
        print("    • Catches nearly every incident")

    if best_model in ['GradientBoosting', 'RandomForest']:
        print("    • Ensemble method - more robust than single trees")
        print("    • Less prone to overfitting")

    # Show what was excluded
    if perfect_models:
        print("\n  ✗ Excluded models (likely overfitted):")
        for model in perfect_models:
            if model != best_model:
                f1 = metrics_df[metrics_df['Model']
                                == model]['F1_Score'].values[0]
                print(
                    f"    • {model} (F1={
                        f1:.4f}) - perfect score is suspicious")

    print("\n  Why this model for Partial Dependence Analysis?")
    print("    • Production-ready model selection")
    print("    • Considers overfitting risks")
    print("    • Ensures PDP insights are meaningful for real deployment")

    return best_model


def load_model_and_data(
    model_name: str
) -> Tuple[Any, pd.DataFrame, List[str]]:
    """
    Load trained model, test data, and feature names.

    Args:
        model_name: Name of model to load

    Returns:
        Tuple of (model, X_test, feature_names)
    """
    print_header("STEP 2: LOADING MODEL AND DATA")

    # Load model
    model_path = Config.MODELS_DIR / f'{model_name.lower()}_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Please run 03_train_models.py first!"
        )

    model = load_pickle(model_path)
    print(f"✓ Loaded model: {model_name}")

    # Load test data
    if not Config.X_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test data not found at {Config.X_TEST_FILE}\n"
            f"Please run 02_preprocessing.py first!"
        )

    X_test = pd.read_csv(Config.X_TEST_FILE)
    print(
        f"✓ Loaded test data: {
            X_test.shape[0]} samples × {
            X_test.shape[1]} features")

    # Load feature names
    if not Config.FEATURE_NAMES_FILE.exists():
        raise FileNotFoundError(
            f"Feature names not found at {Config.FEATURE_NAMES_FILE}\n"
            f"Please run 02_preprocessing.py first!"
        )

    feature_names = load_pickle(Config.FEATURE_NAMES_FILE)
    print(f"✓ Loaded feature names: {len(feature_names)} features")

    return model, X_test, feature_names


def identify_top_features(
        model: Any,
        feature_names: List[str],
        n_top: int = 3) -> List[str]:
    """
    Identify top N most important features.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        n_top: Number of top features to return

    Returns:
        List of top N feature names
    """
    print_header("STEP 3: IDENTIFYING TOP FEATURES")

    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Get top N
    top_features = importance_df.head(n_top)['Feature'].tolist()

    print(f"\nTop {n_top} most important features for PDP analysis:")
    for idx, row in importance_df.head(n_top).iterrows():
        pct = row['Importance'] * 100
        print(f"  {idx + 1}. {row['Feature']:<40} ({pct:5.2f}%)")

    print("\n✓ Selected features for Partial Dependence analysis")

    return top_features
# SECTION 4: PARTIAL DEPENDENCE PLOTS (1D)
# ================================================================================


def plot_partial_dependence_1d(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: List[str],
    top_features: List[str]
) -> Dict[str, np.ndarray]:
    """
    Create 1D Partial Dependence Plots for top features.

    Args:
        model: Trained model
        X_test: Test features
        feature_names: All feature names
        top_features: Top features to analyze

    Returns:
        Dictionary with PDP data for each feature

    CRITICAL CONCEPT: PARTIAL DEPENDENCE PLOTS (PDP)
    -------------------------------------------------
    PDPs show the marginal effect of a feature on predictions.

    What is "marginal effect"?
    - Effect of one feature while averaging out all others
    - Answers: "How does prediction change as this feature changes,
               on average across all other feature combinations?"

    How PDP works (simplified):
    1. Pick a feature (e.g., maintenance_hours)
    2. Create grid of values (0, 50, 100, 150, ..., 500)
    3. For each grid value:
       a. Replace ALL samples' maintenance_hours with that value
       b. Keep all other features at their original values
       c. Get predictions for all samples
       d. Average predictions
    4. Plot average prediction vs grid values

    Mathematical intuition:
    PDP(x_s) = E[f(x_s, X_c)]

    Where:
    - x_s: Feature of interest (we vary this)
    - X_c: All other features (we average over these)
    - f: Model prediction function
    - E: Expected value (average)

    Example interpretation:
    Maintenance hours PDP:
    - At 0 hours: Average incident probability = 5%
    - At 100 hours: Average incident probability = 8%
    - At 300 hours: Average incident probability = 25%
    → Insight: Risk increases with maintenance delay
    → Action: Schedule maintenance before 300 hours

    Benefits of PDP:
    1. Model-agnostic (works with any model)
    2. Intuitive visualization
    3. Reveals non-linearities and thresholds
    4. Directly actionable insights
    5. Validates model makes sense

    Limitations:
    1. ASSUMES FEATURE INDEPENDENCE
       - If features correlated, PDP can be misleading
       - Example: If age and maintenance_hours correlated,
         PDP shows unrealistic scenarios (old aircraft with 0 maintenance hours)

    2. Shows AVERAGE effect only
       - Hides heterogeneity (individual variation)
       - Solution: Use ICE plots (Individual Conditional Expectation)

    3. Can't detect interactions between features
       - Shows marginal effects only
       - Solution: Use 2D PDPs for pairs of features

    When to trust PDP:
    - Features relatively independent
    - PDP shape makes business sense
    - Confirmed by domain experts
    - Consistent across multiple models

    When to be cautious:
    - Highly correlated features
    - Unexpected non-monotonic relationships
    - Sharp discontinuities (may indicate data issues)
    - Contradicts domain knowledge

    Business applications:
    1. Policy recommendations:
       - "Maintain aircraft before X hours"
       - "Assign experienced pilots (>Y years) to high-risk flights"

    2. Risk assessment:
       - Quantify risk increase per unit change
       - Identify critical thresholds

    3. Model validation:
       - Check predictions align with expectations
       - Detect potential bugs or data issues

    4. Stakeholder communication:
       - Explain model behavior to non-technical audience
       - Build trust in AI system
    """
    print_header("STEP 4: GENERATING 1D PARTIAL DEPENDENCE PLOTS")

    # Create output directory
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating PDPs for top features...")

    # Get feature indices
    feature_indices = [feature_names.index(feat) for feat in top_features]

    # Create figure with subplots
    fig, axes = plt.subplots(1, len(top_features), figsize=Config.FIGSIZE_1D)

    # Ensure axes is iterable
    if len(top_features) == 1:
        axes = [axes]

    # Store PDP data for interpretation
    pdp_data = {}

    # Generate PDP for each feature
    for idx, (feat_idx, feat_name) in enumerate(
            zip(feature_indices, top_features)):
        print(f"  Processing: {feat_name}")

        # Calculate partial dependence
        pd_result = partial_dependence(
            model,
            X_test,
            features=[feat_idx],
            grid_resolution=Config.GRID_RESOLUTION,
            kind='average'
        )

        # Store data
        pdp_data[feat_name] = {
            'values': pd_result['grid_values'][0],
            'average': pd_result['average'][0]
        }

        # Plot
        ax = axes[idx]
        ax.plot(
            pd_result['grid_values'][0],
            pd_result['average'][0],
            linewidth=3,
            color='#2E86AB',
            marker='o',
            markersize=4,
            alpha=0.8
        )

        # Formatting
        ax.set_xlabel(
            feat_name.replace(
                '_',
                ' ').title(),
            fontsize=11,
            fontweight='bold')
        ax.set_ylabel(
            'Average Prediction\n(Incident Probability)',
            fontsize=11,
            fontweight='bold')
        ax.set_title(
            f'Partial Dependence: {
                feat_name.replace(
                    "_",
                    " ").title()}',
            fontsize=12,
            fontweight='bold',
            pad=15)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add horizontal line at 0 for reference
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Overall title
    fig.suptitle(
        'Partial Dependence Plots - Top 3 Features',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )

    # Add explanation text
    fig.text(
        0.5, -0.05,
        'PDP shows average effect of feature on prediction, marginalizing over other features',
        ha='center',
        fontsize=10,
        style='italic'
    )

    plt.tight_layout()

    # Save
    output_path = Config.GRAPHICS_DIR / 'partial_dependence_top3.png'
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 1D Partial Dependence plots saved: {output_path.name}")

    return pdp_data


# ================================================================================
# SECTION 5: PARTIAL DEPENDENCE INTERPRETATION
# ================================================================================


def interpret_pdp_relationship(
    feature_name: str,
    values: np.ndarray,
    predictions: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze PDP relationship type and extract insights.

    Args:
        feature_name: Name of feature
        values: Feature values (x-axis)
        predictions: Average predictions (y-axis)

    Returns:
        Dictionary with interpretation metadata
    """
    # Calculate statistics
    min_pred = predictions.min()
    max_pred = predictions.max()
    range_pred = max_pred - min_pred

    # Determine relationship type
    # 1. Check monotonicity
    diffs = np.diff(predictions)
    increasing = np.sum(diffs > 0) / len(diffs)
    decreasing = np.sum(diffs < 0) / len(diffs)

    if increasing > 0.8:
        monotonicity = "monotonic_increasing"
        description = "consistently increases"
    elif decreasing > 0.8:
        monotonicity = "monotonic_decreasing"
        description = "consistently decreases"
    elif abs(increasing - decreasing) < 0.2:
        monotonicity = "non_monotonic"
        description = "varies non-monotonically"
    else:
        monotonicity = "mostly_monotonic"
        description = "generally increases" if increasing > decreasing else "generally decreases"

    # 2. Check linearity (using R² of linear fit)
    from sklearn.metrics import r2_score

    # Fit linear model
    coef = np.polyfit(values, predictions, 1)
    linear_pred = np.polyval(coef, values)
    r2 = r2_score(predictions, linear_pred)

    if r2 > 0.95:
        linearity = "linear"
        shape = "Linear relationship"
    elif r2 > 0.80:
        linearity = "mostly_linear"
        shape = "Approximately linear"
    else:
        linearity = "non_linear"
        shape = "Non-linear relationship"

    # 3. Find thresholds (points of significant change)
    # Calculate second derivative (curvature)
    second_deriv = np.diff(diffs)
    threshold_indices = np.where(
        np.abs(second_deriv) > np.percentile(
            np.abs(second_deriv), 90))[0]

    thresholds = []
    if len(threshold_indices) > 0:
        for idx in threshold_indices[:2]:  # Max 2 thresholds
            if idx < len(values) - 1:
                thresholds.append({
                    'value': values[idx + 1],
                    'prediction_before': predictions[idx],
                    'prediction_after': predictions[idx + 2] if idx + 2 < len(predictions) else predictions[-1]
                })

    return {
        'feature': feature_name,
        'monotonicity': monotonicity,
        'description': description,
        'linearity': linearity,
        'shape': shape,
        'min_prediction': min_pred,
        'max_prediction': max_pred,
        'range': range_pred,
        'thresholds': thresholds,
        'r2_linear': r2
    }


def print_interpretations(
    pdp_data: Dict[str, np.ndarray],
    feature_names: List[str]
) -> None:
    """
    Print detailed interpretations of PDP results.

    Args:
        pdp_data: Dictionary with PDP data for each feature
        feature_names: List of feature names
    """
    print_header("STEP 5: INTERPRETING PARTIAL DEPENDENCE PLOTS")

    for feat_name in feature_names:
        if feat_name not in pdp_data:
            continue

        values = pdp_data[feat_name]['values']
        predictions = pdp_data[feat_name]['average']

        # Analyze relationship
        analysis = interpret_pdp_relationship(feat_name, values, predictions)

        print(f"\n{'=' * 100}")
        print(f"FEATURE: {feat_name.replace('_', ' ').upper()}")
        print('=' * 100)

        # Basic statistics
        print("\n📊 Prediction Range:")
        print(f"  Minimum: {analysis['min_prediction']:.4f}")
        print(f"  Maximum: {analysis['max_prediction']:.4f}")
        print(f"  Range:   {analysis['range']:.4f}")

        # Calculate percentage change
        if analysis['min_prediction'] != 0:
            pct_change = (analysis['range'] /
                          abs(analysis['min_prediction'])) * 100
        else:
            pct_change = 0

        print(
            f"  → Feature causes {
                abs(pct_change):.1f}% change in prediction")

        # Relationship type
        print("\n📈 Relationship Type:")
        print(f"  Shape: {analysis['shape']}")
        print(
            f"  Monotonicity: {
                analysis['monotonicity'].replace(
                    '_', ' ').title()}")
        print(
            f"  Description: Incident probability {
                analysis['description']} as {feat_name} increases")

        if analysis['linearity'] == 'linear':
            print(
                f"  → Linear relationship (R² = {
                    analysis['r2_linear']:.3f})")
            print("  → Simple business rule possible")
        else:
            print(
                f"  → Non-linear relationship (R² = {analysis['r2_linear']:.3f})")
            print("  → Complex patterns require model-based decisions")

        # Thresholds
        if analysis['thresholds']:
            print("\n⚠️  Critical Thresholds Detected:")
            for i, threshold in enumerate(analysis['thresholds'], 1):
                print(f"  Threshold {i}: Around {threshold['value']:.2f}")
                change = threshold['prediction_after'] - \
                    threshold['prediction_before']
                print(f"    Prediction change: {change:+.4f}")
                if change > 0:
                    print("    → Risk increases significantly beyond this point")
                else:
                    print("    → Risk decreases significantly beyond this point")

        # Feature-specific insights
        print("\n💡 Business Insights:")

        if 'horas' in feat_name.lower() or 'manutencao' in feat_name.lower():
            print("  • Maintenance timing is critical for incident prevention")
            if analysis['monotonicity'] == 'monotonic_increasing':
                print("  • Risk consistently increases with delayed maintenance")
                max_safe_hours = values[np.argmax(predictions < (
                    predictions.min() + analysis['range'] * 0.3))]
                print(
                    f"  • Recommendation: Schedule maintenance before {
                        max_safe_hours:.0f} hours")
            print("  • Implement alerts for approaching maintenance threshold")

        elif 'turb' in feat_name.lower():
            print("  • Weather conditions strongly impact incident risk")
            if len(predictions) >= 3:
                print("  • Risk levels by turbulence forecast:")
                print(
                    f"    - Low turbulence:    {predictions[0]:.4f} (baseline)")
                if len(predictions) > 1:
                    print(f"    - Medium turbulence: {predictions[len(predictions) // 2]:.4f} "
                          f"({(predictions[len(predictions) // 2] / predictions[0]):.1f}× baseline)")
                print(f"    - High turbulence:   {predictions[-1]:.4f} "
                      f"({(predictions[-1] / predictions[0]):.1f}× baseline)")
            print("  • Consider flight rescheduling during high turbulence forecasts")

        elif 'idade' in feat_name.lower() or 'age' in feat_name.lower():
            print("  • Aircraft age influences incident probability")
            if analysis['monotonicity'] == 'monotonic_increasing':
                print("  • Older aircraft require additional safety measures")
                print("  • Consider early retirement for aged fleet")
            print("  • Prioritize inspections for older aircraft")

        elif 'experiencia' in feat_name.lower() or 'experience' in feat_name.lower():
            print("  • Pilot experience is a key safety factor")
            if analysis['monotonicity'] == 'monotonic_decreasing':
                print("  • More experienced pilots show lower incident rates")
                # Find experience level where risk plateaus
                if len(predictions) > 5:
                    plateau_idx = np.argmax(
                        np.abs(np.diff(predictions)) < 0.001)
                    if plateau_idx > 0:
                        plateau_exp = values[plateau_idx]
                        print(
                            f"  • Experience effect plateaus around {
                                plateau_exp:.1f} years")
                print("  • Assign high-risk flights to experienced pilots")
                print("  • Implement enhanced training for junior pilots")

        elif 'missao' in feat_name.lower() or 'mission' in feat_name.lower():
            print("  • Mission type affects incident patterns")
            print("  • Develop mission-specific safety protocols")
            print("  • Adjust risk assessments based on mission type")

        # Actionability assessment
        print("\n🎯 Actionability:")
        if analysis['monotonicity'] in [
            'monotonic_increasing',
                'monotonic_decreasing']:
            print("  ✓ Clear monotonic relationship enables simple decision rules")
            print("  ✓ Business stakeholders can easily understand and act on insights")
        else:
            print("  ⚠️  Complex relationship requires careful interpretation")
            print("  ⚠️  Consider consulting domain experts for validation")


# ================================================================================
# SECTION 6: 2D PARTIAL DEPENDENCE (INTERACTION)
# ================================================================================


def plot_partial_dependence_2d(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: List[str],
    top_features: List[str]
) -> None:
    """
    Create 2D Partial Dependence Plot for top 2 features.

    Args:
        model: Trained model
        X_test: Test features
        feature_names: All feature names
        top_features: Top features (use first 2)

    CRITICAL CONCEPT: 2D PARTIAL DEPENDENCE (INTERACTION EFFECTS)
    -------------------------------------------------------------
    2D PDP shows how TWO features jointly influence predictions.

    Why 2D PDP?
    - 1D PDP shows marginal effect of single feature
    - But features might interact (effect depends on other feature)
    - 2D PDP reveals these interactions

    Example interaction:
    Feature 1: Maintenance hours
    Feature 2: Aircraft age

    No interaction:
    - Effect of maintenance hours same for all ages
    - 1D PDPs sufficient

    Strong interaction:
    - Old aircraft: Maintenance delay → High risk increase
    - New aircraft: Maintenance delay → Small risk increase
    - Effect of maintenance depends on age!
    - Need 2D PDP to see this

    Reading 2D PDP:
    - Contour plot or heatmap
    - X-axis: Feature 1
    - Y-axis: Feature 2
    - Color: Prediction
    - Parallel contours = No interaction (additive effects)
    - Curved/twisted contours = Interaction present

    Business value:
    - Identify synergistic risk factors
    - Develop conditional policies
    - Example: "If aircraft age > X AND maintenance > Y, require inspection"

    Limitations:
    - Only shows pair-wise interactions
    - Computationally expensive
    - Hard to visualize >2 features
    """
    print_header("STEP 6: GENERATING 2D PARTIAL DEPENDENCE PLOT")

    if len(top_features) < 2:
        print("⚠️  Need at least 2 features for 2D PDP. Skipping...")
        return

    print(f"\nGenerating 2D PDP for: {top_features[0]} × {top_features[1]}")

    # Get feature indices
    feature_indices = [feature_names.index(top_features[0]),
                       feature_names.index(top_features[1])]

    # Create figure
    fig, ax = plt.subplots(figsize=Config.FIGSIZE_2D)

    # Generate 2D PDP
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_test,
        features=[feature_indices],
        grid_resolution=Config.GRID_RESOLUTION,
        ax=ax,
        kind='average'
    )

    # Enhance formatting
    ax.set_xlabel(
        top_features[0].replace(
            '_',
            ' ').title(),
        fontsize=12,
        fontweight='bold')
    ax.set_ylabel(
        top_features[1].replace(
            '_',
            ' ').title(),
        fontsize=12,
        fontweight='bold')
    ax.set_title(
        f'2D Partial Dependence: {
            top_features[0].replace(
                "_",
                " ").title()} × ' f'{
            top_features[1].replace(
                "_",
                " ").title()}',
        fontsize=13,
        fontweight='bold',
        pad=20)

    # Add colorbar label
    cbar = display.figure_.axes[-1]
    cbar.set_ylabel('Average Prediction\n(Incident Probability)',
                    fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Config.GRAPHICS_DIR / 'partial_dependence_interaction.png'
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ 2D Partial Dependence plot saved: {output_path.name}")

    # Interpretation
    print("\n📊 2D PDP Interpretation Guide:")
    print("  • Parallel contours = No interaction (effects are additive)")
    print("  • Curved contours = Interaction present (effects depend on each other)")
    print("  • Color intensity = Incident probability")
    print("  • Hot zones (red) = High-risk combinations")
    print("  • Cool zones (blue) = Low-risk combinations")

    print("\n💡 Business Application:")
    print("  • Identify dangerous feature combinations")
    print("  • Develop conditional safety rules")
    print(
        f"  • Example: 'If {
            top_features[0]} > X AND {
            top_features[1]} > Y, alert'")


# ================================================================================
# SECTION 7: SUMMARY AND RECOMMENDATIONS
# ================================================================================


def print_summary_and_recommendations(
    model_name: str,
    top_features: List[str],
    pdp_data: Dict[str, np.ndarray]
) -> None:
    """
    Print comprehensive summary and actionable recommendations.

    Args:
        model_name: Name of analyzed model
        top_features: List of top features
        pdp_data: PDP data for interpretation
    """
    print_header("PARTIAL DEPENDENCE ANALYSIS SUMMARY", "=")

    print("\n📋 ANALYSIS OVERVIEW")
    print("-" * 100)
    print(f"  Model analyzed: {model_name}")
    print(f"  Features analyzed: {len(top_features)}")
    print("  Method: Partial Dependence Plots (PDP)")
    print(f"  Grid resolution: {Config.GRID_RESOLUTION} points")

    print("\n🎯 KEY FINDINGS")
    print("-" * 100)

    # Rank features by impact
    impacts = []
    for feat in top_features:
        if feat in pdp_data:
            preds = pdp_data[feat]['average']
            impact = preds.max() - preds.min()
            impacts.append((feat, impact))

    impacts.sort(key=lambda x: x[1], reverse=True)

    print("\n  Feature impact ranking (by prediction range):")
    for rank, (feat, impact) in enumerate(impacts, 1):
        print(f"    {rank}. {feat:<40}: {impact:.4f} prediction range")

    print("\n🔍 XAI INSIGHTS")
    print("-" * 100)
    print("  Explainable AI (XAI) techniques used:")
    print("    • Partial Dependence Plots (PDP)")
    print("      → Shows marginal effect of features")
    print("      → Model-agnostic (works with any model)")
    print("      → Reveals non-linearities and thresholds")

    print("\n  Benefits of PDP analysis:")
    print("    ✓ Makes 'black box' models interpretable")
    print("    ✓ Validates model behavior makes sense")
    print("    ✓ Identifies actionable risk factors")
    print("    ✓ Builds stakeholder trust")
    print("    ✓ Enables data-driven policies")

    print("\n⚠️  LIMITATIONS TO CONSIDER")
    print("-" * 100)
    print("  1. Independence assumption:")
    print("     • PDP assumes features are independent")
    print("     • May show unrealistic scenarios if features correlated")
    print("     • Example: Old aircraft with zero maintenance hours")

    print("\n  2. Average effects only:")
    print("     • PDP shows average across all samples")
    print("     • Hides individual variation (heterogeneity)")
    print("     • Solution: Use ICE plots for individual effects")

    print("\n  3. Limited interaction detection:")
    print("     • 1D PDP can't show interactions between features")
    print("     • 2D PDP limited to pairs")
    print("     • Complex interactions may remain hidden")

    print("\n📊 COMPLEMENTARY TECHNIQUES")
    print("-" * 100)
    print("  To enhance explainability, consider:")
    print("    • SHAP values: Individual prediction explanations")
    print("    • ICE plots: Individual Conditional Expectation curves")
    print("    • LIME: Local surrogate models")
    print("    • Feature importance: Which features matter most")
    print("    • Counterfactual explanations: 'What if' scenarios")

    print("\n🎯 ACTIONABLE RECOMMENDATIONS")
    print("-" * 100)
    print("\n  1. Operational Policies:")
    print("     • Use PDP insights to set maintenance schedules")
    print("     • Implement risk thresholds based on identified patterns")
    print("     • Develop conditional safety protocols")

    print("\n  2. Risk Management:")
    print("     • Focus monitoring on high-impact features")
    print("     • Create alerts for critical threshold violations")
    print("     • Prioritize interventions on modifiable features")

    print("\n  3. Model Governance:")
    print("     • Document PDP findings for regulatory compliance")
    print("     • Share visualizations with stakeholders")
    print("     • Update analysis when model retrained")
    print("     • Validate predictions align with domain expertise")

    print("\n  4. Continuous Improvement:")
    print("     • Monitor if real-world patterns match PDP predictions")
    print("     • Refine features based on importance and interpretability")
    print("     • Consider simpler models if PDP shows linear relationships")
    print("     • Retrain models if PDP patterns shift over time")

    print("\n💡 BUSINESS VALUE")
    print("-" * 100)
    print("  PDP analysis enables:")
    print("    • Evidence-based decision making")
    print("    • Transparent AI systems")
    print("    • Regulatory compliance (explainability requirements)")
    print("    • Stakeholder trust and buy-in")
    print("    • Identification of root causes")
    print("    • Data-driven policy optimization")


# ================================================================================
# SECTION 8: MAIN FUNCTION
# ================================================================================


def main() -> None:
    """
    Main pipeline for Partial Dependence analysis.

    Workflow:
    1. Identify best model based on F1-Score
    2. Load model, test data, and feature names
    3. Identify top 3 most important features
    4. Generate 1D PDPs for top 3 features
    5. Interpret PDP relationships and extract insights
    6. Generate 2D PDP for top 2 features (interaction)
    7. Print comprehensive summary and recommendations
    """
    print("=" * 100)
    print("FLIGHT INCIDENT PREDICTION - PARTIAL DEPENDENCE ANALYSIS (XAI)")
    print("=" * 100)
    print("\nExplainable AI (XAI) technique: Partial Dependence Plots")
    print("Goal: Understand how top features influence incident probability")
    print("\n✨ IMPROVED: Now uses production-aware model selection")
    print("  • Detects and excludes overfitted models")
    print("  • Prioritizes models that generalize well")
    print("  • Ensures XAI insights are production-relevant")

    # Step 1: Identify best model
    model_name = identify_best_model()

    # Step 2: Load model and data
    model, X_test, feature_names = load_model_and_data(model_name)

    # Step 3: Identify top features
    top_features = identify_top_features(model, feature_names, n_top=3)

    # Step 4: Generate 1D PDPs
    pdp_data = plot_partial_dependence_1d(
        model, X_test, feature_names, top_features)

    # Step 5: Interpret PDPs
    print_interpretations(pdp_data, top_features)

    # Step 6: Generate 2D PDP
    plot_partial_dependence_2d(model, X_test, feature_names, top_features)

    # Step 7: Summary and recommendations
    print_summary_and_recommendations(model_name, top_features, pdp_data)

    print("\n" + "=" * 100)
    print("✅ PARTIAL DEPENDENCE ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 100)

    print("\n💡 Model Selection Note:")
    print(f"  • Analysis performed on production-ready model: {model_name}")
    print("  • Models with perfect scores were excluded (overfitting concerns)")
    print("  • This ensures PDP insights will be relevant in production")

    print("\nOutputs saved:")
    print(
        f"  • 1D PDPs: {
            Config.GRAPHICS_DIR /
            'partial_dependence_top3.png'}")
    print(
        f"  • 2D PDP:  {
            Config.GRAPHICS_DIR /
            'partial_dependence_interaction.png'}")
    print("\nKey achievements:")
    print("  1. ✓ Model behavior explained through PDP visualizations")
    print("  2. ✓ Feature-target relationships quantified")
    print("  3. ✓ Actionable business insights identified")
    print("  4. ✓ Risk thresholds and patterns revealed")
    print("\nNext steps:")
    print("  • Validate findings with domain experts")
    print("  • Implement recommended operational policies")
    print("  • Consider SHAP analysis for instance-level explanations")
    print("  • Update analysis when model is retrained")


# ================================================================================
# EXECUTION
# ================================================================================

if __name__ == "__main__":
    main()
