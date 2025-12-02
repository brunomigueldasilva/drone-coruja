#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FLIGHT INCIDENT PREDICTION - ENSEMBLE MODEL TRAINING
================================================================================

Purpose: Train and compare ensemble methods (Bagging vs Boosting)

This script trains 4 models to understand ensemble learning:
1. Decision Tree (baseline) - demonstrates overfitting
2. Random Forest (Bagging) - reduces variance through averaging
3. XGBoost (Boosting) - reduces bias through sequential correction
4. Gradient Boosting (Boosting alternative) - classic boosting approach

Key Learning Objectives:
- Understand bias-variance trade-off
- Compare Bagging (parallel) vs Boosting (sequential)
- Handle imbalanced data with class weights
- Measure and compare training performance
- Save models for production deployment

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
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


# Configuration
class Config:
    """Configuration parameters for model training pipeline."""
    # Input paths
    DATA_DIR = Path('outputs') / 'data_processed'
    X_TRAIN_FILE = DATA_DIR / 'X_train_scaled.csv'
    X_TEST_FILE = DATA_DIR / 'X_test_scaled.csv'
    Y_TRAIN_FILE = DATA_DIR / 'y_train.csv'
    Y_TEST_FILE = DATA_DIR / 'y_test.csv'

    # Output paths
    MODELS_DIR = Path('outputs') / 'models'
    PREDICTIONS_DIR = Path('outputs') / 'predictions'

    # Random state for reproducibility
    RANDOM_STATE = 42

    # Model hyperparameters
    # Decision Tree: No constraints (demonstrates overfitting)
    DT_PARAMS = {
        'random_state': RANDOM_STATE
        # No max_depth: tree will grow until pure leaves (overfitting!)
    }

    # Random Forest: Constrained trees with bagging
    RF_PARAMS = {
        'n_estimators': 100,        # 100 trees
        'max_depth': 10,            # Limit depth to prevent individual tree overfitting
        'oob_score': True,          # Out-of-bag score (internal validation)
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced',  # Handle imbalanced classes
        'n_jobs': -1                # Use all CPU cores
    }

    # XGBoost: Boosting with gradient optimization
    XGB_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,       # Step size shrinkage
        # Shallower than RF (boosting uses simpler trees)
        'max_depth': 5,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',   # Evaluation metric
        'use_label_encoder': False,
        'n_jobs': -1
        # scale_pos_weight: calculated dynamically based on class imbalance
    }

    # Gradient Boosting: Classic boosting approach
    GB_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': RANDOM_STATE
        # No built-in class_weight, will handle through sample_weight if needed
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


def save_pickle(obj: Any, filepath: Path) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to serialize
        filepath: Output file path
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  ✓ Saved: {filepath.name}")


def calculate_class_weight(y_train: pd.Series) -> float:
    """
    Calculate XGBoost scale_pos_weight parameter.

    Args:
        y_train: Training labels

    Returns:
        Ratio of negative to positive samples

    CRITICAL CONCEPT: HANDLING CLASS IMBALANCE
    ------------------------------------------
    Imbalanced datasets (e.g., 90% class 0, 10% class 1) pose challenges:
    - Model can achieve 90% accuracy by always predicting class 0
    - Minority class (incidents) is what we care about most
    - Need to give more importance to minority class

    Two main strategies:

    1. class_weight='balanced' (Scikit-learn):
       - Automatically adjusts weights inversely proportional to class frequencies
       - Formula: weight[i] = n_samples / (n_classes × n_samples[i])
       - Example: If class 0=900, class 1=100, total=1000
         → weight[0] = 1000/(2×900) = 0.556
         → weight[1] = 1000/(2×100) = 5.0
       - Effect: Each minority sample counts 9× more than majority sample

    2. scale_pos_weight (XGBoost):
       - Ratio of negative to positive samples
       - Formula: scale_pos_weight = count(negative) / count(positive)
       - Example: 900 negative / 100 positive = 9.0
       - Effect: Loss from positive samples multiplied by this factor

    Why this helps:
    - Model penalized more for misclassifying minority class
    - Encourages learning patterns in minority class
    - Balances precision-recall trade-off
    """
    counts = y_train.value_counts()
    scale_pos_weight = counts[0] / counts[1]

    print("  Class distribution:")
    print(f"    Class 0 (negative): {counts[0]:,} samples")
    print(f"    Class 1 (positive): {counts[1]:,} samples")
    print(f"    Imbalance ratio: {counts[0] / counts[1]:.2f}:1")
    print(f"\n  Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"    → Positive samples weighted {scale_pos_weight:.0f}× more than negative")

    return scale_pos_weight


# ================================================================================
# SECTION 3: DATA LOADING
# ================================================================================


def load_processed_data() -> Tuple[pd.DataFrame,
                                   pd.DataFrame, pd.Series, pd.Series]:
    """
    Load preprocessed training and test data.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Raises:
        FileNotFoundError: If any required file is missing
    """
    print_header("STEP 1: LOADING PREPROCESSED DATA")

    # Check all files exist
    required_files = [
        Config.X_TRAIN_FILE,
        Config.X_TEST_FILE,
        Config.Y_TRAIN_FILE,
        Config.Y_TEST_FILE
    ]

    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n" +
            "\n".join(f"  - {f}" for f in missing_files) +
            "\n\nPlease run 02_preprocessing.py first!"
        )

    # Load data
    X_train = pd.read_csv(Config.X_TRAIN_FILE)
    X_test = pd.read_csv(Config.X_TEST_FILE)
    y_train = pd.read_csv(Config.Y_TRAIN_FILE).squeeze()  # Convert to Series
    y_test = pd.read_csv(Config.Y_TEST_FILE).squeeze()

    print("✓ Data loaded successfully")
    print("\n  Training set:")
    print(f"    X_train: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"    y_train: {len(y_train):,} labels")

    print("\n  Test set:")
    print(f"    X_test: {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
    print(f"    y_test: {len(y_test):,} labels")

    print(f"\n  Features: {list(X_train.columns)}")

    return X_train, X_test, y_train, y_test


# ================================================================================
# SECTION 4: MODEL 1 - DECISION TREE (BASELINE)
# ================================================================================


def train_decision_tree(X_train: pd.DataFrame,
                        y_train: pd.Series) -> Dict[str, Any]:
    """
    Train baseline Decision Tree (no constraints).

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Dictionary with model, training time, and metadata

    CRITICAL CONCEPT: OVERFITTING IN DECISION TREES
    ------------------------------------------------
    A single unconstrained Decision Tree is prone to OVERFITTING:

    What is overfitting?
    - Model learns the training data TOO well
    - Captures noise and random fluctuations as if they were patterns
    - High accuracy on training data
    - Poor accuracy on test data (doesn't generalize)

    Why Decision Trees overfit:
    1. HIGH VARIANCE: Small changes in data lead to very different trees
    2. MEMORIZATION: Tree can grow until each leaf has one sample
    3. NO AVERAGING: Single tree's mistakes aren't corrected

    Example:
    - Training accuracy: 100% (perfect!)
    - Test accuracy: 75% (poor generalization)
    - Difference = overfitting

    Visual analogy:
    - Imagine fitting a curve through points
    - Overfit: Curve passes through EVERY point (wiggly, complex)
    - Good fit: Smooth curve captures general trend (simpler)

    Why we train this model:
    - Establishes baseline performance
    - Demonstrates the overfitting problem
    - Shows what ensemble methods improve upon

    In practice:
    - Never use unconstrained Decision Tree in production
    - Always use ensemble methods (RF, XGBoost) instead
    - Or constrain single tree with max_depth, min_samples_split, etc.
    """
    print_header("STEP 2A: TRAINING DECISION TREE (BASELINE)")

    print("Model configuration:")
    print("  Type: Single Decision Tree")
    print("  Constraints: None (tree will grow until pure leaves)")
    print("  ⚠️  WARNING: This model will likely OVERFIT")
    print("\n  Why no constraints?")
    print("    → To demonstrate overfitting problem")
    print("    → Shows what ensemble methods solve")
    print("    → Establishes baseline for comparison")

    # Initialize model
    model = DecisionTreeClassifier(**Config.DT_PARAMS)

    # Train and time
    print("\n  Training Decision Tree...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Get model complexity metrics
    n_leaves = model.get_n_leaves()
    max_depth = model.get_depth()

    print(f"  ✓ Training completed in {training_time:.3f} seconds")
    print("\n  Model complexity:")
    print(f"    Max depth reached: {max_depth}")
    print(f"    Number of leaves: {n_leaves}")
    print("    → More leaves = More complex = Higher overfitting risk")

    return {
        'model': model,
        'training_time': training_time,
        'n_leaves': n_leaves,
        'max_depth': max_depth,
        'name': 'DecisionTree'
    }


# ================================================================================
# SECTION 5: MODEL 2 - RANDOM FOREST (BAGGING)
# ================================================================================


def train_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series) -> Dict[str, Any]:
    """
    Train Random Forest using Bagging.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Dictionary with model, training time, OOB score, and metadata

    CRITICAL CONCEPT: BAGGING (BOOTSTRAP AGGREGATING)
    -------------------------------------------------
    Random Forest uses BAGGING to reduce variance and prevent overfitting.

    How Bagging works:
    1. Create multiple bootstrap samples (random sampling with replacement)
    2. Train one tree on each bootstrap sample
    3. Average predictions from all trees

    Example with 3 trees:
    - Original data: 1000 samples
    - Bootstrap 1: Random 1000 samples (some repeated, ~63% unique)
    - Bootstrap 2: Different random 1000 samples
    - Bootstrap 3: Different random 1000 samples
    - Train tree on each → 3 different trees
    - Final prediction: Average of 3 trees' predictions

    Why this reduces overfitting (lowers variance):
    - Each tree sees slightly different data
    - Each tree makes different mistakes
    - Averaging cancels out random errors
    - What remains: consistent patterns (signal, not noise)

    Mathematical intuition:
    - Variance of average = Variance of individual / n_trees
    - 100 trees → Variance reduced by factor of ~10
    - Like asking 100 experts vs 1 expert

    Additional Random Forest features:
    1. Feature randomness: Each split considers random subset of features
       → Further decorrelates trees
       → Even better variance reduction

    2. Out-of-Bag (OOB) Score:
       - Each tree only sees ~63% of data (bootstrap sample)
       - Remaining ~37% is "out-of-bag"
       - Use OOB samples for validation (free cross-validation!)
       - Reliable estimate of generalization without separate validation set

    3. class_weight='balanced':
       - Handles imbalanced data
       - Gives more weight to minority class
       - Prevents model from ignoring incidents

    Bias-Variance Trade-off:
    - Single tree: Low bias, HIGH variance (overfits)
    - Random Forest: Low bias, LOWER variance (generalizes better)
    - Result: Better test performance
    """
    print_header("STEP 2B: TRAINING RANDOM FOREST (BAGGING)")

    print("Model configuration:")
    print("  Type: Random Forest (Ensemble of Decision Trees)")
    print("  Method: Bagging (Bootstrap Aggregating)")
    print("  Number of trees: 100")
    print("  Max depth per tree: 10 (prevents individual tree overfitting)")
    print("  Class weight: balanced (handles imbalanced data)")
    print("  OOB score: True (internal validation)")

    print("\n  How Random Forest reduces overfitting:")
    print("    1. Trains 100 trees on different bootstrap samples")
    print("    2. Each tree makes different mistakes")
    print("    3. Averaging cancels out random errors")
    print("    4. What remains: true patterns (signal)")
    print("    → Result: Lower variance = Better generalization")

    # Initialize model
    model = RandomForestClassifier(**Config.RF_PARAMS)

    # Train and time
    print("\n  Training Random Forest...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Get OOB score
    oob_score = model.oob_score_

    print(f"  ✓ Training completed in {training_time:.3f} seconds")
    print("\n  Model statistics:")
    print(f"    Number of trees: {model.n_estimators}")
    print(f"    Features per split: {model.max_features}")
    print(f"    OOB Score: {oob_score:.4f}")
    print("\n  About OOB Score:")
    print("    → Out-of-Bag accuracy on unseen samples")
    print("    → Like cross-validation, but free!")
    print("    → Each tree validated on ~37% of data it didn't see")
    print("    → Reliable estimate of generalization performance")

    return {
        'model': model,
        'training_time': training_time,
        'n_estimators': model.n_estimators,
        'oob_score': oob_score,
        'name': 'RandomForest'
    }


# ================================================================================
# SECTION 6: MODEL 3 - XGBOOST (BOOSTING)
# ================================================================================


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Train XGBoost using Gradient Boosting.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Dictionary with model, training time, and metadata

    CRITICAL CONCEPT: BOOSTING (SEQUENTIAL ERROR CORRECTION)
    --------------------------------------------------------
    XGBoost uses BOOSTING to reduce bias and improve accuracy.

    How Boosting works (simplified):
    1. Train tree 1 on original data
    2. Find samples that tree 1 got wrong
    3. Train tree 2 focusing on those mistakes
    4. Train tree 3 focusing on remaining mistakes
    5. Continue for 100 trees
    6. Final prediction: Weighted sum of all trees

    Key difference from Bagging (Random Forest):
    - Bagging: Trees trained in PARALLEL (independent)
    - Boosting: Trees trained SEQUENTIALLY (dependent)
    - Bagging: Reduces variance (averaging)
    - Boosting: Reduces bias (error correction)

    Detailed XGBoost process:
    1. Start with initial prediction (e.g., mean)
    2. Calculate errors (residuals)
    3. Train tree to predict these errors
    4. Add tree's predictions to model (with learning rate)
    5. Recalculate errors
    6. Repeat: Each tree corrects previous trees' mistakes

    Example (intuitive):
    - Tree 1: "I think sample A is class 0" (wrong, actual is 1)
    - Tree 2: "Let me focus on sample A... I think it's class 1"
    - Tree 3: "Good! Now let me correct other mistakes..."
    - Final: Weighted vote of all trees (usually correct)

    XGBoost advantages:
    1. Gradient-based optimization (mathematically optimal)
    2. Built-in regularization (prevents overfitting)
    3. Handles missing values automatically
    4. Very fast (optimized C++ implementation)
    5. Often wins Kaggle competitions

    Hyperparameters:
    - learning_rate (0.1): How much each tree contributes
      → Lower = More trees needed, but better generalization
    - max_depth (5): Shallower than RF
      → Boosting uses many simple trees (weak learners)
      → RF uses fewer, more complex trees
    - scale_pos_weight: Handles class imbalance
      → Like class_weight='balanced', but for XGBoost

    Bias-Variance Trade-off:
    - Single tree: LOW bias, high variance
    - Random Forest: LOW bias, LOWER variance
    - XGBoost: LOWER bias, low variance
    - Result: Often best test performance

    Overfitting risk:
    - Boosting CAN overfit if too many trees
    - Use early stopping in practice
    - Monitor validation error
    - Stop when validation error stops improving
    """
    print_header("STEP 2C: TRAINING XGBOOST (BOOSTING)")

    print("Model configuration:")
    print("  Type: XGBoost (Extreme Gradient Boosting)")
    print("  Method: Boosting (Sequential Error Correction)")
    print("  Number of trees: 100")
    print("  Learning rate: 0.1 (step size for each tree)")
    print("  Max depth: 5 (shallower than RF - boosting uses weak learners)")

    # Calculate scale_pos_weight for imbalanced data
    print("\n  Calculating class imbalance adjustment:")
    scale_pos_weight = calculate_class_weight(y_train)

    print("\n  How XGBoost reduces bias:")
    print("    1. Tree 1 makes predictions")
    print("    2. Calculate errors (residuals)")
    print("    3. Tree 2 trained to predict these errors")
    print("    4. Add Tree 2's corrections to model")
    print("    5. Repeat 100 times")
    print("    → Each tree corrects previous trees' mistakes")
    print("    → Final model = Sum of all corrections")

    # Update parameters with calculated weight
    xgb_params = Config.XGB_PARAMS.copy()
    xgb_params['scale_pos_weight'] = scale_pos_weight

    # Initialize model
    model = XGBClassifier(**xgb_params)

    # Train and time
    print("\n  Training XGBoost...")
    start_time = time.time()
    model.fit(X_train, y_train, verbose=False)
    training_time = time.time() - start_time

    print(f"  ✓ Training completed in {training_time:.3f} seconds")
    print("\n  Model statistics:")
    print(f"    Number of boosting rounds: {model.n_estimators}")
    print(f"    Learning rate: {model.learning_rate}")
    print(f"    Max depth: {model.max_depth}")

    return {
        'model': model,
        'training_time': training_time,
        'n_estimators': model.n_estimators,
        'learning_rate': model.learning_rate,
        'name': 'XGBoost'
    }


# ================================================================================
# SECTION 7: MODEL 4 - GRADIENT BOOSTING (BOOSTING ALTERNATIVE)
# ================================================================================


def train_gradient_boosting(X_train: pd.DataFrame,
                            y_train: pd.Series) -> Dict[str, Any]:
    """
    Train Gradient Boosting (Scikit-learn implementation).

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Dictionary with model, training time, and metadata

    GRADIENT BOOSTING vs XGBOOST
    -----------------------------
    Both are boosting methods, but with differences:

    Similarities:
    - Sequential tree building
    - Each tree corrects previous errors
    - Gradient-based optimization
    - Reduce bias through boosting

    Differences:

    1. Implementation:
       - GradientBoosting: Pure Python (sklearn)
       - XGBoost: Optimized C++ (much faster)

    2. Speed:
       - GradientBoosting: Slower training
       - XGBoost: 10-100× faster

    3. Memory:
       - GradientBoosting: More memory efficient
       - XGBoost: Uses more memory for speed

    4. Regularization:
       - GradientBoosting: L2 only
       - XGBoost: L1 + L2 + more advanced

    5. Missing values:
       - GradientBoosting: Requires imputation
       - XGBoost: Handles automatically

    6. Class imbalance:
       - GradientBoosting: Manual sample_weight
       - XGBoost: Built-in scale_pos_weight

    When to use each:
    - XGBoost: Usually better (faster, more features)
    - GradientBoosting: If no XGBoost available, memory constrained
    - In practice: XGBoost has become the standard

    We train both to compare performance and understand differences.
    """
    print_header("STEP 2D: TRAINING GRADIENT BOOSTING (BOOSTING ALTERNATIVE)")

    print("Model configuration:")
    print("  Type: Gradient Boosting (Scikit-learn)")
    print("  Method: Boosting (same principle as XGBoost)")
    print("  Number of trees: 100")
    print("  Learning rate: 0.1")
    print("  Max depth: 5")

    print("\n  Gradient Boosting vs XGBoost:")
    print("    Similarities:")
    print("      • Both use sequential tree building")
    print("      • Both reduce bias through boosting")
    print("      • Both use gradient-based optimization")
    print("    Differences:")
    print("      • GB: Pure Python (sklearn) → Slower")
    print("      • XGB: Optimized C++ → 10-100× faster")
    print("      • XGB: More advanced regularization")
    print("      • XGB: Better handling of missing values")

    # Initialize model
    model = GradientBoostingClassifier(**Config.GB_PARAMS)

    # Train and time
    print("\n  Training Gradient Boosting...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"  ✓ Training completed in {training_time:.3f} seconds")
    print("\n  Model statistics:")
    print(f"    Number of boosting stages: {model.n_estimators}")
    print(f"    Learning rate: {model.learning_rate}")
    print(f"    Max depth: {model.max_depth}")

    return {
        'model': model,
        'training_time': training_time,
        'n_estimators': model.n_estimators,
        'learning_rate': model.learning_rate,
        'name': 'GradientBoosting'
    }


# ================================================================================
# SECTION 8: GENERATE PREDICTIONS
# ================================================================================


def generate_predictions(
    models_dict: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate predictions for all models.

    Args:
        models_dict: Dictionary of trained models
        X_test: Test features

    Returns:
        Dictionary with predictions for each model

    Predictions generated:
    - Class labels (0 or 1)
    - Probabilities for class 0
    - Probabilities for class 1

    Why probabilities matter:
    - Class labels: Binary decision (0 or 1)
    - Probabilities: Confidence level (0.0 to 1.0)

    Example:
    - Model predicts class 1 with probability 0.95 → Very confident
    - Model predicts class 1 with probability 0.51 → Barely confident

    Use cases for probabilities:
    1. Threshold tuning: Adjust decision boundary
       - Default: 0.5 (predict 1 if prob > 0.5)
       - Can adjust: 0.3 (more sensitive to incidents)

    2. Risk scoring: Rank samples by risk
       - Highest probability → Highest risk
       - Useful for prioritization

    3. Calibration: Assess confidence quality
       - Are 90% confidence predictions correct 90% of the time?

    4. ROC-AUC calculation: Requires probabilities
       - Measures performance across all thresholds
    """
    print_header("STEP 3: GENERATING PREDICTIONS")

    predictions = {}

    for model_name, model_info in models_dict.items():
        print(f"\n  Generating predictions for {model_name}...")

        model = model_info['model']

        # Predict class labels (0 or 1)
        y_pred = model.predict(X_test)

        # Predict probabilities [prob_class_0, prob_class_1]
        y_pred_proba = model.predict_proba(X_test)

        predictions[model_name] = {
            'labels': y_pred,
            'probabilities': y_pred_proba,
            'proba_class_0': y_pred_proba[:, 0],
            'proba_class_1': y_pred_proba[:, 1]
        }

        print("    ✓ Predictions generated")
        print(f"      Classes: {len(y_pred)} predictions")
        print(f"      Probabilities shape: {y_pred_proba.shape}")

    print("\n✓ All predictions generated successfully")

    return predictions


# ================================================================================
# SECTION 9: SAVE MODELS AND PREDICTIONS
# ================================================================================


def save_models(models_dict: Dict[str, Dict[str, Any]]) -> None:
    """
    Save all trained models to pickle files.

    Args:
        models_dict: Dictionary of trained models
    """
    print_header("STEP 4: SAVING TRAINED MODELS")

    # Create models directory
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving models to: {Config.MODELS_DIR}/")

    for model_name, model_info in models_dict.items():
        model = model_info['model']
        filepath = Config.MODELS_DIR / f'{model_name.lower()}_model.pkl'
        save_pickle(model, filepath)

    print("\n✓ All models saved successfully")


def save_predictions(
    predictions: Dict[str, Dict[str, np.ndarray]],
    y_test: pd.Series
) -> None:
    """
    Save predictions to CSV files.

    Args:
        predictions: Dictionary of predictions for each model
        y_test: True test labels
    """
    print_header("STEP 5: SAVING PREDICTIONS")

    # Create predictions directory
    Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving predictions to: {Config.PREDICTIONS_DIR}/")

    for model_name, pred_dict in predictions.items():
        # Create DataFrame with all prediction info
        df = pd.DataFrame({
            'true_label': y_test.values,
            'predicted_label': pred_dict['labels'],
            'probability_class_0': pred_dict['proba_class_0'],
            'probability_class_1': pred_dict['proba_class_1']
        })

        filepath = Config.PREDICTIONS_DIR / f'{model_name.lower()}_predictions.csv'
        df.to_csv(filepath, index=False)
        print(f"  ✓ Saved: {filepath.name}")

    print("\n✓ All predictions saved successfully")


# ================================================================================
# SECTION 10: TRAINING SUMMARY
# ================================================================================


def print_training_summary(models_dict: Dict[str, Dict[str, Any]]) -> None:
    """
    Print comprehensive training summary table.

    Args:
        models_dict: Dictionary of trained models with metadata
    """
    print_header("TRAINING SUMMARY", "=")

    print("\n📊 MODEL COMPARISON TABLE")
    print("-" * 100)
    print(f"{'Model':<20} {'Type':<15} {'Trees':<10} {'Training Time':<15} {'Special Features':<30}")
    print("-" * 100)

    # Decision Tree
    dt_info = models_dict['DecisionTree']
    print(f"{'Decision Tree':<20} {'Single Tree':<15} {'1':<10} "
          f"{dt_info['training_time']:.3f}s        "
          f"Max depth: {dt_info['max_depth']}, Leaves: {dt_info['n_leaves']}")

    # Random Forest
    rf_info = models_dict['RandomForest']
    print(f"{'Random Forest':<20} {'Bagging':<15} {rf_info['n_estimators']:<10} "
          f"{rf_info['training_time']:.3f}s        "
          f"OOB Score: {rf_info['oob_score']:.4f}")

    # XGBoost
    xgb_info = models_dict['XGBoost']
    print(f"{'XGBoost':<20} {'Boosting':<15} {xgb_info['n_estimators']:<10} "
          f"{xgb_info['training_time']:.3f}s        "
          f"LR: {xgb_info['learning_rate']}")

    # Gradient Boosting
    gb_info = models_dict['GradientBoosting']
    print(f"{'Gradient Boosting':<20} {'Boosting':<15} {gb_info['n_estimators']:<10} "
          f"{gb_info['training_time']:.3f}s        "
          f"LR: {gb_info['learning_rate']}")

    print("-" * 100)

    # Key insights
    print("\n🎯 KEY INSIGHTS")
    print("-" * 100)

    print("\n1. OVERFITTING (Decision Tree):")
    print(f"   • Single tree with {dt_info['n_leaves']} leaves")
    print(f"   • Max depth: {dt_info['max_depth']}")
    print("   → High variance: Likely overfits training data")
    print("   → Use as baseline to show what ensembles improve")

    print("\n2. BAGGING (Random Forest):")
    print(f"   • {rf_info['n_estimators']} trees trained independently")
    print(f"   • OOB Score: {rf_info['oob_score']:.4f} (internal validation)")
    print("   → Reduces variance through averaging")
    print("   → Each tree makes different mistakes")
    print("   → Averaging cancels out random errors")

    print("\n3. BOOSTING (XGBoost & Gradient Boosting):")
    print(f"   • {xgb_info['n_estimators']} trees trained sequentially")
    print("   • Each tree corrects previous errors")
    print("   → Reduces bias through error correction")
    print("   → Often achieves best performance")
    print("   → XGBoost typically faster than Gradient Boosting")

    print("\n4. TRAINING TIME:")
    print(f"   • Fastest: Decision Tree ({dt_info['training_time']:.3f}s)")
    print(f"   • XGBoost: {xgb_info['training_time']:.3f}s (optimized C++)")
    print(f"   • Random Forest: {rf_info['training_time']:.3f}s (parallel)")
    print(f"   • Gradient Boosting: {gb_info['training_time']:.3f}s (sequential Python)")

    # Bias-Variance summary
    print("\n📈 BIAS-VARIANCE TRADE-OFF")
    print("-" * 100)
    print("  Model                 Bias        Variance    Typical Result")
    print("  " + "-" * 96)
    print("  Decision Tree         Low         HIGH        Overfits (high train, low test)")
    print("  Random Forest         Low         LOWER       Better generalization")
    print("  XGBoost/GB            LOWER       Low         Often best performance")
    print("  " + "-" * 96)

    print("\n💡 NEXT STEPS")
    print("-" * 100)
    print("  1. Evaluate models on test set (04_evaluate_models.py)")
    print("  2. Compare metrics: Accuracy, Precision, Recall, F1, ROC-AUC")
    print("  3. Analyze confusion matrices")
    print("  4. Choose best model based on business requirements")
    print("  5. Consider hyperparameter tuning for best model")


# ================================================================================
# SECTION 11: MAIN FUNCTION
# ================================================================================


def main() -> None:
    """
    Main training pipeline.

    Workflow:
    1. Load preprocessed data
    2. Train 4 models (Decision Tree, Random Forest, XGBoost, Gradient Boosting)
    3. Generate predictions on test set
    4. Save models and predictions
    5. Print training summary
    """
    print("=" * 100)
    print("FLIGHT INCIDENT PREDICTION - ENSEMBLE MODEL TRAINING")
    print("=" * 100)
    print("\nTraining 4 models to understand ensemble learning:")
    print("  1. Decision Tree      → Baseline (demonstrates overfitting)")
    print("  2. Random Forest      → Bagging (reduces variance)")
    print("  3. XGBoost            → Boosting (reduces bias)")
    print("  4. Gradient Boosting  → Boosting alternative")

    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Step 2: Train all models
    models_dict = {}

    # 2A: Decision Tree
    models_dict['DecisionTree'] = train_decision_tree(X_train, y_train)

    # 2B: Random Forest
    models_dict['RandomForest'] = train_random_forest(X_train, y_train)

    # 2C: XGBoost
    models_dict['XGBoost'] = train_xgboost(X_train, y_train)

    # 2D: Gradient Boosting
    models_dict['GradientBoosting'] = train_gradient_boosting(X_train, y_train)

    # Step 3: Generate predictions
    predictions = generate_predictions(models_dict, X_test)

    # Step 4: Save models
    save_models(models_dict)

    # Step 5: Save predictions
    save_predictions(predictions, y_test)

    # Step 6: Print summary
    print_training_summary(models_dict)

    print("\n" + "=" * 100)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 100)
    print("\nOutputs saved:")
    print(f"  • Models: {Config.MODELS_DIR}/")
    print(f"  • Predictions: {Config.PREDICTIONS_DIR}/")
    print("\nReady for model evaluation (04_evaluate_models.py)")


# ================================================================================
# EXECUTION
# ================================================================================

if __name__ == "__main__":
    main()
