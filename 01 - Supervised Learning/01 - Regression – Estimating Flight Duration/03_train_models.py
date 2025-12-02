#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLIGHT TELEMETRY - MODEL TRAINING
==============================================================================

Purpose: Train multiple regression models on preprocessed flight telemetry data

This script:
1. Loads preprocessed datasets and artifacts from preprocessing stage
2. Trains five different regression models:
   a) Simple Linear Regression (single feature: distancia_planeada only)
   b) Multiple Linear Regression (all features)
   c) Ridge Regression (L2 regularization, alpha=1.0)
   d) Lasso Regression (L1 regularization, alpha=0.001)
   e) Polynomial Regression (degree=2 on numeric features + categorical features)
3. Records training times for each model
4. Generates predictions on test set for each model
5. Persists trained models and predictions for evaluation phase

Model Characteristics:
- Simple Linear: Baseline model, most interpretable, uses only distance feature
- Multiple Linear: Uses all available features, risk of multicollinearity
- Ridge: Shrinks coefficients toward zero, handles multicollinearity well
- Lasso: Can zero out coefficients, performs automatic feature selection
- Polynomial: Captures non-linear relationships, risk of overfitting

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path
from typing import Dict, Any, Tuple
import time

import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore')


class Config:
    """Model training configuration parameters."""
    # Directories
    OUTPUT_DIR = Path("outputs")
    ARTIFACTS_DIR = OUTPUT_DIR / "models"
    TABLES_DIR = OUTPUT_DIR / "results"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

    # Input files
    PREPROCESSOR_FILE = ARTIFACTS_DIR / "preprocessor.pkl"
    X_TRAIN_FILE = ARTIFACTS_DIR / "X_train.pkl"
    X_TEST_FILE = ARTIFACTS_DIR / "X_test.pkl"
    Y_TRAIN_FILE = ARTIFACTS_DIR / "y_train.pkl"
    Y_TEST_FILE = ARTIFACTS_DIR / "y_test.pkl"
    FEATURE_NAMES_FILE = TABLES_DIR / "feature_names_after_oh.csv"

    # Output files
    TRAINING_TIMES_FILE = TABLES_DIR / "training_times.csv"

    # Model parameters
    RANDOM_STATE = 42
    RIDGE_ALPHA = 1.0
    LASSO_ALPHA = 0.001
    POLY_DEGREE = 2


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    separator = char * 120
    print(f"\n{separator}")
    print(title)
    print(separator)


def save_model(model: Any, filename: str) -> None:
    """
    Save model to pickle file.

    Args:
        model: Trained model object
        filename: Name of the pickle file (without path)
    """
    filepath = Config.ARTIFACTS_DIR / filename
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {filepath}")


def save_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str) -> None:
    """
    Save predictions to CSV file.

    Args:
        y_true: True values
        y_pred: Predicted values
        filename: Name of the CSV file (without path)
    """
    preds_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    filepath = Config.PREDICTIONS_DIR / filename
    preds_df.to_csv(filepath, index=False)
    print(f"✓ Predictions saved to: {filepath}")


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load preprocessed training and test data.

    Returns:
        Tuple containing:
        - X_train: Training features (transformed)
        - X_test: Test features (transformed)
        - y_train: Training target
        - y_test: Test target
        - feature_names: List of feature names after transformation
    """
    print_section("[SECTION 1] LOAD PREPROCESSED ARTIFACTS")

    # Load preprocessor
    print("Loading preprocessor...")
    with open(Config.PREPROCESSOR_FILE, 'rb') as f:
        pickle.load(f)
    print(f"✓ Preprocessor loaded from: {Config.PREPROCESSOR_FILE}")
    print()

    # Load training features
    print("Loading training features...")
    with open(Config.X_TRAIN_FILE, 'rb') as f:
        X_train = pickle.load(f)
    print(f"✓ X_train loaded from: {Config.X_TRAIN_FILE}")
    print(f"  Shape: {X_train.shape}")
    print()

    # Load test features
    print("Loading test features...")
    with open(Config.X_TEST_FILE, 'rb') as f:
        X_test = pickle.load(f)
    print(f"✓ X_test loaded from: {Config.X_TEST_FILE}")
    print(f"  Shape: {X_test.shape}")
    print()

    # Load training target
    print("Loading training target...")
    with open(Config.Y_TRAIN_FILE, 'rb') as f:
        y_train = pickle.load(f)
    print(f"✓ y_train loaded from: {Config.Y_TRAIN_FILE}")
    print(f"  Shape: {y_train.shape}")
    print()

    # Load test target
    print("Loading test target...")
    with open(Config.Y_TEST_FILE, 'rb') as f:
        y_test = pickle.load(f)
    print(f"✓ y_test loaded from: {Config.Y_TEST_FILE}")
    print(f"  Shape: {y_test.shape}")
    print()

    # Load feature names
    print("Loading feature names after preprocessing...")
    feature_names_df = pd.read_csv(Config.FEATURE_NAMES_FILE)
    feature_names = feature_names_df['Feature_Name'].tolist()
    print(f"✓ Feature names loaded from: {Config.FEATURE_NAMES_FILE}")
    print(f"  Number of features: {len(feature_names)}")
    print()

    print("Feature names after preprocessing:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    print()

    return X_train, X_test, y_train, y_test, feature_names


# ==============================================================================
# SECTION 4: SIMPLE LINEAR REGRESSION
# ==============================================================================


def train_simple_linear_model(X_train: np.ndarray,
                              X_test: np.ndarray,
                              y_train: np.ndarray,
                              y_test: np.ndarray,
                              feature_names: list) -> Dict[str,
                                                           Any]:
    """
    Train simple linear regression with only 'distancia_planeada' feature.

    Args:
        X_train: Training features (all)
        X_test: Test features (all)
        y_train: Training target
        y_test: Test target
        feature_names: List of feature names

    Returns:
        Dictionary with model info and training time
    """
    print_section("[SECTION 2] SIMPLE LINEAR REGRESSION (1 FEATURE)")

    # Find distancia_planeada feature index
    print("Locating 'distancia_planeada' feature in preprocessed data...")
    print()

    distancia_feature_idx = None
    distancia_feature_name = None

    for idx, name in enumerate(feature_names):
        if 'distancia_planeada' in name:
            distancia_feature_idx = idx
            distancia_feature_name = name
            break

    if distancia_feature_idx is None:
        raise ValueError(
            "Could not locate 'distancia_planeada' feature in preprocessed data")

    print("✓ Feature 'distancia_planeada' located:")
    print(f"  Column index: {distancia_feature_idx}")
    print(f"  Feature name: {distancia_feature_name}")
    print()

    # Extract single feature
    X_train_simple = X_train[:, distancia_feature_idx].reshape(-1, 1)
    X_test_simple = X_test[:, distancia_feature_idx].reshape(-1, 1)

    print("Simple linear regression datasets created:")
    print(f"  X_train_simple: {X_train_simple.shape} (single feature)")
    print(f"  X_test_simple:  {X_test_simple.shape} (single feature)")
    print()

    print("RATIONALE FOR SIMPLE LINEAR REGRESSION:")
    print("  • Serves as baseline model for comparison")
    print("  • Most interpretable: duracao_voo = β₀ + β₁ × distancia_planeada")
    print("  • From EDA, 'distancia_planeada' has strong correlation with target")
    print("  • Helps understand isolated effect of distance on flight duration")
    print("  • Any improvement from multiple regression shows value of additional features")
    print()

    # Train model
    print("Training simple linear regression model...")
    model = LinearRegression()
    start_time = time.time()
    model.fit(X_train_simple, y_train)
    training_time = time.time() - start_time

    print(f"✓ Model trained in {training_time:.4f} seconds")
    print()

    # Display parameters
    print("Learned parameters:")
    print(f"  Intercept (β₀): {model.intercept_:.4f}")
    print(f"  Coefficient (β₁): {model.coef_[0]:.4f}")
    print()

    # Generate predictions
    print("Generating predictions on test set...")
    y_pred = model.predict(X_test_simple)
    print(f"✓ Predictions generated: {y_pred.shape}")
    print()

    # Save model and predictions
    save_model(model, "model_linear_simple.pkl")
    save_predictions(y_test, y_pred, "preds_linear_simple.csv")
    print()

    return {
        'Model': 'Simple Linear Regression',
        'Training_Time_Seconds': training_time,
        'Number_of_Features': X_train_simple.shape[1],
        'Number_of_Parameters': len(model.coef_) + 1
    }


# ==============================================================================
# SECTION 5: MULTIPLE LINEAR REGRESSION
# ==============================================================================


def train_multiple_linear_model(X_train: np.ndarray,
                                X_test: np.ndarray,
                                y_train: np.ndarray,
                                y_test: np.ndarray) -> Dict[str,
                                                            Any]:
    """
    Train multiple linear regression with all features.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with model info and training time
    """
    print_section("[SECTION 3] MULTIPLE LINEAR REGRESSION (ALL FEATURES)")

    print(f"Training with all {X_train.shape[1]} features")
    print()

    print("CHARACTERISTICS:")
    print("  • Uses all available features")
    print("  • Assumes linear relationships")
    print("  • Risk of multicollinearity with correlated features")
    print("  • Closed-form solution: β = (X'X)⁻¹X'y")
    print()

    # Train model
    print("Training multiple linear regression model...")
    model = LinearRegression()
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"✓ Model trained in {training_time:.4f} seconds")
    print()

    # Display parameters
    print("Learned parameters:")
    print(f"  Intercept (β₀): {model.intercept_:.4f}")
    print(f"  Number of coefficients: {len(model.coef_)}")
    print()

    # Generate predictions
    print("Generating predictions on test set...")
    y_pred = model.predict(X_test)
    print(f"✓ Predictions generated: {y_pred.shape}")
    print()

    # Save model and predictions
    save_model(model, "model_linear_multiple.pkl")
    save_predictions(y_test, y_pred, "preds_linear_multiple.csv")
    print()

    return {
        'Model': 'Multiple Linear Regression',
        'Training_Time_Seconds': training_time,
        'Number_of_Features': X_train.shape[1],
        'Number_of_Parameters': len(model.coef_) + 1
    }


# ==============================================================================
# SECTION 6: RIDGE REGRESSION
# ==============================================================================


def train_ridge_model(X_train: np.ndarray,
                      X_test: np.ndarray,
                      y_train: np.ndarray,
                      y_test: np.ndarray) -> Dict[str,
                                                  Any]:
    """
    Train Ridge regression with L2 regularization.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with model info and training time
    """
    print_section("[SECTION 4] RIDGE REGRESSION (L2 REGULARIZATION)")

    print(f"Training with alpha={Config.RIDGE_ALPHA}")
    print()

    print("RIDGE REGRESSION CHARACTERISTICS:")
    print("  • L2 regularization: penalizes sum of squared coefficients")
    print("  • Shrinks coefficients toward zero (but never exactly zero)")
    print("  • Handles multicollinearity better than OLS")
    print("  • Trade-off controlled by alpha hyperparameter")
    print(f"  • Alpha={Config.RIDGE_ALPHA}: moderate regularization strength")
    print()

    # Train model
    print("Training Ridge regression model...")
    model = Ridge(alpha=Config.RIDGE_ALPHA, random_state=Config.RANDOM_STATE)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"✓ Model trained in {training_time:.4f} seconds")
    print()

    # Display parameters
    print("Learned parameters:")
    print(f"  Intercept (β₀): {model.intercept_:.4f}")
    print(f"  Number of coefficients: {len(model.coef_)}")
    print()

    # Generate predictions
    print("Generating predictions on test set...")
    y_pred = model.predict(X_test)
    print(f"✓ Predictions generated: {y_pred.shape}")
    print()

    # Save model and predictions
    save_model(model, "model_ridge.pkl")
    save_predictions(y_test, y_pred, "preds_ridge.csv")
    print()

    return {
        'Model': 'Ridge Regression',
        'Training_Time_Seconds': training_time,
        'Number_of_Features': X_train.shape[1],
        'Number_of_Parameters': len(model.coef_) + 1
    }


# ==============================================================================
# SECTION 7: LASSO REGRESSION
# ==============================================================================


def train_lasso_model(X_train: np.ndarray,
                      X_test: np.ndarray,
                      y_train: np.ndarray,
                      y_test: np.ndarray) -> Dict[str,
                                                  Any]:
    """
    Train Lasso regression with L1 regularization.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with model info and training time
    """
    print_section("[SECTION 5] LASSO REGRESSION (L1 REGULARIZATION)")

    print(f"Training with alpha={Config.LASSO_ALPHA}")
    print()

    print("LASSO REGRESSION CHARACTERISTICS:")
    print("  • L1 regularization: penalizes sum of absolute coefficients")
    print("  • Can set coefficients EXACTLY to zero (feature selection)")
    print("  • Produces sparse models (automatic feature selection)")
    print("  • Useful when many features are irrelevant")
    print(
        f"  • Alpha={
            Config.LASSO_ALPHA}: weak regularization (keep most features)")
    print()

    # Train model
    print("Training Lasso regression model...")
    model = Lasso(
        alpha=Config.LASSO_ALPHA,
        random_state=Config.RANDOM_STATE,
        max_iter=10000)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"✓ Model trained in {training_time:.4f} seconds")
    print()

    # Analyze sparsity
    num_nonzero = np.sum(model.coef_ != 0)
    num_zero = np.sum(model.coef_ == 0)

    print("Learned parameters:")
    print(f"  Intercept (β₀): {model.intercept_:.4f}")
    print(f"  Total coefficients: {len(model.coef_)}")
    print(f"  Non-zero coefficients: {num_nonzero}")
    print(f"  Zero coefficients: {num_zero}")
    print(f"  Sparsity: {num_zero / len(model.coef_) * 100:.1f}%")
    print()

    # Generate predictions
    print("Generating predictions on test set...")
    y_pred = model.predict(X_test)
    print(f"✓ Predictions generated: {y_pred.shape}")
    print()

    # Save model and predictions
    save_model(model, "model_lasso.pkl")
    save_predictions(y_test, y_pred, "preds_lasso.csv")
    print()

    return {
        'Model': 'Lasso Regression',
        'Training_Time_Seconds': training_time,
        'Number_of_Features': X_train.shape[1],
        'Number_of_Parameters': num_nonzero + 1
    }


# ==============================================================================
# SECTION 8: POLYNOMIAL REGRESSION
# ==============================================================================


def train_polynomial_model(X_train: np.ndarray,
                           X_test: np.ndarray,
                           y_train: np.ndarray,
                           y_test: np.ndarray,
                           feature_names: list) -> Dict[str,
                                                        Any]:
    """
    Train polynomial regression with degree 2 features.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        feature_names: List of feature names

    Returns:
        Dictionary with model info and training time
    """
    print_section("[SECTION 6] POLYNOMIAL REGRESSION (DEGREE=2)")

    print("Creating polynomial features (degree=2) for numeric features...")
    print()

    # Identify numeric and categorical features
    numeric_feature_names = [
        'distancia_planeada',
        'carga_util_kg',
        'altitude_media_m']
    numeric_indices = []
    categorical_indices = []

    for idx, name in enumerate(feature_names):
        is_numeric = any(
            num_feat in name for num_feat in numeric_feature_names)
        if is_numeric:
            numeric_indices.append(idx)
        else:
            categorical_indices.append(idx)

    print("Identified features:")
    print(f"  Numeric features: {len(numeric_indices)}")
    print(f"  Categorical features: {len(categorical_indices)}")
    print()

    # Extract numeric and categorical features
    X_train_numeric = X_train[:, numeric_indices]
    X_test_numeric = X_test[:, numeric_indices]
    X_train_categorical = X_train[:, categorical_indices]
    X_test_categorical = X_test[:, categorical_indices]

    # Apply polynomial transformation to numeric features only
    print(
        f"Applying PolynomialFeatures(degree={
            Config.POLY_DEGREE}, include_bias=False)...")
    poly_transformer = PolynomialFeatures(
        degree=Config.POLY_DEGREE, include_bias=False)

    X_train_poly_numeric = poly_transformer.fit_transform(X_train_numeric)
    X_test_poly_numeric = poly_transformer.transform(X_test_numeric)

    print(f"  Original numeric features: {X_train_numeric.shape[1]}")
    print(f"  Polynomial numeric features: {X_train_poly_numeric.shape[1]}")
    print()

    # Concatenate polynomial numeric features with categorical features
    X_train_poly = np.hstack([X_train_poly_numeric, X_train_categorical])
    X_test_poly = np.hstack([X_test_poly_numeric, X_test_categorical])

    print("Final polynomial regression datasets:")
    print(f"  X_train_poly: {X_train_poly.shape}")
    print(f"  X_test_poly:  {X_test_poly.shape}")
    print()

    print("POLYNOMIAL REGRESSION FEATURES:")
    print(f"  • Original features: {X_train.shape[1]}")
    print(f"  • Polynomial features: {X_train_poly.shape[1]}")
    print(f"  • Added features: {X_train_poly.shape[1] - X_train.shape[1]}")
    print()

    # Train model
    print("Training polynomial regression model...")
    model = LinearRegression()
    start_time = time.time()
    model.fit(X_train_poly, y_train)
    training_time = time.time() - start_time

    print(f"✓ Model trained in {training_time:.4f} seconds")
    print()

    # Display parameters
    print("Learned parameters:")
    print(f"  Intercept (β₀): {model.intercept_:.4f}")
    print(f"  Number of coefficients: {len(model.coef_)}")
    print()

    # Generate predictions
    print("Generating predictions on test set...")
    y_pred = model.predict(X_test_poly)
    print(f"✓ Predictions generated: {y_pred.shape}")
    print()

    # Save model and transformer
    save_model(model, "model_polynomial_deg2.pkl")
    save_model(poly_transformer, "poly_transformer_deg2.pkl")
    print("  (Polynomial transformer required for transforming new data in deployment)")
    save_predictions(y_test, y_pred, "preds_polynomial_deg2.csv")
    print()

    return {
        'Model': 'Polynomial Regression (degree=2)',
        'Training_Time_Seconds': training_time,
        'Number_of_Features': X_train_poly.shape[1],
        'Number_of_Parameters': len(model.coef_) + 1
    }


# ==============================================================================
# SECTION 9: SAVE TRAINING SUMMARY
# ==============================================================================


def save_training_summary(training_results: list) -> None:
    """
    Save comprehensive training summary to CSV file.

    Args:
        training_results: List of dictionaries with model training info
    """
    print_section("[SECTION 7] TRAINING TIMES SUMMARY")

    # Create DataFrame
    training_times_df = pd.DataFrame(training_results)

    print("Training times comparison:")
    print(training_times_df.to_string(index=False))
    print()

    # Save to CSV
    training_times_df.to_csv(Config.TRAINING_TIMES_FILE, index=False)
    print(f"✓ Training times saved to: {Config.TRAINING_TIMES_FILE}")
    print()

    # Print observations
    print("TRAINING TIME OBSERVATIONS:")
    fastest_model = training_times_df.loc[
        training_times_df['Training_Time_Seconds'].idxmin(), 'Model']
    slowest_model = training_times_df.loc[
        training_times_df['Training_Time_Seconds'].idxmax(), 'Model']

    print(f"  • Fastest: {fastest_model}")
    print(f"  • Slowest: {slowest_model}")
    print()
    print("Notes:")
    print("  • Simple linear is fastest (fewest features)")
    print("  • Lasso may be slower due to iterative coordinate descent algorithm")
    print("  • Polynomial has most features but still fast (OLS has closed-form solution)")
    print("  • For deployment, prediction time is usually more critical than training time")
    print()


# ==============================================================================
# SECTION 10: MAIN EXECUTION
# ==============================================================================


def main():
    """Main training pipeline execution."""
    print("=" * 120)
    print("MODEL TRAINING - FLIGHT TELEMETRY")
    print("=" * 120)
    print()

    try:
        # Set random seed
        np.random.seed(Config.RANDOM_STATE)

        # Create output directories
        Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Directories verified/created: {Config.PREDICTIONS_DIR}")
        print()

        # 1. Load data
        X_train, X_test, y_train, y_test, feature_names = load_data()

        # 2. Train all models
        training_results = []

        # Model 1: Simple Linear
        result = train_simple_linear_model(
            X_train, X_test, y_train, y_test, feature_names)
        training_results.append(result)

        # Model 2: Multiple Linear
        result = train_multiple_linear_model(X_train, X_test, y_train, y_test)
        training_results.append(result)

        # Model 3: Ridge
        result = train_ridge_model(X_train, X_test, y_train, y_test)
        training_results.append(result)

        # Model 4: Lasso
        result = train_lasso_model(X_train, X_test, y_train, y_test)
        training_results.append(result)

        # Model 5: Polynomial
        result = train_polynomial_model(
            X_train, X_test, y_train, y_test, feature_names)
        training_results.append(result)

        # 3. Save training summary
        save_training_summary(training_results)

        # Final summary
        print("\n" + "=" * 120)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 120)
        print()

        print("SUMMARY OF TRAINED MODELS:")
        print()

        for i, result in enumerate(training_results, 1):
            print(f"{i}. {result['Model'].upper()}")
            print(f"   • Features: {result['Number_of_Features']}")
            print(f"   • Parameters: {result['Number_of_Parameters']}")
            print()

        print("=" * 120)
        print("ARTIFACTS SAVED:")
        print(f"  Models:      {Config.ARTIFACTS_DIR}/ (5 .pkl files)")
        print(f"  Predictions: {Config.PREDICTIONS_DIR}/ (5 .csv files)")
        print(f"  Timings:     {Config.TRAINING_TIMES_FILE}")
        print()
        print("NEXT STEPS:")
        print("  1. Evaluate models using MAE and RMSE metrics")
        print("  2. Compare train vs test performance to assess overfitting")
        print("  3. Perform residual analysis to check model assumptions")
        print("  4. Consider hyperparameter tuning (alpha for Ridge/Lasso, degree for Polynomial)")
        print("  5. Explore ensemble methods or non-linear models (Random Forest, XGBoost)")
        print("=" * 120)
        print()

        print("Script completed successfully!")

    except Exception as e:
        print("\n✗ ERROR during training:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
