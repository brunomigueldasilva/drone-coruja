#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLIGHT TELEMETRY - DATA PREPROCESSING
==============================================================================

Purpose: Prepare flight telemetry data for machine learning modeling

This script:
1. Loads raw telemetry data and validates expected structure
2. Implements missing value imputation strategies (median for numeric, mode for categorical)
3. Separates predictors (X) from target variable (y = duracao_voo_min)
4. Encodes categorical variable 'condicao_meteo' with One-Hot Encoding (drop='first')
5. Scales numeric features using StandardScaler
6. Builds preprocessing pipeline with ColumnTransformer to prevent data leakage
7. Performs train/test split (80/20) with shuffling
8. Applies preprocessing (fit on train only, transform on both train and test)
9. Persists preprocessed datasets and pipeline artifacts using pickle

Key Principles:
- All preprocessing steps (imputation, encoding, scaling) are fit ONLY on training data
- This prevents data leakage from test set into training phase
- Pipeline ensures consistent transformations during model deployment

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Preprocessing configuration parameters."""
    # Directories
    INPUT_DIR = Path('inputs')
    OUTPUT_DIR = Path('outputs')
    ARTIFACTS_DIR = OUTPUT_DIR / 'models'
    TABLES_DIR = OUTPUT_DIR / 'results'

    # Input files
    DATA_FILE = INPUT_DIR / 'voos_telemetria.csv'

    # Output files
    PREPROCESSOR_FILE = ARTIFACTS_DIR / 'preprocessor.pkl'
    X_TRAIN_FILE = ARTIFACTS_DIR / 'X_train.pkl'
    X_TEST_FILE = ARTIFACTS_DIR / 'X_test.pkl'
    Y_TRAIN_FILE = ARTIFACTS_DIR / 'y_train.pkl'
    Y_TEST_FILE = ARTIFACTS_DIR / 'y_test.pkl'
    FEATURE_NAMES_FILE = TABLES_DIR / 'feature_names_after_oh.csv'

    # Expected columns
    NUMERIC_FEATURES = [
        'distancia_planeada',
        'carga_util_kg',
        'altitude_media_m']
    CATEGORICAL_FEATURES = ['condicao_meteo']
    TARGET = 'duracao_voo_min'

    # Preprocessing parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 120)
    print(title)
    print(char * 120)


def print_subsection(title: str) -> None:
    """Print formatted subsection header."""
    print("\n" + title)
    print("-" * 120)


# ==============================================================================
# SECTION 3: DATA LOADING AND VALIDATION
# ==============================================================================


def load_and_validate_data() -> pd.DataFrame:
    """
    Load flight telemetry data and validate structure.

    Returns:
        pd.DataFrame: Validated dataset

    Raises:
        ValueError: If required columns are missing
    """
    print_section("1. DATA LOADING AND VALIDATION")

    # Create directories
    Config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    Config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print("✓ Directories created/verified:")
    print(f"  - {Config.ARTIFACTS_DIR}")
    print(f"  - {Config.TABLES_DIR}")
    print()

    # Load dataset
    df = pd.read_csv(Config.DATA_FILE)
    print(
        f"✓ Dataset loaded successfully: {
            df.shape[0]} rows × {
            df.shape[1]} columns")
    print()

    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())
    print()

    # Validate columns
    all_expected = Config.NUMERIC_FEATURES + \
        Config.CATEGORICAL_FEATURES + [Config.TARGET]

    print("Column validation:")
    missing_columns = []
    for col in all_expected:
        if col in df.columns:
            print(f"  [OK] {col}")
        else:
            print(f"  [MISSING] {col}")
            missing_columns.append(col)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    print()

    # Check data types
    print("Data types verification:")
    print(df.dtypes)
    print()

    # Missing values analysis
    print_subsection("Missing values analysis:")

    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df)) * 100,
        'Data_Type': df.dtypes.values
    })

    print(missing_summary.to_string(index=False))
    print()

    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        print(f"Total missing values found: {total_missing}")
    else:
        print("No missing values detected in the dataset.")
    print()

    return df


# ==============================================================================
# SECTION 4: IMPUTATION STRATEGY EXPLANATION
# ==============================================================================


def explain_imputation_strategy() -> None:
    """Display detailed explanation of imputation strategy and data leakage prevention."""
    print_section("2. MISSING VALUE IMPUTATION STRATEGY")

    print("IMPUTATION STRATEGY DEFINITION:")
    print()
    print("1. NUMERIC VARIABLES (distancia_planeada, carga_util_kg, altitude_media_m):")
    print("   Strategy: MEDIAN imputation")
    print("   Rationale:")
    print("     • Median is robust to outliers (as identified in EDA)")
    print("     • For continuous variables, median preserves central tendency better than mean when data is skewed")
    print("     • Prevents extreme values from biasing imputation")
    print()

    print("2. CATEGORICAL VARIABLES (condicao_meteo):")
    print("   Strategy: MOST FREQUENT (mode) imputation")
    print("   Rationale:")
    print("     • Most frequent category is the natural choice for categorical data")
    print("     • Maintains distribution of existing categories")
    print("     • Avoids creating new/invalid category values")
    print()

    print("3. DATA LEAKAGE PREVENTION:")
    print("   ⚠ CRITICAL: Imputation parameters (median, mode) must be calculated ONLY on training data.")
    print()
    print("   Why this matters:")
    print("     • If we calculate imputation values on the full dataset (train + test), we are 'leaking'")
    print("       information from the test set into the training process.")
    print("     • This artificially improves model performance estimates because the model has 'seen' test")
    print("       data statistics.")
    print("     • In production, we won't have access to future data statistics, so our model would")
    print("       perform worse.")
    print()
    print("   Our approach:")
    print("     • We will use sklearn Pipeline + ColumnTransformer")
    print("     • Imputers will be FIT on training data only (learn median/mode from train)")
    print("     • Then TRANSFORM both train and test using those learned values")
    print("     • This simulates real-world deployment where we only know training data statistics")
    print()


# ==============================================================================
# SECTION 5: FEATURE PREPARATION
# ==============================================================================


def prepare_features_and_target(
        df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Separate features from target and identify feature types.

    Args:
        df: Input dataframe

    Returns:
        Tuple containing:
            - X: Feature matrix
            - y: Target variable
            - numeric_features: List of numeric feature names
            - categorical_features: List of categorical feature names
    """
    print_section("3. FEATURE AND TARGET SEPARATION")

    # Define target variable
    y = df[Config.TARGET].copy()

    # Define feature matrix
    feature_cols = [col for col in df.columns if col != Config.TARGET]
    X = df[feature_cols].copy()

    print(f"Target variable (y): {Config.TARGET}")
    print(f"  Shape: {y.shape}")
    print(f"  Type: {y.dtype}")
    print()

    print(f"Feature matrix (X): {len(feature_cols)} features")
    print(f"  Shape: {X.shape}")
    print(f"  Features: {feature_cols}")
    print()

    # Identify feature types
    numeric_features = [
        col for col in Config.NUMERIC_FEATURES if col in X.columns]
    categorical_features = [
        col for col in Config.CATEGORICAL_FEATURES if col in X.columns]

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(
        f"Categorical features ({
            len(categorical_features)}): {categorical_features}")
    print()

    return X, y, numeric_features, categorical_features


# ==============================================================================
# SECTION 6: TRAIN/TEST SPLIT
# ==============================================================================


def perform_train_test_split(X: pd.DataFrame,
                             y: pd.Series) -> Tuple[pd.DataFrame,
                                                    pd.DataFrame,
                                                    pd.Series,
                                                    pd.Series]:
    """
    Split data into training and test sets.

    Args:
        X: Feature matrix
        y: Target variable

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print_section("4. TRAIN/TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        shuffle=True
    )

    print("Split configuration:")
    print(f"  • Test size: {Config.TEST_SIZE * 100:.0f}%")
    print(f"  • Random state: {Config.RANDOM_STATE}")
    print("  • Shuffle: True")
    print()

    print("Split results:")
    print(
        f"  • Training set: {
            X_train.shape[0]} samples ({
            X_train.shape[0] /
            len(X) *
            100:.1f}%)")
    print(
        f"  • Test set: {
            X_test.shape[0]} samples ({
            X_test.shape[0] /
            len(X) *
            100:.1f}%)")
    print()

    print("Target distribution:")
    print("  Training set:")
    print(f"    - Mean: {y_train.mean():.2f}")
    print(f"    - Median: {y_train.median():.2f}")
    print(f"    - Std: {y_train.std():.2f}")
    print()
    print("  Test set:")
    print(f"    - Mean: {y_test.mean():.2f}")
    print(f"    - Median: {y_test.median():.2f}")
    print(f"    - Std: {y_test.std():.2f}")
    print()

    print("WHY NO STRATIFICATION:")
    print("  • Target variable is CONTINUOUS (not categorical)")
    print("  • Stratification is only applicable to classification problems")
    print("  • For regression, random shuffle ensures representative distribution")
    print()

    return X_train, X_test, y_train, y_test


# ==============================================================================
# SECTION 7: BUILD PREPROCESSING PIPELINE
# ==============================================================================


def build_preprocessing_pipeline(
        numeric_features: List[str],
        categorical_features: List[str]) -> ColumnTransformer:
    """
    Build preprocessing pipeline with imputation, encoding, and scaling.

    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names

    Returns:
        ColumnTransformer: Fitted preprocessing pipeline
    """
    print_section("5. BUILD PREPROCESSING PIPELINE")

    print("Pipeline structure:")
    print()
    print("NUMERIC PIPELINE:")
    print("  1. SimpleImputer(strategy='median')")
    print("     • Fills missing values with column median")
    print("  2. StandardScaler()")
    print("     • Transforms to zero mean and unit variance")
    print()

    print("CATEGORICAL PIPELINE:")
    print("  1. SimpleImputer(strategy='most_frequent')")
    print("     • Fills missing values with most common category")
    print("  2. OneHotEncoder(drop='first', handle_unknown='ignore')")
    print("     • Creates binary columns for each category")
    print("     • Drops first category to avoid multicollinearity")
    print("     • Handles unknown categories in test set gracefully")
    print()

    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(
                strategy='most_frequent')), ('encoder', OneHotEncoder(
                    drop='first', handle_unknown='ignore', sparse_output=False))])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    print("ColumnTransformer configuration:")
    print(
        f"  • Numeric features ({
            len(numeric_features)}): {numeric_features}")
    print(
        f"  • Categorical features ({
            len(categorical_features)}): {categorical_features}")
    print("  • Remainder policy: drop (exclude any unlisted columns)")
    print()

    return preprocessor


# ==============================================================================
# SECTION 8: FIT AND TRANSFORM DATA
# ==============================================================================


def fit_and_transform_data(preprocessor: ColumnTransformer,
                           X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           numeric_features: List[str],
                           categorical_features: List[str]) -> Tuple[np.ndarray,
                                                                     np.ndarray,
                                                                     List[str]]:
    """
    Fit preprocessor on training data and transform both sets.

    Args:
        preprocessor: ColumnTransformer pipeline
        X_train: Training features
        X_test: Test features
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names

    Returns:
        Tuple of (X_train_processed, X_test_processed, feature_names)
    """
    print_section("6. FIT PREPROCESSOR ON TRAINING DATA")

    print("⚠️  CRITICAL: Fitting transformers ONLY on training data")
    print("   This prevents data leakage from test set")
    print()

    # Fit on training data
    preprocessor.fit(X_train)
    print("✓ Preprocessor fitted on training data")
    print()

    print("What was learned from training data:")
    print()

    # Display numeric statistics
    print("NUMERIC FEATURES (StandardScaler):")
    numeric_transformer = preprocessor.named_transformers_['num']
    scaler = numeric_transformer.named_steps['scaler']
    imputer = numeric_transformer.named_steps['imputer']

    print("  Imputation values (median):")
    for feat, median_val in zip(numeric_features, imputer.statistics_):
        print(f"    • {feat}: {median_val:.2f}")
    print()

    print("  Scaling parameters:")
    for feat, mean_val, std_val in zip(
            numeric_features, scaler.mean_, scaler.scale_):
        print(f"    • {feat}:")
        print(f"      - Mean: {mean_val:.2f}")
        print(f"      - Std: {std_val:.2f}")
    print()

    # Display categorical statistics
    print("CATEGORICAL FEATURES (OneHotEncoder):")
    categorical_transformer = preprocessor.named_transformers_['cat']
    encoder = categorical_transformer.named_steps['encoder']
    cat_imputer = categorical_transformer.named_steps['imputer']

    print("  Imputation values (most frequent):")
    for feat, mode_val in zip(categorical_features, cat_imputer.statistics_):
        print(f"    • {feat}: {mode_val}")
    print()

    print("  Categories learned:")
    for feat, categories in zip(categorical_features, encoder.categories_):
        print(f"    • {feat}: {list(categories)}")
        print(f"      - Total categories: {len(categories)}")
        print(f"      - Dropped (first): {categories[0]}")
        print(f"      - Encoded categories: {list(categories[1:])}")
    print()

    # Transform both sets
    print_section("7. TRANSFORM BOTH TRAIN AND TEST SETS")

    print("Applying learned transformations to training set...")
    X_train_processed = preprocessor.transform(X_train)
    print(f"✓ X_train transformed: {X_train_processed.shape}")
    print()

    print("Applying learned transformations to test set...")
    print("IMPORTANT: We use the same parameters learned from training data.")
    print("           We do NOT refit on test data (this would cause data leakage).")
    X_test_processed = preprocessor.transform(X_test)
    print(f"✓ X_test transformed: {X_test_processed.shape}")
    print()

    # Shape comparison
    print("SHAPE COMPARISON:")
    print(
        f"  Original X_train:   {
            X_train.shape} → {
            X_train.shape[1]} features")
    print(
        f"  Processed X_train:  {
            X_train_processed.shape} → {
            X_train_processed.shape[1]} features")
    print()
    print(f"  Original X_test:    {X_test.shape} → {X_test.shape[1]} features")
    print(
        f"  Processed X_test:   {
            X_test_processed.shape} → {
            X_test_processed.shape[1]} features")
    print()

    # Feature count analysis
    original_feature_count = X_train.shape[1]
    processed_feature_count = X_train_processed.shape[1]
    feature_count_change = processed_feature_count - original_feature_count

    print("FEATURE COUNT ANALYSIS:")
    print(f"  Original features: {original_feature_count}")
    print(f"    • Numeric: {len(numeric_features)}")
    print(f"    • Categorical: {len(categorical_features)}")
    print()
    print(f"  After preprocessing: {processed_feature_count}")
    print(f"    • Numeric (scaled): {len(numeric_features)}")
    print(
        f"    • Categorical (one-hot encoded): {
            processed_feature_count -
            len(numeric_features)}")
    print()
    print(f"  Net change: {feature_count_change:+d} features")
    print()

    if feature_count_change > 0:
        print("Explanation of feature increase:")
        print(
            "  One-Hot Encoding expands categorical variables into multiple binary columns.")
        print("  For 'condicao_meteo' with drop='first':")
        print("    • Original: 1 categorical column")
        print(
            f"    • After OHE: {
                processed_feature_count -
                len(numeric_features)} binary columns")
    print()

    # Extract feature names
    print_section("8. EXTRACT FEATURE NAMES AFTER PREPROCESSING")

    try:
        feature_names_out = preprocessor.get_feature_names_out()
        print(
            f"✓ Extracted {
                len(feature_names_out)} feature names after preprocessing:")
        print()

        for i, name in enumerate(feature_names_out, 1):
            print(f"  {i:2d}. {name}")
        print()

        feature_names = list(feature_names_out)

    except AttributeError:
        print(
            "⚠️  Warning: Could not extract feature names (sklearn version may not support "
            "get_feature_names_out)")
        feature_names = []
        print()

    return X_train_processed, X_test_processed, feature_names


# ==============================================================================
# SECTION 9: SAVE ARTIFACTS
# ==============================================================================


def save_artifacts(preprocessor: ColumnTransformer,
                   X_train_processed: np.ndarray,
                   X_test_processed: np.ndarray,
                   y_train: pd.Series,
                   y_test: pd.Series,
                   feature_names: List[str]) -> None:
    """
    Save all preprocessing artifacts using pickle.

    Args:
        preprocessor: Fitted column transformer
        X_train_processed: Transformed training features
        X_test_processed: Transformed test features
        y_train: Training target
        y_test: Test target
        feature_names: List of feature names after transformation
    """
    print_section("9. PERSIST ARTIFACTS WITH PICKLE")

    print("Saving preprocessor and datasets for future use...")
    print()

    # Save preprocessor
    with open(Config.PREPROCESSOR_FILE, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"✓ Preprocessor saved to: {Config.PREPROCESSOR_FILE}")

    # Save processed training features
    with open(Config.X_TRAIN_FILE, 'wb') as f:
        pickle.dump(X_train_processed, f)
    print(f"✓ X_train (processed) saved to: {Config.X_TRAIN_FILE}")

    # Save processed test features
    with open(Config.X_TEST_FILE, 'wb') as f:
        pickle.dump(X_test_processed, f)
    print(f"✓ X_test (processed) saved to: {Config.X_TEST_FILE}")

    # Save training target
    with open(Config.Y_TRAIN_FILE, 'wb') as f:
        pickle.dump(y_train, f)
    print(f"✓ y_train saved to: {Config.Y_TRAIN_FILE}")

    # Save test target
    with open(Config.Y_TEST_FILE, 'wb') as f:
        pickle.dump(y_test, f)
    print(f"✓ y_test saved to: {Config.Y_TEST_FILE}")
    print()

    # Save feature names
    if feature_names:
        feature_names_df = pd.DataFrame({
            'Feature_Index': range(len(feature_names)),
            'Feature_Name': feature_names
        })
        feature_names_df.to_csv(Config.FEATURE_NAMES_FILE, index=False)
        print(f"✓ Feature names saved to: {Config.FEATURE_NAMES_FILE}")
        print()

    print("All artifacts saved successfully!")
    print()


# ==============================================================================
# SECTION 10: MAIN EXECUTION
# ==============================================================================


def main():
    """Main preprocessing pipeline execution."""
    print("=" * 120)
    print("DATA PREPROCESSING - FLIGHT TELEMETRY")
    print("=" * 120)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Set random seed for reproducibility
    np.random.seed(Config.RANDOM_STATE)

    try:
        # 1. Load and validate data
        df = load_and_validate_data()

        # 2. Explain imputation strategy
        explain_imputation_strategy()

        # 3. Prepare features and target
        X, y, numeric_features, categorical_features = prepare_features_and_target(
            df)

        # 4. Train/test split
        X_train, X_test, y_train, y_test = perform_train_test_split(X, y)

        # 5. Build preprocessing pipeline
        preprocessor = build_preprocessing_pipeline(
            numeric_features, categorical_features)

        # 6-8. Fit and transform data
        X_train_processed, X_test_processed, feature_names = fit_and_transform_data(
            preprocessor, X_train, X_test, numeric_features, categorical_features)

        # 9. Save artifacts
        save_artifacts(
            preprocessor,
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
            feature_names)

        # Final summary
        print("=" * 120)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 120)
        print()

        print("SUMMARY OF PREPROCESSING STEPS:")
        print()

        print("1. DATA LOADING AND VALIDATION")
        print(
            f"   • Loaded {
                df.shape[0]} samples with {
                df.shape[1]} features from {
                Config.DATA_FILE}")
        print("   • Validated all expected columns present")
        print()

        print("2. MISSING VALUE STRATEGY")
        print("   • Numeric features: Median imputation (robust to outliers)")
        print("   • Categorical features: Most frequent category imputation")
        print("   • Imputation values learned from training data only (NO data leakage)")
        print()

        print("3. FEATURE ENGINEERING")
        print(f"   • Target variable: {Config.TARGET}")
        print(
            f"   • Numeric features ({
                len(numeric_features)}): {numeric_features}")
        print(
            f"   • Categorical features ({
                len(categorical_features)}): {categorical_features}")
        print("   • One-Hot Encoding applied with drop='first' to prevent multicollinearity")
        print("   • StandardScaler applied to numeric features (mean=0, std=1)")
        print()

        print("4. TRAIN/TEST SPLIT")
        print(
            f"   • Training set: {
                X_train.shape[0]} samples ({
                X_train.shape[0] /
                len(X) *
                100:.1f}%)")
        print(
            f"   • Test set: {
                X_test.shape[0]} samples ({
                X_test.shape[0] /
                len(X) *
                100:.1f}%)")
        print(
            "   • Split method: Random shuffle (no stratification for continuous target)")
        print()

        print("5. PREPROCESSING PIPELINE")
        print("   • Built ColumnTransformer with separate pipelines for numeric and categorical features")
        print("   • Fitted on training data: learned medians, modes, means, stds, and categories")
        print("   • Transformed both train and test using learned parameters")
        print(
            f"   • Final feature count: {
                X_train_processed.shape[1]} (from {
                X_train.shape[1]} original)")
        print()

        print("6. ARTIFACTS SAVED")
        print(f"   • {Config.PREPROCESSOR_FILE}")
        print(f"   • {Config.X_TRAIN_FILE}")
        print(f"   • {Config.X_TEST_FILE}")
        print(f"   • {Config.Y_TRAIN_FILE}")
        print(f"   • {Config.Y_TEST_FILE}")
        if feature_names:
            print(f"   • {Config.FEATURE_NAMES_FILE}")
        print()

        print("=" * 120)
        print("NEXT STEPS:")
        print("  1. Load the preprocessed datasets (X_train, X_test, y_train, y_test) for model training")
        print("  2. Train various regression models (Linear Regression, Random Forest, XGBoost, etc.)")
        print("  3. Evaluate models using MAE and RMSE metrics")
        print("  4. Use the saved preprocessor to transform new data in production")
        print("=" * 120)
        print()

        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Script completed successfully!")

    except Exception as e:
        print("\n✗ ERROR during preprocessing:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
