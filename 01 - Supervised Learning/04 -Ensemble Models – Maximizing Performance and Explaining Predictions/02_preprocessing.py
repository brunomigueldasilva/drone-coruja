#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FLIGHT INCIDENT PREDICTION - DATA PREPROCESSING
================================================================================

Purpose: Prepare flight data for machine learning (Random Forest, XGBoost)

This script implements critical preprocessing concepts:
- Stratified train/test split for imbalanced datasets
- Data leakage prevention through proper scaling
- Artifact persistence for future predictions

Key Learning Objectives:
1. Maintain class distribution through stratified splitting
2. Prevent data leakage by fitting scalers only on training data
3. Save preprocessing artifacts for production deployment

Author: Bruno Silva
Date: 2025
================================================================================
"""

# ================================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ================================================================================

import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Configuration
class Config:
    """Configuration parameters for preprocessing pipeline."""
    # File paths
    INPUT_FILE = Path('inputs') / 'voos_pre_voo_clean.csv'
    OUTPUT_DIR = Path('outputs') / 'data_processed'

    # Target column
    TARGET_COL = 'incidente_reportado'

    # Splitting parameters
    TEST_SIZE = 0.20        # 20% for testing
    RANDOM_STATE = 42       # For reproducibility


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


def save_pickle(obj, filepath: Path) -> None:
    """
    Save Python object to pickle file.

    Args:
        obj: Object to serialize
        filepath: Output file path

    Why we save artifacts:
    - Allows us to apply identical transformations to new data
    - Essential for production deployment
    - Ensures consistency between training and prediction
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  ✓ Saved: {filepath.name}")


# ================================================================================
# SECTION 3: DATA LOADING
# ================================================================================


def load_dataset() -> pd.DataFrame:
    """
    Load flight incident dataset from CSV.

    Returns:
        DataFrame with flight data

    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    print_header("STEP 1: LOADING DATASET")

    if not Config.INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {Config.INPUT_FILE.absolute()}\n"
            f"Please ensure the file exists in the inputs/ directory."
        )

    df = pd.read_csv(Config.INPUT_FILE)

    print("✓ Dataset loaded successfully")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\n  Columns:")
    for col in df.columns:
        print(f"    - {col}")

    # Verify target column exists
    if Config.TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{Config.TARGET_COL}' not found in dataset")

    return df


# ================================================================================
# SECTION 4: FEATURE ENGINEERING
# ================================================================================


def encode_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Get feature names from cleaned dataset.

    Args:
        X: Feature DataFrame (already encoded from exploratory analysis)

    Returns:
        Tuple of (DataFrame, list of feature names)

    Note: The input dataset already has:
    - turb_num (turbulence encoded as 0, 1, 2)
    - missao_* columns (one-hot encoded mission types)
    - razao_manut_idade (interaction feature)
    - exp_x_turb (interaction feature)
    """
    print_header("STEP 2: FEATURES FROM CLEANED DATASET")

    feature_names = X.columns.tolist()

    print(f"✓ Total features: {len(feature_names)}")
    print("  Feature list:")
    for i, feat in enumerate(feature_names, 1):
        print(f"    {i}. {feat}")

    print("\nNote: Features already preprocessed in exploratory analysis:")
    print("  • turb_num: turbulence encoded (0=Baixa, 1=Média, 2=Alta)")
    print("  • missao_*: one-hot encoded mission types")
    print("  • razao_manut_idade: maintenance ratio interaction")
    print("  • exp_x_turb: experience × turbulence interaction")

    return X, feature_names


# ================================================================================
# SECTION 5: TRAIN-TEST SPLIT WITH STRATIFICATION
# ================================================================================


def perform_stratified_split(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/test sets with stratification.

    Args:
        X: Features
        y: Target variable

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    CRITICAL CONCEPT: STRATIFIED SPLITTING
    --------------------------------------
    Regular random split might create imbalanced train/test sets, especially
    with imbalanced data (e.g., 90% no incident, 10% incident).

    Example of the problem (WITHOUT stratification):
    - Overall: 900 no incident, 100 incident (9:1 ratio)
    - By chance, train set might get: 850 no incident, 50 incident (17:1 ratio)
    - Test set would get: 50 no incident, 50 incident (1:1 ratio)
    - This misrepresents the true class distribution!

    Stratified split SOLUTION:
    - Maintains the exact class proportion in both train and test sets
    - If overall is 9:1, both train and test will also be 9:1
    - More reliable evaluation metrics
    - Particularly important for imbalanced datasets

    Why it matters:
    - Test set should represent the real-world distribution
    - Without stratification, you might train on different distribution than you evaluate on
    - This leads to misleading performance metrics
    """
    print_header("STEP 3: STRATIFIED TRAIN-TEST SPLIT")

    # Check class distribution before split
    class_counts = y.value_counts().sort_index()
    print("Original class distribution:")
    for label, count in class_counts.items():
        print(f"  Class {label}: {count:,} samples ({count / len(y) * 100:.2f}%)")

    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y  # KEY: Maintain class proportions
    )

    print("\n✓ Split completed:")
    print(f"  Training set:   {X_train.shape[0]:,} samples ({(1 - Config.TEST_SIZE) * 100:.0f}%)")
    print(f"  Test set:       {X_test.shape[0]:,} samples ({Config.TEST_SIZE * 100:.0f}%)")

    # Verify stratification worked
    print("\nClass distribution after split:")
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    print("  Training set:")
    for label, count in train_counts.items():
        print(f"    Class {label}: {count:,} samples ({count / len(y_train) * 100:.2f}%)")

    print("  Test set:")
    for label, count in test_counts.items():
        print(f"    Class {label}: {count:,} samples ({count / len(y_test) * 100:.2f}%)")

    print("  ✓ Stratification successful - proportions maintained!")

    return X_train, X_test, y_train, y_test


# ================================================================================
# SECTION 6: FEATURE SCALING (WITH DATA LEAKAGE PREVENTION)
# ================================================================================


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler, preventing data leakage.

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler)

    CRITICAL CONCEPT: DATA LEAKAGE PREVENTION
    -----------------------------------------
    Data leakage occurs when information from the test set "leaks" into the
    training process, leading to overly optimistic performance estimates.

    WRONG approach (causes data leakage):
    ```python
    scaler = StandardScaler()
    scaler.fit(pd.concat([X_train, X_test]))  # ❌ WRONG! Leakage!
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

    Why this is wrong:
    - The scaler learns mean and std from BOTH train and test data
    - Test data influences the scaling parameters
    - Model has indirect access to test set information
    - Results in overfitting and unrealistic metrics

    CORRECT approach (no leakage):
    ```python
    scaler = StandardScaler()
    scaler.fit(X_train)              # ✓ Learn only from training data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

    Why this is correct:
    - Scaler learns parameters ONLY from training data
    - Test set remains completely unseen during fitting
    - Simulates real production scenario where new data has unknown distribution
    - Provides honest performance estimates

    StandardScaler formula:
    - z = (x - mean) / std
    - For each feature: subtract mean, divide by standard deviation
    - Results in features with mean ≈ 0 and std ≈ 1

    Why we need scaling:
    - Many ML algorithms are sensitive to feature scales
    - Features with large ranges can dominate smaller-scale features
    - Gradient descent converges faster with normalized features
    - Some models (SVM, Neural Networks) require it
    - Tree-based models (RF, XGBoost) are less sensitive but can still benefit

    Note for tree-based models:
    - Decision trees and Random Forests are invariant to feature scaling
    - They split based on feature values directly
    - However, scaling doesn't hurt and maintains consistency
    - XGBoost can benefit from scaling in regularization
    """
    print_header("STEP 4: FEATURE SCALING (StandardScaler)")

    print("Why StandardScaler?")
    print("  • Transforms features to have mean=0 and std=1")
    print("  • Formula: z = (x - mean) / std")
    print("  • Ensures all features contribute equally to model")
    print("  • Required for distance-based algorithms")
    print("  • Improves gradient descent convergence\n")

    # Initialize scaler
    scaler = StandardScaler()

    # CRITICAL: Fit ONLY on training data
    print("Fitting scaler on TRAINING data only...")
    scaler.fit(X_train)
    print("  ✓ Learned mean and std from training set")

    # Transform both sets
    print("\nTransforming features...")
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    print(f"  ✓ Training set scaled: {X_train_scaled.shape}")

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    print(f"  ✓ Test set scaled: {X_test_scaled.shape}")

    # Verify scaling (training set should have mean ≈ 0, std ≈ 1)
    print("\nVerifying scaling (training set):")
    print(f"  Mean (should be ≈0):  {X_train_scaled.mean().mean():.6f}")
    print(f"  Std  (should be ≈1):  {X_train_scaled.std().mean():.6f}")

    print("\n⚠️  IMPORTANT: Test set statistics will differ slightly")
    print("    This is CORRECT and expected - it simulates production where new data")
    print("    may have different statistics than training data.")

    return X_train_scaled, X_test_scaled, scaler


# ================================================================================
# SECTION 7: SAVE PROCESSED DATA
# ================================================================================


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
    feature_names: List[str]
) -> None:
    """
    Save all processed data and preprocessing artifacts.

    Args:
        X_train: Scaled training features
        X_test: Scaled test features
        y_train: Training labels
        y_test: Test labels
        scaler: Fitted StandardScaler object
        feature_names: List of feature names
    """
    print_header("STEP 5: SAVING PROCESSED DATA")

    # Create output directory
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {Config.OUTPUT_DIR.absolute()}")

    print("\nSaving processed datasets (CSV format):")

    # Save training data
    X_train.to_csv(Config.OUTPUT_DIR / 'X_train_scaled.csv', index=False)
    print(f"  ✓ X_train_scaled.csv ({X_train.shape[0]} rows × {X_train.shape[1]} cols)")

    # Save test data
    X_test.to_csv(Config.OUTPUT_DIR / 'X_test_scaled.csv', index=False)
    print(f"  ✓ X_test_scaled.csv ({X_test.shape[0]} rows × {X_test.shape[1]} cols)")

    # Save labels
    y_train.to_csv(Config.OUTPUT_DIR / 'y_train.csv', index=False, header=True)
    print(f"  ✓ y_train.csv ({len(y_train)} samples)")

    y_test.to_csv(Config.OUTPUT_DIR / 'y_test.csv', index=False, header=True)
    print(f"  ✓ y_test.csv ({len(y_test)} samples)")

    print("\nSaving preprocessing artifacts (pickle format):")

    # Save scaler
    save_pickle(scaler, Config.OUTPUT_DIR / 'scaler.pkl')
    print("    Purpose: Apply same standardization to new data")

    # Save feature names
    save_pickle(feature_names, Config.OUTPUT_DIR / 'feature_names.pkl')
    print("    Purpose: Ensure correct feature order in production")

    print(f"\n✓ All files saved successfully to: {Config.OUTPUT_DIR}/")


# ================================================================================
# SECTION 8: FINAL SUMMARY
# ================================================================================


def print_final_summary(
    df_shape: Tuple[int, int],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_original_features: int,
    n_final_features: int
) -> None:
    """
    Print comprehensive preprocessing summary.

    Args:
        df_shape: Original DataFrame shape
        X_train: Processed training features
        X_test: Processed test features
        y_train: Training labels
        y_test: Test labels
        n_original_features: Number of features before encoding
        n_final_features: Number of features after encoding
    """
    print_header("PREPROCESSING SUMMARY", "=")

    # Calculate class distributions
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    summary = f"""
📊 DATASET INFORMATION
{'-' * 100}
Input file:     {Config.INPUT_FILE}
Original shape: {df_shape[0]:,} rows × {df_shape[1]} columns

📝 PREPROCESSING STEPS COMPLETED
{'-' * 100}
✓ [1] Loaded cleaned dataset (features already engineered from exploratory analysis)
✓ [2] Features include:
      • 3 original numeric features (scaled)
      • turb_num: turbulence encoded (0=Baixa, 1=Média, 2=Alta)
      • missao_*: one-hot encoded mission types
      • razao_manut_idade: maintenance hours / aircraft age
      • exp_x_turb: pilot experience × turbulence
✓ [3] Performed stratified train-test split (80/20)
✓ [4] Applied StandardScaler (fitted on train only, no data leakage)
✓ [5] Saved processed data and artifacts

📦 OUTPUT FILES
{'-' * 100}
Location: {Config.OUTPUT_DIR}/

CSV Files (processed data):
  • X_train_scaled.csv  - {X_train.shape[0]:,} samples × {X_train.shape[1]} features
  • X_test_scaled.csv   - {X_test.shape[0]:,} samples × {X_test.shape[1]} features
  • y_train.csv         - {len(y_train):,} labels
  • y_test.csv          - {len(y_test):,} labels

Pickle Files (artifacts for deployment):
  • scaler.pkl          - Fitted StandardScaler object
  • feature_names.pkl   - List of {n_final_features} feature names

📈 CLASS DISTRIBUTION (STRATIFIED)
{'-' * 100}
Training set ({len(y_train):,} samples):
  Class 0 (No incident):  {train_counts[0]:,} samples ({train_counts[0] / len(y_train) * 100:.2f}%)
  Class 1 (With incident): {train_counts[1]:,} samples ({train_counts[1] / len(y_train) * 100:.2f}%)
  Ratio: {train_counts[0] / train_counts[1]:.2f}:1

Test set ({len(y_test):,} samples):
  Class 0 (No incident):  {test_counts[0]:,} samples ({test_counts[0] / len(y_test) * 100:.2f}%)
  Class 1 (With incident): {test_counts[1]:,} samples ({test_counts[1] / len(y_test) * 100:.2f}%)
  Ratio: {test_counts[0] / test_counts[1]:.2f}:1

✓ Class proportions maintained in both sets (stratification successful)

🔬 FEATURE ENGINEERING
{'-' * 100}
Final features:     {n_final_features}

Feature transformations:
  • All numeric features → StandardScaler (mean=0, std=1)
  • turb_num → Already encoded from exploratory analysis
  • missao_* → Already one-hot encoded from exploratory analysis
  • Interaction features included: razao_manut_idade, exp_x_turb

🎯 KEY DECISIONS AND RATIONALE
{'-' * 100}
1. Using Cleaned Dataset:
   ✓ Features already engineered in exploratory analysis
   ✓ Turbulence already encoded (0, 1, 2)
   ✓ Mission types already one-hot encoded
   ✓ Interaction features already created

2. Stratified Split:
   ✓ Maintains class balance in train/test
   ✓ Critical for imbalanced datasets (~9:1 ratio)
   ✓ Ensures reliable model evaluation

3. Data Leakage Prevention:
   ✓ Scaler fitted ONLY on training data
   ✓ Test set remains completely unseen during fitting
   ✓ Simulates real production scenario
   ✓ Prevents overly optimistic metrics

🚀 NEXT STEPS
{'-' * 100}
✓ Data is now ready for model training
✓ Use X_train_scaled.csv and y_train.csv for training
✓ Use X_test_scaled.csv and y_test.csv for evaluation
✓ Models to try: Decision Tree, Random Forest, XGBoost, Gradient Boosting
✓ Remember to use class_weight='balanced' for imbalanced data
✓ Focus on F1-Score, ROC-AUC (not just accuracy)

💡 PRODUCTION DEPLOYMENT
{'-' * 100}
To process new data for predictions:
  1. Load scaler.pkl and feature_names.pkl
  2. Ensure new data has same features (from exploratory analysis)
  3. Use scaler.transform() on new data (NOT fit_transform!)
  4. Ensure feature order matches feature_names.pkl
  5. Pass to trained model for predictions
"""
    print(summary)


# ================================================================================
# SECTION 9: MAIN FUNCTION
# ================================================================================


def main() -> None:
    """
    Main preprocessing pipeline.

    Workflow:
    1. Load cleaned dataset
    2. Separate features (X) and target (y)
    3. Get feature names
    4. Stratified train-test split
    5. Scale features (preventing data leakage)
    6. Save processed data and artifacts
    7. Print summary
    """
    print("=" * 100)
    print("FLIGHT INCIDENT PREDICTION - DATA PREPROCESSING PIPELINE")
    print("=" * 100)

    # Step 1: Load data
    df = load_dataset()
    original_shape = df.shape

    # Step 2: Separate features and target
    print_header("SEPARATING FEATURES AND TARGET")
    y = df[Config.TARGET_COL].copy()
    X = df.drop(columns=[Config.TARGET_COL])

    print(f"✓ Features (X): {X.shape[0]:,} rows × {X.shape[1]} columns")
    print(f"✓ Target (y):   {len(y):,} samples")

    n_original_features = X.shape[1]

    # Step 3: Get feature names
    X_encoded, feature_names = encode_features(X)

    # Step 4: Stratified split
    X_train, X_test, y_train, y_test = perform_stratified_split(X_encoded, y)

    # Step 5: Scale features (preventing data leakage)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 6: Save everything
    save_processed_data(
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scaler, feature_names
    )

    # Step 7: Print final summary
    print_final_summary(
        original_shape,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        n_original_features,
        len(feature_names)
    )

    print("=" * 100)
    print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 100)


# ================================================================================
# EXECUTION
# ================================================================================

if __name__ == "__main__":
    main()
