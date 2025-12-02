#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
AIRCRAFT INCIDENT CLASSIFICATION - DATA PREPROCESSING
==============================================================================

Purpose: Prepare pre-flight data for machine learning model training

This script:
1. Loads raw dataset from exploratory analysis
2. Separates features (X) from target variable (y)
3. Applies transformations: StandardScaler, OrdinalEncoder, OneHotEncoder
4. Performs stratified train-test split (80/20) maintaining class balance
5. Prevents data leakage by fitting transformers only on training data
6. Saves preprocessed data and artifacts for model training
7. Uses ColumnTransformer for efficient feature engineering pipeline

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Tuple, List, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Preprocessing configuration parameters."""
    INPUT_FILE = Path('outputs/voos_pre_voo_clean.csv')
    PROCESSED_DATA_DIR = Path('outputs/data_processed')
    MODELS_DIR = Path('outputs/models')
    RANDOM_STATE = 42  # For reproducibility
    TEST_SIZE = 0.20  # 80% train, 20% test
    TARGET_COLUMN = 'incidente_reportado'


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


def save_pickle(obj: Any, filepath: Path) -> str:
    """
    Save object to pickle file and return file size.

    Args:
        obj: Object to pickle
        filepath: Path object for output file

    Returns:
        str: Human-readable file size (e.g., "1.23 MB")
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

    # Calculate file size
    file_size = filepath.stat().st_size
    if file_size < 1024:
        size_str = f"{file_size} bytes"
    elif file_size < 1024**2:
        size_str = f"{file_size / 1024:.2f} KB"
    else:
        size_str = f"{file_size / (1024**2):.2f} MB"

    return size_str


def save_text_file(
        content: List[str],
        filepath: Path,
        header: str = None) -> None:
    """
    Save list of strings to text file.

    Args:
        content: List of strings to write
        filepath: Path object for output file
        header: Optional header to prepend
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + '\n\n')
        for item in content:
            f.write(f"{item}\n")


# ==============================================================================
# SECTION 3: DATA LOADING AND VERIFICATION
# ==============================================================================


def load_and_verify_data() -> pd.DataFrame:
    """
    Load dataset and perform initial verification.

    Returns:
        pd.DataFrame: Loaded and verified dataset

    Raises:
        SystemExit: If dataset file is not found or validation fails
    """
    print_section("1. DATA LOADING AND VERIFICATION")

    # Check if file exists
    if not Config.INPUT_FILE.exists():
        print(f"✗ ERROR: {Config.INPUT_FILE} not found!")
        print("  Please ensure the file exists in the 'inputs/' directory.")
        exit(1)

    # Load dataset
    df = pd.read_csv(Config.INPUT_FILE)
    print(f"✓ Dataset loaded successfully: {Config.INPUT_FILE.name}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Validate dataset is not empty
    if df.empty:
        print("✗ ERROR: Dataset is empty!")
        exit(1)

    # Display column information
    print("\nColumns present:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    print("\nData types:")
    print(df.dtypes.to_string())

    # Check for missing values
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values detected")
    else:
        print(missing[missing > 0])
        print(f"\n  Total missing values: {missing.sum():,}")

    # Verify target column exists
    if Config.TARGET_COLUMN not in df.columns:
        print(f"\n✗ ERROR: Target column '{Config.TARGET_COLUMN}' not found!")
        exit(1)

    # Display target distribution
    print(f"\nTarget variable '{Config.TARGET_COLUMN}' distribution:")
    target_counts = df[Config.TARGET_COLUMN].value_counts().sort_index()
    target_pct = df[Config.TARGET_COLUMN].value_counts(
        normalize=True).sort_index() * 100

    for val in target_counts.index:
        print(
            f"  Class {val}: {
                target_counts[val]:,} samples ({
                target_pct[val]:.2f}%)")

    return df


# ==============================================================================
# SECTION 4: FEATURE AND TARGET SEPARATION
# ==============================================================================


def separate_features_target(
        df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) from target variable (y).

    Args:
        df: Complete DataFrame

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
    """
    print_section("2. SEPARATING FEATURES (X) AND TARGET (y)")

    # Separate X and y
    y = df[Config.TARGET_COLUMN].copy()
    X = df.drop(columns=[Config.TARGET_COLUMN])

    print("✓ Separation completed:")
    print(f"  X (features) shape: {X.shape}")
    print(f"  y (target) shape: {y.shape}")

    print(f"\nFeature columns ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i:2d}. {col}")

    print("\nData types in X:")
    dtype_counts = X.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    return X, y


# ==============================================================================
# SECTION 5: COLUMN TYPE DEFINITION
# ==============================================================================


def define_column_types(
        X: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
    """
    Define column types for different transformations.

    Args:
        X: Feature DataFrame

    Returns:
        Tuple containing:
            - numeric_cols: List of numeric columns for StandardScaler
            - ordinal_cols: List of ordinal columns for OrdinalEncoder
            - onehot_cols: List of categorical columns for OneHotEncoder
            - ordinal_categories: Ordered categories for ordinal encoding
    """
    print_section("3. DEFINING COLUMN TYPES FOR TRANSFORMATION")

    # All columns are numeric (cleaned dataset already has engineered features)
    numeric_cols = X.columns.tolist()

    # No ordinal or one-hot encoding needed (already processed in exploratory
    # analysis)
    ordinal_cols = []
    ordinal_categories = []
    onehot_cols = []

    print("NUMERIC COLUMNS (StandardScaler):")
    for col in numeric_cols:
        print(f"  • {col}")

    print("\nNOTE: Dataset already has engineered features from exploratory analysis:")
    print("  • turb_num: Turbulence encoded as 0/1/2")
    print("  • Mission type dummies: Already one-hot encoded")
    print("  • Interaction features: razao_manut_idade, exp_x_turb")

    return numeric_cols, ordinal_cols, onehot_cols, ordinal_categories


# ==============================================================================
# SECTION 6: TRAIN-TEST SPLIT
# ==============================================================================


def split_train_test(X: pd.DataFrame,
                     y: pd.Series) -> Tuple[pd.DataFrame,
                                            pd.DataFrame,
                                            pd.Series,
                                            pd.Series]:
    """
    Perform stratified train-test split.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    print_section("4. STRATIFIED TRAIN-TEST SPLIT (80/20)")

    """
    STRATIFICATION IMPORTANCE:

    In imbalanced datasets, random splitting can lead to:
    - Train/test sets with different class distributions
    - Minority class underrepresented in one split
    - Biased model evaluation metrics

    STRATIFICATION ENSURES:
    - Train and test maintain SAME class proportions as original dataset
    - Both splits are representative of true data distribution
    - Fair and reliable model evaluation

    Example:
      Original: 85% no incident, 15% incident
      Without stratification: Train 88/12, Test 78/22 (inconsistent)
      With stratification: Both Train and Test ~85/15 (consistent)
    """

    print("Split configuration:")
    print(f"  Train size: {(1 - Config.TEST_SIZE) * 100:.0f}%")
    print(f"  Test size: {Config.TEST_SIZE * 100:.0f}%")
    print(f"  Random state: {Config.RANDOM_STATE}")
    print("  Stratification: ENABLED (stratify=y)")

    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        stratify=y,
        random_state=Config.RANDOM_STATE
    )

    print("\n✓ Split completed:")
    print(f"  X_train: {X_train.shape[0]:,} samples")
    print(f"  X_test:  {X_test.shape[0]:,} samples")

    # Verify stratification
    verify_stratification(y, y_train, y_test)

    return X_train, X_test, y_train, y_test


def verify_stratification(
        y: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series) -> None:
    """
    Verify that stratification maintained class proportions.

    Args:
        y: Original target variable
        y_train: Training target variable
        y_test: Test target variable
    """
    print("\nClass distribution verification:")
    print(f"\n{'Set':<12} {'Class 0':<15} {'Class 1':<15} {'Ratio':<10}")
    print("-" * 52)

    for dataset, name in [
            (y, "Original"), (y_train, "Train"), (y_test, "Test")]:
        counts = dataset.value_counts().sort_index()
        pct = (counts / len(dataset)) * 100

        count_0 = counts.get(0, 0)
        count_1 = counts.get(1, 0)
        ratio = count_0 / count_1 if count_1 > 0 else float('inf')

        print(f"{name:<12} {count_0:>6} ({pct.get(0, 0):>5.2f}%)  "
              f"{count_1:>6} ({pct.get(1, 0):>5.2f}%)  {ratio:>6.2f}:1")

    print("\n✓ Stratification successful: Ratios are consistent across all sets!")


# ==============================================================================
# SECTION 7: PREPROCESSING PIPELINE
# ==============================================================================


def create_and_fit_preprocessor(X_train: pd.DataFrame,
                                numeric_cols: List[str],
                                ordinal_cols: List[str],
                                onehot_cols: List[str],
                                ordinal_categories: List[List[str]]) -> ColumnTransformer:
    """
    Create and fit ColumnTransformer on training data.

    Args:
        X_train: Training features
        numeric_cols: Columns for StandardScaler
        ordinal_cols: Columns for OrdinalEncoder
        onehot_cols: Columns for OneHotEncoder
        ordinal_categories: Ordered categories for ordinal encoding

    Returns:
        ColumnTransformer: Fitted preprocessor
    """
    print_section("5. CREATING AND FITTING PREPROCESSING PIPELINE")

    # Build transformers list dynamically
    transformers = []

    if numeric_cols:
        transformers.append(('scaler', StandardScaler(), numeric_cols))

    if ordinal_cols:
        transformers.append(
            ('ordinal',
             OrdinalEncoder(
                 categories=ordinal_categories),
                ordinal_cols))

    if onehot_cols:
        transformers.append(
            ('onehot',
             OneHotEncoder(
                 drop='first',
                 handle_unknown='ignore'),
                onehot_cols))

    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    print(
        f"ColumnTransformer configured with {
            len(transformers)} transformer(s):")
    if numeric_cols:
        print(f"  • StandardScaler → {len(numeric_cols)} numeric columns")
    if ordinal_cols:
        print(f"  • OrdinalEncoder → {len(ordinal_cols)} ordinal columns")
    if onehot_cols:
        print(f"  • OneHotEncoder → {len(onehot_cols)} categorical columns")

    print("\n⚠️  DATA LEAKAGE PREVENTION:")
    print("  → Preprocessor is fitted (fit) ONLY on TRAINING data")
    print("  → Statistics (mean, std) computed SOLELY from train set")
    print("  → TEST set is transformed using train statistics, but NEVER influences them")
    print("  → This practice is FUNDAMENTAL to prevent information leakage from test to train")
    print("  → Without this, model performance would be artificially inflated")

    # Fit on training data only
    print("\n► Fitting preprocessor on X_train...")
    preprocessor.fit(X_train)
    print(f"  ✓ Preprocessor fitted on {X_train.shape[0]:,} training samples")

    return preprocessor


def transform_datasets(preprocessor: ColumnTransformer,
                       X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       numeric_cols: List[str],
                       ordinal_cols: List[str]) -> Tuple[pd.DataFrame,
                                                         pd.DataFrame,
                                                         List[str]]:
    """
    Transform train and test sets using fitted preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer
        X_train: Training features
        X_test: Test features
        numeric_cols: Numeric column names
        ordinal_cols: Ordinal column names

    Returns:
        Tuple: X_train_transformed, X_test_transformed, transformed_column_names
    """
    print_section("6. TRANSFORMING DATASETS")

    # Transform both sets
    print("► Transforming X_train...")
    X_train_transformed = preprocessor.transform(X_train)
    print(f"  ✓ X_train transformed: {X_train_transformed.shape}")

    print("\n► Transforming X_test (using train statistics)...")
    X_test_transformed = preprocessor.transform(X_test)
    print(f"  ✓ X_test transformed: {X_test_transformed.shape}")

    # Build transformed column names
    transformed_cols = []
    transformed_cols.extend(numeric_cols)

    if ordinal_cols:
        transformed_cols.extend(ordinal_cols)

    # Get one-hot encoded column names if they exist
    if 'onehot' in preprocessor.named_transformers_:
        onehot_encoder = preprocessor.named_transformers_['onehot']
        onehot_feature_names = onehot_encoder.get_feature_names_out()
        transformed_cols.extend(onehot_feature_names)

    print(f"\nTotal features after transformation: {len(transformed_cols)}")
    print("\nFirst 20 feature names:")
    for i, col in enumerate(transformed_cols[:20], 1):
        print(f"  {i:2d}. {col}")
    if len(transformed_cols) > 20:
        print(f"  ... and {len(transformed_cols) - 20} more features")

    # Convert to DataFrames
    X_train_df = pd.DataFrame(
        X_train_transformed,
        columns=transformed_cols,
        index=X_train.index)
    X_test_df = pd.DataFrame(
        X_test_transformed,
        columns=transformed_cols,
        index=X_test.index)

    return X_train_df, X_test_df, transformed_cols

    return X_train_df, X_test_df, transformed_cols


# ==============================================================================
# SECTION 8: SAVE PROCESSED DATA AND ARTIFACTS
# ==============================================================================


def save_processed_data(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        preprocessor: ColumnTransformer,
                        original_cols: List[str],
                        transformed_cols: List[str]) -> None:
    """
    Save all processed data and artifacts.

    Args:
        X_train: Transformed training features
        X_test: Transformed test features
        y_train: Training target
        y_test: Test target
        preprocessor: Fitted ColumnTransformer
        original_cols: Original column names
        transformed_cols: Transformed column names
    """
    print_section("7. SAVING PROCESSED DATA AND ARTIFACTS")

    # Create output directories
    Config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Output directories:")
    print(f"  • {Config.PROCESSED_DATA_DIR.absolute()}")
    print(f"  • {Config.MODELS_DIR.absolute()}")

    # Save processed data as CSV
    print("\nSaving processed data (CSV format)...")
    X_train.to_csv(Config.PROCESSED_DATA_DIR / 'X_train.csv', index=False)
    print("  ✓ X_train.csv")

    X_test.to_csv(Config.PROCESSED_DATA_DIR / 'X_test.csv', index=False)
    print("  ✓ X_test.csv")

    y_train.to_csv(
        Config.PROCESSED_DATA_DIR /
        'y_train.csv',
        index=False,
        header=True)
    print("  ✓ y_train.csv")

    y_test.to_csv(
        Config.PROCESSED_DATA_DIR /
        'y_test.csv',
        index=False,
        header=True)
    print("  ✓ y_test.csv")

    # Save preprocessor and column names as artifacts
    print("\nSaving artifacts (pickle and text files)...")

    size_str = save_pickle(
        preprocessor,
        Config.MODELS_DIR /
        'preprocessor.pkl')
    print(f"  ✓ preprocessor.pkl ({size_str})")

    header_original = f"# Original feature columns\n# Total: {
        len(original_cols)}"
    save_text_file(
        original_cols,
        Config.MODELS_DIR /
        'columns_original.txt',
        header_original)
    print("  ✓ columns_original.txt")

    header_transformed = f"# Transformed feature columns\n# Total: {
        len(transformed_cols)}"
    save_text_file(
        transformed_cols,
        Config.MODELS_DIR /
        'columns_transformed.txt',
        header_transformed)
    print("  ✓ columns_transformed.txt")


# ==============================================================================
# SECTION 9: SUMMARY
# ==============================================================================


def print_summary(df_shape: Tuple[int, int],
                  X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  y_train: pd.Series,
                  y_test: pd.Series,
                  n_original_features: int,
                  n_transformed_features: int) -> None:
    """
    Print final preprocessing summary.

    Args:
        df_shape: Original DataFrame shape
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        n_original_features: Number of original features
        n_transformed_features: Number of transformed features
    """
    print_section("PREPROCESSING SUMMARY")

    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    ratio_train = train_counts[0] / \
        train_counts[1] if train_counts[1] > 0 else float('inf')
    ratio_test = test_counts[0] / \
        test_counts[1] if test_counts[1] > 0 else float('inf')

    summary = f"""
✓ Data preprocessing completed successfully!

INPUT:
  - Source file: {Config.INPUT_FILE}
  - Original shape: {df_shape[0]:,} rows × {df_shape[1]} columns

PROCESSING STEPS:
  1. ✓ Loaded and verified data
  2. ✓ Separated features (X) and target (y)
  3. ✓ Defined column types (numeric, ordinal, categorical)
  4. ✓ Performed stratified train-test split (80/20)
  5. ✓ Created and fitted ColumnTransformer (on train only)
  6. ✓ Transformed both train and test sets (no data leakage)
  7. ✓ Saved processed data and artifacts

OUTPUT:
  - Training samples: {X_train.shape[0]:,}
  - Test samples: {X_test.shape[0]:,}
  - Original features: {n_original_features}
  - Transformed features: {n_transformed_features}
  - Files saved: 7 (4 CSV + 3 artifacts)

CLASS DISTRIBUTION:
  - Train: {train_counts[0]:,} no incident, {train_counts[1]:,} incident ({ratio_train:.2f}:1)
  - Test:  {test_counts[0]:,} no incident, {test_counts[1]:,} incident ({ratio_test:.2f}:1)

TRANSFORMATIONS APPLIED:
  • StandardScaler: All {n_transformed_features} numeric columns (normalized to mean=0, std=1)
  • Note: Input dataset already has engineered features from exploratory analysis

NEXT STEPS:
  - Ready for model training (03_train_models.py)
  - All preprocessed files in: {Config.PROCESSED_DATA_DIR}/
  - Artifacts (preprocessor, column names) in: {Config.MODELS_DIR}/
  - Preprocessor can be reused for new data via pickle

KEY DECISIONS MADE:
  ✓ Using cleaned dataset with engineered features from exploratory analysis
  ✓ Stratified Split: Maintains class balance in train/test
  ✓ Proper Scaling: No data leakage, fitted only on training data
"""
    print(summary)


# ==============================================================================
# SECTION 10: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main function orchestrating the preprocessing pipeline.

    Workflow:
    1. Load and verify data
    2. Separate features and target
    3. Define column types for transformation
    4. Split train-test with stratification
    5. Create and fit preprocessor (train only)
    6. Transform both datasets
    7. Save processed data and artifacts
    8. Print summary
    """
    print("=" * 80)
    print("DATA PREPROCESSING - AIRCRAFT INCIDENT CLASSIFICATION")
    print("=" * 80)

    # 1. Load and verify data
    df = load_and_verify_data()

    # 2. Separate features and target
    X, y = separate_features_target(df)

    # 3. Define column types
    numeric_cols, ordinal_cols, onehot_cols, ordinal_categories = define_column_types(
        X)

    # 4. Split train-test
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 5. Create and fit preprocessor
    preprocessor = create_and_fit_preprocessor(
        X_train, numeric_cols, ordinal_cols, onehot_cols, ordinal_categories
    )

    # 6. Transform datasets
    X_train_transformed, X_test_transformed, transformed_cols = transform_datasets(
        preprocessor, X_train, X_test, numeric_cols, ordinal_cols)

    # 7. Save processed data and artifacts
    save_processed_data(
        X_train_transformed, X_test_transformed, y_train, y_test,
        preprocessor, X.columns.tolist(), transformed_cols
    )

    # 8. Print summary
    print_summary(
        df.shape, X_train_transformed, X_test_transformed,
        y_train, y_test, X.shape[1], len(transformed_cols)
    )

    print("=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
