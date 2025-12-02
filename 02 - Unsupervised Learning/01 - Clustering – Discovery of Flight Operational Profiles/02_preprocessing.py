#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
DATA PREPROCESSING - FLIGHT TELEMETRY DATA
==============================================================================

Purpose: Preprocess flight telemetry data for distance-based clustering algorithms.

This script:
1. Loads flight telemetry CSV data
2. Identifies numeric and categorical column types
3. Handles missing values using median/mode imputation
4. Encodes categorical features using label encoding
5. Applies standard scaling to prepare for clustering
6. Saves processed data and fitted scaler for reproducibility

Expected Input:
- CSV file with flight telemetry data (numeric and/or categorical features)

Outputs:
- processed_X.npy: Scaled feature matrix ready for clustering
- standard_scaler.pkl: Fitted scaler for transforming new data
- numeric_features.csv: List of numeric features
- categorical_features.csv: List of categorical features
- processed_data_preview.csv: Sample of processed data
- preprocessing_notes.md: Detailed preprocessing documentation

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuration Constants
SEED = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore')


class Config:
    """Preprocessing configuration parameters."""
    # Directories
    INPUT_DIR = Path('inputs')
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'
    DATA_PROCESSED_DIR = OUTPUT_DIR / 'data_processed'
    MODELS_DIR = OUTPUT_DIR / 'models'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'

    # Input file
    DATASET_PATH = INPUT_DIR / 'voos_telemetria_completa.csv'

    # Imputation strategies
    NUMERIC_IMPUTE_STRATEGY = 'median'
    CATEGORICAL_IMPUTE_STRATEGY = 'most_frequent'


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """
    Print formatted section header.

    Args:
        title: Section title text
        char: Character to use for separator line
    """
    print("\n" + char * 120)
    print(title)
    print(char * 120)


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the flight telemetry dataset from CSV file.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.absolute()}")

    df = pd.read_csv(path, encoding='utf-8')
    print("\n✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")

    return df


# ==============================================================================
# SECTION 4: COLUMN TYPE IDENTIFICATION
# ==============================================================================


def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns based on data types.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (numeric_columns, categorical_columns) lists
    """
    print_section("IDENTIFYING COLUMN TYPES", "-")

    numeric_cols = df.select_dtypes(
        include=[
            'int64',
            'int32',
            'float64',
            'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=[
            'object',
            'category',
            'bool']).columns.tolist()

    print(f"\nNumeric columns ({len(numeric_cols)}):")
    for col in numeric_cols:
        print(f"  • {col} ({df[col].dtype})")

    print(f"\nCategorical columns ({len(categorical_cols)}):")
    if categorical_cols:
        for col in categorical_cols:
            print(f"  • {col} ({df[col].dtype})")
    else:
        print("  • None found (all features are numeric)")

    return numeric_cols, categorical_cols


# ==============================================================================
# SECTION 5: MISSING VALUE HANDLING
# ==============================================================================


def handle_missing_values(df: pd.DataFrame, numeric_cols: List[str],
                          categorical_cols: List[str]) -> pd.DataFrame:
    """
    Handle missing values in numeric and categorical columns.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names

    Returns:
        DataFrame with imputed values
    """
    print_section("HANDLING MISSING VALUES", "-")

    df_imputed = df.copy()

    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()

    print(f"\nTotal missing values in dataset: {total_missing}")

    if total_missing > 0:
        print("\nMissing values per column:")
        for col in df.columns:
            if missing_summary[col] > 0:
                pct = (missing_summary[col] / len(df)) * 100
                print(f"  • {col}: {missing_summary[col]} ({pct:.2f}%)")

    if numeric_cols:
        numeric_missing = [
            col for col in numeric_cols if df[col].isnull().sum() > 0]

        if numeric_missing:
            print(
                f"\n→ Imputing numeric columns with {
                    Config.NUMERIC_IMPUTE_STRATEGY.upper()} strategy")
            imputer_numeric = SimpleImputer(
                strategy=Config.NUMERIC_IMPUTE_STRATEGY)
            df_imputed[numeric_missing] = imputer_numeric.fit_transform(
                df[numeric_missing])
            print(
                f"  ✓ Imputed {
                    len(numeric_missing)} numeric columns: {numeric_missing}")
        else:
            print("\n→ No missing values in numeric columns")

    if categorical_cols:
        categorical_missing = [
            col for col in categorical_cols if df[col].isnull().sum() > 0]

        if categorical_missing:
            print(
                f"\n→ Imputing categorical columns with {
                    Config.CATEGORICAL_IMPUTE_STRATEGY.upper()} strategy")
            imputer_categorical = SimpleImputer(
                strategy=Config.CATEGORICAL_IMPUTE_STRATEGY)
            df_imputed[categorical_missing] = imputer_categorical.fit_transform(
                df[categorical_missing])
            print(
                f"  ✓ Imputed {
                    len(categorical_missing)} categorical columns: {categorical_missing}")
        else:
            print("\n→ No missing values in categorical columns")

    total_after = df_imputed.isnull().sum().sum()
    print(f"\n✓ Missing values after imputation: {total_after}")

    return df_imputed


# ==============================================================================
# SECTION 6: CATEGORICAL ENCODING
# ==============================================================================


def encode_categorical_features(
        df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical features using label encoding.

    Args:
        df: Input DataFrame with imputed values
        categorical_cols: List of categorical column names

    Returns:
        Tuple of (encoded DataFrame, encoding information dictionary)
    """
    print_section("ENCODING CATEGORICAL FEATURES", "-")

    df_encoded = df.copy()
    encoding_info = {}

    if not categorical_cols:
        print("\n→ No categorical columns to encode")
        return df_encoded, encoding_info

    print(
        f"\nEncoding {
            len(categorical_cols)} categorical columns using LabelEncoder:")

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoding_info[col] = {
            'encoder': le,
            'classes': le.classes_.tolist(),
            'n_classes': len(le.classes_)
        }
        print(
            f"  • {col}: {len(le.classes_)} unique categories → [0, {len(le.classes_) - 1}]")
        print(f"    Categories: {le.classes_.tolist()}")

    print("\n✓ All categorical features encoded successfully")

    return df_encoded, encoding_info


# ==============================================================================
# SECTION 7: FEATURE SCALING
# ==============================================================================


def apply_standard_scaling(df: pd.DataFrame,
                           feature_cols: List[str],
                           models_dir: Path,
                           data_dir: Path) -> Tuple[np.ndarray,
                                                    StandardScaler]:
    """
    Apply standard scaling to features and save scaler.

    Args:
        df: DataFrame with all features (numeric and encoded categorical)
        feature_cols: List of all feature column names
        models_dir: Directory to save the scaler
        data_dir: Directory to save processed data

    Returns:
        Tuple of (scaled feature array, fitted scaler)
    """
    print_section("APPLYING STANDARD SCALING", "-")

    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nScaling statistics:")
    print(f"  Original data shape: {X.shape}")
    print(f"  Scaled data shape: {X_scaled.shape}")
    print(f"\n  Feature means (before scaling): {X.mean(axis=0)[:5]} ...")
    print(f"  Feature means (after scaling): {X_scaled.mean(axis=0)[:5]} ...")
    print(f"\n  Feature stds (before scaling): {X.std(axis=0)[:5]} ...")
    print(f"  Feature stds (after scaling): {X_scaled.std(axis=0)[:5]} ...")

    scaler_path = models_dir / 'standard_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n✓ Scaler saved to: {scaler_path}")

    data_path = data_dir / 'processed_X.npy'
    np.save(data_path, X_scaled)
    print(f"✓ Processed data saved to: {data_path}")

    return X_scaled, scaler


# ==============================================================================
# SECTION 8: EXPORT FUNCTIONS
# ==============================================================================


def save_feature_lists(
        numeric_cols: List[str],
        categorical_cols: List[str],
        tables_dir: Path) -> None:
    """
    Save lists of numeric and categorical features to CSV files.

    Args:
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        tables_dir: Directory to save the CSV files
    """
    print_section("SAVING FEATURE LISTS", "-")

    numeric_df = pd.DataFrame({'feature': numeric_cols, 'type': 'numeric'})
    numeric_path = tables_dir / 'numeric_features.csv'
    numeric_df.to_csv(numeric_path, index=False, encoding='utf-8')
    print(f"✓ Saved numeric features to: {numeric_path}")

    categorical_df = pd.DataFrame(
        {'feature': categorical_cols, 'type': 'categorical'})
    categorical_path = tables_dir / 'categorical_features.csv'
    categorical_df.to_csv(categorical_path, index=False, encoding='utf-8')
    print(f"✓ Saved categorical features to: {categorical_path}")


def save_processed_preview(X_scaled: np.ndarray, feature_cols: List[str],
                           tables_dir: Path, n_rows: int = 10) -> None:
    """
    Save a preview of the processed data to CSV.

    Args:
        X_scaled: Scaled feature array
        feature_cols: List of feature column names
        tables_dir: Directory to save the preview
        n_rows: Number of rows to include in preview
    """
    print_section("SAVING PROCESSED DATA PREVIEW", "-")

    preview_df = pd.DataFrame(X_scaled[:n_rows], columns=feature_cols)
    preview_path = tables_dir / 'processed_data_preview.csv'
    preview_df.to_csv(preview_path, index=False, encoding='utf-8')
    print(f"✓ Saved preview ({n_rows} rows) to: {preview_path}")


def create_preprocessing_notes(df_original: pd.DataFrame,
                               numeric_cols: List[str],
                               categorical_cols: List[str],
                               encoding_info: Dict[str,
                                                   Any],
                               logs_dir: Path) -> None:
    """
    Generate detailed preprocessing notes in Markdown format.

    Args:
        df_original: Original DataFrame before preprocessing
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        encoding_info: Dictionary with encoding information
        logs_dir: Directory to save the notes
    """
    print_section("GENERATING PREPROCESSING NOTES", "-")

    notes_path = logs_dir / 'preprocessing_notes.md'

    with open(notes_path, 'w', encoding='utf-8') as f:
        f.write("# Data Preprocessing Notes\n\n")
        f.write("## 1. Dataset Overview\n\n")
        f.write(
            f"**Original shape:** {
                df_original.shape[0]} rows × {
                df_original.shape[1]} columns\n\n")

        f.write("**Column types:**\n")
        f.write(f"- Numeric features: {len(numeric_cols)}\n")
        f.write(f"- Categorical features: {len(categorical_cols)}\n\n")

        f.write("## 2. Missing Value Analysis\n\n")
        missing_summary = df_original.isnull().sum()
        total_missing = missing_summary.sum()

        if total_missing > 0:
            f.write(
                f"**Total missing values:** {total_missing} ({(total_missing / df_original.size * 100):.2f}%)\n\n")
            f.write("**Missing values by column:**\n\n")
            f.write("| Column | Missing Count | Percentage |\n")
            f.write("|--------|---------------|------------|\n")
            for col in df_original.columns:
                if missing_summary[col] > 0:
                    pct = (missing_summary[col] / len(df_original)) * 100
                    f.write(
                        f"| {col} | {
                            missing_summary[col]} | {
                            pct:.2f}% |\n")
            f.write("\n")
        else:
            f.write("**No missing values detected in the dataset.**\n\n")

        f.write("**Imputation strategy:**\n")
        f.write(f"- Numeric columns: {Config.NUMERIC_IMPUTE_STRATEGY}\n")
        f.write(
            f"- Categorical columns: {Config.CATEGORICAL_IMPUTE_STRATEGY}\n\n")

        f.write("## 3. Categorical Encoding\n\n")

        if categorical_cols:
            f.write("**Encoding method:** Label Encoding\n\n")
            f.write("**Encoded features:**\n\n")
            for col in categorical_cols:
                if col in encoding_info:
                    info = encoding_info[col]
                    f.write(f"### {col}\n")
                    f.write(
                        f"- Number of unique categories: {info['n_classes']}\n")
                    f.write(f"- Categories: {info['classes']}\n")
                    f.write(
                        f"- Encoded range: [0, {info['n_classes'] - 1}]\n\n")
        else:
            f.write("**No categorical features to encode.**\n\n")

        f.write("## 4. Why Standard Scaling is MANDATORY for K-Means\n\n")
        f.write("### The Problem: Feature Scale Dominance\n\n")

        scales = {}
        for col in numeric_cols:
            scales[col] = {
                'mean': df_original[col].mean(),
                'std': df_original[col].std(),
                'range': df_original[col].max() - df_original[col].min()
            }

        f.write("**Original feature scales:**\n\n")
        f.write("| Feature | Mean | Std Dev | Range |\n")
        f.write("|---------|------|---------|-------|\n")
        for col, stats in sorted(
                scales.items(), key=lambda x: x[1]['std'], reverse=True)[
                :10]:
            f.write(
                f"| {col} | {
                    stats['mean']:.2f} | {
                    stats['std']:.2f} | {
                    stats['range']:.2f} |\n")
        f.write("\n")

        max_std = max([s['std'] for s in scales.values()])
        min_std = min([s['std'] for s in scales.values()])
        scale_ratio = max_std / min_std if min_std > 0 else np.inf

        f.write(f"**Scale ratio (max std / min std):** {scale_ratio:.2f}x\n\n")

        f.write("### Why This Matters:\n\n")
        f.write("**K-Means uses Euclidean distance:**\n")
        f.write("```\n")
        f.write("distance = sqrt((x1-y1)² + (x2-y2)² + ... + (xn-yn)²)\n")
        f.write("```\n\n")

        f.write("**Without scaling:**\n")
        f.write(
            "- Features with larger scales (higher variance) dominate the distance calculation\n")
        f.write(
            "- Example: If altitude ranges 0-10,000m and duration 0-180min, altitude dominates\n")
        f.write("- Clusters form primarily based on high-variance features\n")
        f.write(
            "- Low-variance features become essentially invisible to the algorithm\n")
        f.write(
            "- This violates the assumption that all features contribute equally\n\n")

        f.write("**With StandardScaler:**\n")
        f.write("- Transforms each feature to have mean=0 and std=1\n")
        f.write("- Formula: `z = (x - mean) / std`\n")
        f.write("- All features now operate on the same scale\n")
        f.write("- Each feature contributes proportionally to distance calculations\n")
        f.write("- Centroids are not biased toward high-variance features\n\n")

        f.write("## 5. Impact of Outliers on Clustering\n\n")

        f.write("### How Outliers Affect K-Means:\n\n")
        f.write("1. **Centroid distortion:**\n")
        f.write("   - K-Means uses arithmetic mean to compute centroids\n")
        f.write("   - Outliers pull centroids away from the true cluster center\n")
        f.write("   - This can cause misclassification of normal points\n\n")

        f.write("2. **Cluster fragmentation:**\n")
        f.write("   - Extreme outliers may form their own singleton clusters\n")
        f.write("   - Reduces the number of meaningful clusters\n")
        f.write("   - Wastes computational resources on noise points\n\n")

        f.write("3. **Distance inflation:**\n")
        f.write("   - Outliers increase within-cluster sum of squares (WCSS)\n")
        f.write("   - Makes elbow plots harder to interpret\n")
        f.write("   - Can lead to selecting suboptimal k values\n\n")

        f.write("### Mitigation Strategies:\n\n")
        f.write("**Option 1: Robust Scaling**\n")
        f.write("- Use `RobustScaler` instead of `StandardScaler`\n")
        f.write("- Centers using median, scales using IQR (Q3 - Q1)\n")
        f.write("- Less sensitive to extreme values\n")
        f.write("- Formula: `z = (x - median) / IQR`\n\n")

        f.write("**Option 2: Outlier Removal**\n")
        f.write("- Use IQR method: remove points beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR\n")
        f.write("- Or use statistical tests (Z-score > 3, modified Z-score)\n")
        f.write("- Risk: May remove legitimate rare patterns\n")
        f.write("- Decision: Review outliers manually before removal\n\n")

        f.write("**Option 3: Capping/Winsorizing**\n")
        f.write("- Cap extreme values at specific percentiles (e.g., 1st and 99th)\n")
        f.write("- Preserves all data points but limits extreme influence\n")
        f.write("- Good compromise between retention and robustness\n\n")

        f.write("**Option 4: Use DBSCAN Instead**\n")
        f.write("- Density-based clustering naturally handles outliers\n")
        f.write("- Outliers classified as noise (cluster label = -1)\n")
        f.write("- No need for outlier removal preprocessing\n")
        f.write("- Better for datasets with significant noise\n\n")

        f.write("## 6. Preprocessing Pipeline Summary\n\n")
        f.write("**Steps completed:**\n")
        f.write("1. ✓ Load raw data\n")
        f.write("2. ✓ Identify numeric vs categorical features\n")
        f.write("3. ✓ Handle missing values (median/mode imputation)\n")
        f.write("4. ✓ Encode categorical features (if present)\n")
        f.write("5. ✓ Apply StandardScaler to all features\n")
        f.write("6. ✓ Save processed data and scaler for reproducibility\n\n")

        f.write("**Outputs generated:**\n")
        f.write("- `processed_X.npy` - Scaled feature matrix ready for clustering\n")
        f.write("- `standard_scaler.pkl` - Fitted scaler for transforming new data\n")
        f.write("- `numeric_features.csv` - List of numeric features\n")
        f.write("- `categorical_features.csv` - List of categorical features\n")
        f.write("- `processed_data_preview.csv` - Sample of processed data\n\n")

        f.write("**Next steps:**\n")
        f.write(
            "1. **Dimensionality reduction:** Apply PCA for visualization and noise reduction\n")
        f.write(
            "2. **Optimal k selection:** Use elbow method, silhouette analysis, gap statistic\n")
        f.write("3. **Clustering:** Apply K-Means, DBSCAN, Hierarchical clustering\n")
        f.write(
            "4. **Evaluation:** Assess cluster quality using internal validation metrics\n")
        f.write(
            "5. **Interpretation:** Analyze cluster characteristics and business meaning\n\n")

        f.write("---\n\n")
        f.write("**End of Preprocessing Notes**\n")

    print(f"\n✓ Preprocessing notes saved to: {notes_path}")


# ==============================================================================
# SECTION 9: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the entire preprocessing workflow."""
    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    Config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("DATA PREPROCESSING - FLIGHT TELEMETRY DATA")
    print("=" * 120)
    print(f"\nRandom seed: {SEED}")
    print(f"Dataset path: {Config.DATASET_PATH.absolute()}")

    print_section("CREATING OUTPUT DIRECTORIES")
    print(f"✓ Output directory: {Config.OUTPUT_DIR.absolute()}")
    print(f"✓ Graphics directory: {Config.GRAPHICS_DIR.absolute()}")
    print(
        f"✓ Data processed directory: {
            Config.DATA_PROCESSED_DIR.absolute()}")
    print(f"✓ Models directory: {Config.MODELS_DIR.absolute()}")
    print(f"✓ Predictions directory: {Config.PREDICTIONS_DIR.absolute()}")

    try:
        # 1. Load dataset
        print_section("STEP 1: LOAD DATASET")
        df = load_dataset(Config.DATASET_PATH)
        df_original = df.copy()

        # 2. Identify column types
        print_section("STEP 2: IDENTIFY COLUMN TYPES")
        numeric_cols, categorical_cols = identify_column_types(df)

        # 3. Handle missing values
        print_section("STEP 3: HANDLE MISSING VALUES")
        df_imputed = handle_missing_values(df, numeric_cols, categorical_cols)

        # 4. Encode categorical features
        print_section("STEP 4: ENCODE CATEGORICAL FEATURES")
        df_encoded, encoding_info = encode_categorical_features(
            df_imputed, categorical_cols)

        # 5. Save feature lists
        print_section("STEP 5: SAVE FEATURE LISTS")
        save_feature_lists(
            numeric_cols,
            categorical_cols,
            Config.DATA_PROCESSED_DIR)

        # 6. Apply standard scaling
        print_section("STEP 6: APPLY STANDARD SCALING")
        all_feature_cols = numeric_cols + categorical_cols
        X_scaled, scaler = apply_standard_scaling(
            df_encoded, all_feature_cols, Config.MODELS_DIR, Config.DATA_PROCESSED_DIR)

        # 7. Save processed data preview
        print_section("STEP 7: SAVE PROCESSED DATA PREVIEW")
        save_processed_preview(
            X_scaled,
            all_feature_cols,
            Config.DATA_PROCESSED_DIR,
            n_rows=10)

        # 8. Create preprocessing notes
        print_section("STEP 8: CREATE PREPROCESSING NOTES")
        create_preprocessing_notes(
            df_original,
            numeric_cols,
            categorical_cols,
            encoding_info,
            Config.OUTPUT_DIR)

        # Final summary
        print_section("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("\nOUTPUTS GENERATED:")
        print("\nDATA PROCESSED:")
        print(f"  • {Config.DATA_PROCESSED_DIR / 'processed_X.npy'}")
        print(f"  • {Config.DATA_PROCESSED_DIR / 'numeric_features.csv'}")
        print(f"  • {Config.DATA_PROCESSED_DIR / 'categorical_features.csv'}")
        print(
            f"  • {
                Config.DATA_PROCESSED_DIR /
                'processed_data_preview.csv'}")
        print("\nMODELS:")
        print(f"  • {Config.MODELS_DIR / 'standard_scaler.pkl'}")
        print("\nNOTES:")
        print(f"  • {Config.OUTPUT_DIR / 'preprocessing_notes.md'}")

        print("\nPROCESSED DATA SHAPE:")
        print(f"  Samples: {X_scaled.shape[0]}")
        print(f"  Features: {X_scaled.shape[1]}")

        print("\n" + "=" * 120)
        print("✓ Data is now ready for clustering algorithms!")
        print("=" * 120)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
