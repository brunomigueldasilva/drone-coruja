#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
TIME SERIES FORECASTING - DATA PREPROCESSING AND FEATURE ENGINEERING
==============================================================================

Purpose: Prepare time series data for forecasting with proper temporal split

This script:
1. Loads raw voltage (minute-level) and missions (daily) datasets
2. Creates calendar features (month, day_of_week, hour, is_weekend)
3. Creates lag features (past values that inform future predictions)
4. Creates rolling window features (moving averages and standard deviations)
5. Performs TEMPORAL train-test split (NO shuffle - critical for time series!)
6. Scales features using StandardScaler (fit only on train to prevent leakage)
7. Saves preprocessed data and artifacts for model training

CRITICAL CONCEPTS FOR TIME SERIES:
==================================

1. TEMPORAL SPLIT (Not Random Split):
   - In time series, we MUST split chronologically (train = past, test = future)
   - WHY? Because we're predicting the FUTURE based on the PAST
   - Random split would let the model "see the future" during training
   - This causes DATA LEAKAGE and inflates performance metrics artificially

2. DATA LEAKAGE:
   - Using information from the future to predict the past
   - Example: If 2024 data is in training and 2023 data is in test, the model
     has already "seen" what happens after the test period
   - This makes the model appear accurate when it would fail in production

3. SCALING ON TRAIN ONLY:
   - We fit the scaler ONLY on training data
   - WHY? Test data represents "unseen future" that we don't have yet
   - If we fit on test data, we're using future statistics to transform past
   - In production, we only have training data when deploying the model

4. LAG FEATURES:
   - Use past values to predict future values (e.g., yesterday predicts today)
   - This is the core intuition: "history repeats itself"
   - We create NaN values at the start (no history available yet)

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from typing import Tuple, List, Any

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Preprocessing configuration parameters."""
    # Input files (raw data)
    VOLTAGE_FILE = Path('inputs/voltagem_bateria.csv')
    MISSIONS_FILE = Path('inputs/missoes_diarias.csv')

    # Output directories
    PROCESSED_DATA_DIR = Path('outputs/data_processed')
    MODELS_DIR = Path('outputs/models')

    # Split configuration
    TEST_SIZE = 0.20  # Last 20% of data as test (most recent)

    # Target columns
    VOLTAGE_TARGET = 'voltagem'
    MISSIONS_TARGET = 'num_missoes'


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


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_time_series_data(filepath: Path, time_col: str,
                          target_col: str) -> pd.DataFrame:
    """
    Load time series dataset and prepare datetime index.

    Args:
        filepath: Path to CSV file
        time_col: Name of the time column
        target_col: Name of the target variable column

    Returns:
        pd.DataFrame: Loaded data with datetime index

    Raises:
        SystemExit: If file not found or validation fails
    """
    # Check if file exists
    if not filepath.exists():
        print(f"✗ ERROR: {filepath} not found!")
        print(
            f"  Please ensure the file exists in the '{
                filepath.parent}' directory.")
        exit(1)

    # Load dataset
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded: {filepath.name}")
    print(f"  Initial shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Convert time column to datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # Sort by time (critical for time series!)
    df = df.sort_values(time_col).reset_index(drop=True)

    # Set datetime index
    df.set_index(time_col, inplace=True)

    # Verify target column exists
    if target_col not in df.columns:
        print(
            f"✗ ERROR: Target column '{target_col}' not found in {
                filepath.name}!")
        exit(1)

    print(f"  Time range: {df.index.min()} to {df.index.max()}")
    print(f"  Target column: '{target_col}'")

    return df


# ==============================================================================
# SECTION 4: CALENDAR FEATURES
# ==============================================================================


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features from datetime index.

    Calendar features capture seasonal patterns and cyclical behavior:
    - Month: Annual seasonality (e.g., winter vs summer)
    - Day of week: Weekly patterns (e.g., weekday vs weekend)
    - Day of month: Monthly patterns (e.g., beginning vs end of month)
    - Hour: Daily patterns (e.g., peak hours vs off-hours)
    - Is weekend: Binary indicator for weekend days

    These features help the model learn recurring temporal patterns.

    Args:
        df: DataFrame with datetime index

    Returns:
        pd.DataFrame: DataFrame with added calendar features
    """
    print("\nCreating calendar features...")

    # Month (1-12): Captures annual seasonality
    df['month'] = df.index.month
    print("  ✓ month (1-12)")

    # Day of week (0=Monday, 6=Sunday): Captures weekly patterns
    df['day_of_week'] = df.index.dayofweek
    print("  ✓ day_of_week (0=Monday, 6=Sunday)")

    # Day of month (1-31): Captures monthly patterns
    df['day_of_month'] = df.index.day
    print("  ✓ day_of_month (1-31)")

    # Hour (0-23): Only for minute-level data
    # Check if index has hour information (frequency < 1 day)
    if hasattr(df.index, 'hour'):
        # Check if hours vary (not all zeros for daily data)
        if df.index.hour.nunique() > 1:
            df['hour'] = df.index.hour
            print("  ✓ hour (0-23)")

    # Is weekend (1=Saturday/Sunday, 0=weekday): Binary weekend indicator
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    print("  ✓ is_weekend (binary)")

    return df


# ==============================================================================
# SECTION 5: LAG FEATURES
# ==============================================================================


def create_lag_features(df: pd.DataFrame, target_col: str,
                        lags: List[int]) -> pd.DataFrame:
    """
    Create lag features (past values of the target variable).

    LAG FEATURES INTUITION:
    - Lag features use past values to predict future values
    - Example: lag_1 = value from 1 period ago (yesterday, last minute, etc.)
    - This captures the autocorrelation: "today's value depends on yesterday's"
    - Multiple lags capture different time scales (recent vs distant past)

    WHY LAGS CREATE NaN:
    - The first few rows have no history available
    - Example: lag_7 for row 5 tries to look 7 periods back (doesn't exist)
    - We'll drop these NaN rows later (acceptable loss for better features)

    CHOOSING LAG PERIODS:
    - Voltage (minute-level): lags 1, 5, 10, 60 capture recent fluctuations
      (1 min ago, 5 min ago, 10 min ago, 1 hour ago)
    - Missions (daily): lags 1, 7, 14, 28 capture weekly/monthly patterns
      (yesterday, last week, 2 weeks ago, 4 weeks ago)

    Args:
        df: DataFrame with target variable
        target_col: Name of target column to create lags from
        lags: List of lag periods (e.g., [1, 7, 14])

    Returns:
        pd.DataFrame: DataFrame with added lag features
    """
    print(f"\nCreating lag features for '{target_col}'...")
    print(f"  Lag periods: {lags}")

    for lag in lags:
        # Shift the target variable back by 'lag' periods
        # This creates a new column containing the value from 'lag' periods ago
        df[f'lag_{lag}'] = df[target_col].shift(lag)
        print(f"  ✓ lag_{lag} (value from {lag} periods ago)")

    return df


# ==============================================================================
# SECTION 6: ROLLING WINDOW FEATURES
# ==============================================================================


def create_rolling_features(df: pd.DataFrame, target_col: str,
                            windows: List[int]) -> pd.DataFrame:
    """
    Create rolling window statistics (moving averages and standard deviations).

    ROLLING FEATURES INTUITION:
    - Rolling mean: Average of last N periods (smooths out noise)
    - Rolling std: Volatility/variability of last N periods
    - These capture short-term trends and stability patterns

    EXAMPLE (window=7):
    - rolling_mean_7: Average of last 7 days (smoothed recent trend)
    - rolling_std_7: How volatile the last 7 days were

    WHY ROLLING WINDOWS ARE USEFUL:
    - Mean captures recent trend direction (going up or down)
    - Std captures recent volatility (stable or erratic)
    - Together they provide context beyond single point values

    Args:
        df: DataFrame with target variable
        target_col: Name of target column to compute rolling stats from
        windows: List of window sizes (e.g., [7, 14])

    Returns:
        pd.DataFrame: DataFrame with added rolling features
    """
    print(f"\nCreating rolling window features for '{target_col}'...")
    print(f"  Window sizes: {windows}")

    for window in windows:
        # Rolling mean: average of last 'window' periods
        df[f'rolling_mean_{window}'] = df[target_col].rolling(
            window=window, min_periods=1).mean()
        print(f"  ✓ rolling_mean_{window} (avg of last {window} periods)")

        # Rolling std: standard deviation of last 'window' periods
        df[f'rolling_std_{window}'] = df[target_col].rolling(
            window=window, min_periods=1).std()
        print(f"  ✓ rolling_std_{window} (std of last {window} periods)")

    return df


# ==============================================================================
# SECTION 7: TEMPORAL TRAIN-TEST SPLIT
# ==============================================================================


def temporal_train_test_split(X: pd.DataFrame, y: pd.Series,
                              test_size: float = 0.2) -> Tuple:
    """
    Perform temporal (chronological) train-test split.

    *** CRITICAL: WHY TEMPORAL SPLIT IS ESSENTIAL FOR TIME SERIES ***

    WRONG APPROACH (Random Split):
    ==============================
    Using train_test_split(shuffle=True) or random sampling:

    Timeline: [2020] [2021] [2022] [2023] [2024]
    Random:   Train  Test   Train  Train  Test

    PROBLEM: Model trains on 2023-2024 data, then tested on 2021-2022
    - The model has "seen the future" (2023-2024) when predicting the past
    - This is DATA LEAKAGE: using future information to predict the past
    - Metrics will be artificially high (model appears accurate but isn't)
    - In production, the model will FAIL because it can't see the future

    CORRECT APPROACH (Temporal Split):
    ==================================
    Chronological split: train = past, test = future

    Timeline: [2020] [2021] [2022] [2023] [2024]
    Temporal: Train  Train  Train  Train  Test

    CORRECT: Model trains ONLY on past (2020-2023), tested on future (2024)
    - Simulates real-world scenario: predict future based on past
    - No data leakage: model never sees test period during training
    - Realistic performance metrics that will generalize to production
    - This is how forecasting actually works in practice

    REAL-WORLD ANALOGY:
    ===================
    Imagine predicting tomorrow's stock price:
    - WRONG: Train on data from next week, test on data from last week
      (You'd be rich if you could see the future!)
    - CORRECT: Train on last year's data, predict tomorrow
      (This is what we actually have available)

    Args:
        X: Feature matrix (DataFrame with datetime index)
        y: Target vector (Series with datetime index)
        test_size: Fraction of most recent data for test set (default: 0.2)

    Returns:
        Tuple: X_train, X_test, y_train, y_test (chronologically split)
    """
    print_section("TEMPORAL TRAIN-TEST SPLIT (NO SHUFFLE!)")

    print("\n⚠️  IMPORTANT: Using TEMPORAL split (not random split)")
    print("   - Random split would cause DATA LEAKAGE in time series")
    print("   - We NEVER use train_test_split() with shuffle=True")
    print("   - Test set MUST be the most recent data (future)")

    # Calculate split point (80% for train, 20% for test)
    n_samples = len(X)
    split_idx = int(n_samples * (1 - test_size))

    # Split chronologically: first 80% = train, last 20% = test
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    # Extract date ranges for reporting
    train_start = X_train.index.min()
    train_end = X_train.index.max()
    test_start = X_test.index.min()
    test_end = X_test.index.max()

    print(f"\n✓ Split completed (test_size={test_size}):")
    print("\n  TRAINING SET (Past Data - Used to Learn Patterns):")
    print(f"    - Samples: {len(X_train):,} ({(1 - test_size) * 100:.0f}%)")
    print(f"    - Date range: {train_start} to {train_end}")
    print(f"    - Features: {X_train.shape[1]}")

    print("\n  TEST SET (Future Data - Used to Evaluate):")
    print(f"    - Samples: {len(X_test):,} ({test_size * 100:.0f}%)")
    print(f"    - Date range: {test_start} to {test_end}")
    print(f"    - Features: {X_test.shape[1]}")

    # Verify no overlap (critical check)
    if train_end >= test_start:
        print("\n  ⚠️  WARNING: Train and test periods overlap!")
        print("     This should not happen with proper temporal split.")
    else:
        print("\n  ✓ No overlap: train ends before test starts")
        print(f"    Gap: {(test_start - train_end).days} days" if hasattr(
            test_start - train_end, 'days') else "")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# SECTION 8: FEATURE SCALING
# ==============================================================================


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> Tuple[np.ndarray,
                                                  np.ndarray,
                                                  StandardScaler]:
    """
    Scale features using StandardScaler (fit on train only).

    *** CRITICAL: WHY WE FIT SCALER ONLY ON TRAINING DATA ***

    STANDARDIZATION:
    ===============
    StandardScaler transforms features to have:
    - Mean = 0 (centered)
    - Standard deviation = 1 (normalized)

    Formula: X_scaled = (X - mean) / std

    DATA LEAKAGE IN SCALING:
    ========================

    WRONG APPROACH (Fit on All Data):
    ---------------------------------
    scaler.fit(pd.concat([X_train, X_test]))  # ❌ WRONG!

    PROBLEM: We're using future statistics (test mean/std) to scale past data
    - Test set represents "unseen future" that we don't have in production
    - Using test statistics is a form of data leakage
    - Inflates performance because test data characteristics leak into training

    CORRECT APPROACH (Fit on Train Only):
    -------------------------------------
    scaler.fit(X_train)  # ✓ CORRECT! Fit only on training data
    X_train_scaled = scaler.transform(X_train)  # Scale train using train stats
    X_test_scaled = scaler.transform(X_test)    # Scale test using train stats

    WHY THIS IS CORRECT:
    - In production, we only have historical data (training set)
    - We compute mean/std from historical data
    - We apply those same statistics to new incoming data (test/future)
    - This simulates real-world deployment scenario

    REAL-WORLD ANALOGY:
    ===================
    Imagine normalizing temperatures:
    - WRONG: Use mean of all months (including future) to normalize January
    - CORRECT: Use historical mean (past years) to normalize current month

    PRODUCTION SCENARIO:
    ====================
    When deploying the model:
    1. We save the scaler fitted on historical data (train)
    2. New data arrives (like test data)
    3. We transform new data using the saved scaler (historical stats)
    4. This is exactly what we're simulating with fit(train), transform(test)

    Args:
        X_train: Training features (DataFrame)
        X_test: Test features (DataFrame)

    Returns:
        Tuple: X_train_scaled (array), X_test_scaled (array), fitted_scaler
    """
    print_section("FEATURE SCALING (FIT ON TRAIN ONLY)")

    print("\n⚠️  IMPORTANT: Preventing data leakage in scaling")
    print("   - Scaler is fit ONLY on training data")
    print("   - Test data is transformed using TRAINING statistics")
    print("   - This simulates real-world production scenario")

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit scaler ONLY on training data
    # This computes mean and std from training set only
    print("\n1. Fitting StandardScaler on TRAINING data only...")
    scaler.fit(X_train)
    print("   ✓ Scaler fitted (computed mean and std from training set)")

    # Transform training data using training statistics
    print("\n2. Transforming TRAINING data...")
    X_train_scaled = scaler.transform(X_train)
    print(f"   ✓ X_train scaled: {X_train_scaled.shape}")
    print("   - Each feature now has mean ≈ 0, std ≈ 1")

    # Transform test data using TRAINING statistics (critical!)
    print("\n3. Transforming TEST data (using training statistics)...")
    X_test_scaled = scaler.transform(X_test)
    print(f"   ✓ X_test scaled: {X_test_scaled.shape}")
    print("   - Scaled using training mean/std (no leakage)")

    print("\n✓ Feature scaling completed successfully")
    print("  - No data leakage: test data never influenced scaling")
    print("  - Scaler can be saved and reused for production deployment")

    return X_train_scaled, X_test_scaled, scaler


# ==============================================================================
# SECTION 9: PREPROCESSING PIPELINE FOR SINGLE DATASET
# ==============================================================================


def preprocess_dataset(filepath: Path, time_col: str, target_col: str,
                       lags: List[int], rolling_windows: List[int],
                       dataset_name: str) -> Tuple:
    """
    Complete preprocessing pipeline for a single time series dataset.

    Pipeline steps:
    1. Load data and prepare datetime index
    2. Create calendar features (month, day_of_week, etc.)
    3. Create lag features (past values)
    4. Create rolling window features (moving stats)
    5. Remove rows with NaN (from lag features)
    6. Separate features (X) from target (y)
    7. Perform temporal train-test split
    8. Scale features (fit on train only)
    9. Save all artifacts

    Args:
        filepath: Path to CSV file
        time_col: Name of time column
        target_col: Name of target column
        lags: List of lag periods to create
        rolling_windows: List of rolling window sizes
        dataset_name: Name for saving files (e.g., "voltage")

    Returns:
        Tuple: All preprocessed data and artifacts
    """
    print_section(f"PREPROCESSING: {dataset_name.upper()}")

    # 1. Load data
    print("\n1. Loading data...")
    df = load_time_series_data(filepath, time_col, target_col)
    print(f"   Initial shape: {df.shape}")

    # 2. Create calendar features
    print("\n2. Creating calendar features...")
    df = create_calendar_features(df)

    # 3. Create lag features
    print("\n3. Creating lag features...")
    df = create_lag_features(df, target_col, lags)

    # 4. Create rolling window features
    print("\n4. Creating rolling window features...")
    df = create_rolling_features(df, target_col, rolling_windows)

    # 5. Remove NaN values (from lag features)
    print("\n5. Removing rows with NaN values...")
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    print(
        f"   Rows removed: {
            rows_removed:,} ({
            rows_removed / rows_before * 100:.1f}%)")
    print(f"   Rows remaining: {rows_after:,}")
    print("   ✓ Dataset ready for modeling")

    # 6. Separate features (X) from target (y)
    print("\n6. Separating features (X) and target (y)...")
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    print(f"   Features (X): {X.shape}")
    print(f"   Target (y): {y.shape}")

    # List feature names
    print(f"\n   Feature columns ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"     {i:2d}. {col}")

    # 7. Temporal train-test split
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X, y, test_size=Config.TEST_SIZE)

    # 8. Feature scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 9. Save artifacts
    print_section(f"SAVING PREPROCESSED DATA: {dataset_name.upper()}")
    save_preprocessed_data(
        X_train_scaled, X_test_scaled, y_train, y_test,
        scaler, X.columns.tolist(), dataset_name
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


# ==============================================================================
# SECTION 10: SAVE PREPROCESSED DATA
# ==============================================================================


def save_preprocessed_data(
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series,
        scaler: StandardScaler,
        feature_names: List[str],
        dataset_name: str) -> None:
    """
    Save all preprocessed data and artifacts to disk.

    Files saved:
    - X_train_scaled.pkl: Scaled training features (numpy array)
    - X_test_scaled.pkl: Scaled test features (numpy array)
    - y_train.pkl: Training target (pandas Series)
    - y_test.pkl: Test target (pandas Series)
    - scaler.pkl: Fitted StandardScaler (for production deployment)
    - feature_names.pkl: List of feature column names

    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training target
        y_test: Test target
        scaler: Fitted StandardScaler
        feature_names: List of feature column names
        dataset_name: Name prefix for files (e.g., "voltage")
    """
    # Create output directory
    output_dir = Config.PROCESSED_DATA_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir.absolute()}")

    # Save scaled features
    print("\nSaving scaled features...")
    size = save_pickle(X_train_scaled, output_dir / 'X_train_scaled.pkl')
    print(f"  ✓ X_train_scaled.pkl ({size})")

    size = save_pickle(X_test_scaled, output_dir / 'X_test_scaled.pkl')
    print(f"  ✓ X_test_scaled.pkl ({size})")

    # Save target variables
    print("\nSaving target variables...")
    size = save_pickle(y_train, output_dir / 'y_train.pkl')
    print(f"  ✓ y_train.pkl ({size})")

    size = save_pickle(y_test, output_dir / 'y_test.pkl')
    print(f"  ✓ y_test.pkl ({size})")

    # Save scaler (for production deployment)
    print("\nSaving scaler (for production)...")
    size = save_pickle(scaler, output_dir / 'scaler.pkl')
    print(f"  ✓ scaler.pkl ({size})")
    print("     This scaler can be loaded and used to transform new data")

    # Save feature names
    print("\nSaving feature names...")
    size = save_pickle(feature_names, output_dir / 'feature_names.pkl')
    print(f"  ✓ feature_names.pkl ({size})")

    print(f"\n✓ All files saved to: {output_dir}/")


# ==============================================================================
# SECTION 11: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main function orchestrating preprocessing for both datasets.

    Workflow:
    1. Preprocess voltage dataset (minute-level data)
    2. Preprocess missions dataset (daily data)
    3. Print summary of all preprocessing steps

    Each dataset gets:
    - Calendar features (month, day_of_week, hour, is_weekend)
    - Lag features (appropriate for data frequency)
    - Rolling window features (mean and std)
    - Temporal train-test split (no shuffle)
    - Feature scaling (fit on train only)
    - All artifacts saved for model training
    """
    print("=" * 80)
    print("TIME SERIES PREPROCESSING - VOLTAGE AND MISSIONS DATA")
    print("=" * 80)
    print("\nThis script performs feature engineering with proper temporal handling")
    print("to avoid data leakage and ensure realistic model evaluation.")

    # Create output directories
    Config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # DATASET 1: VOLTAGE (MINUTE-LEVEL)
    # ========================================================================

    voltage_lags = [1, 5, 10, 60]  # 1 min, 5 min, 10 min, 1 hour
    voltage_windows = [7, 30]       # 7 and 30 period moving averages

    preprocess_dataset(
        filepath=Config.VOLTAGE_FILE,
        time_col='timestamp',
        target_col=Config.VOLTAGE_TARGET,
        lags=voltage_lags,
        rolling_windows=voltage_windows,
        dataset_name='voltage'
    )

    # ========================================================================
    # DATASET 2: MISSIONS (DAILY)
    # ========================================================================

    missions_lags = [1, 7, 14, 28]  # 1 day, 1 week, 2 weeks, 4 weeks
    missions_windows = [7, 14]       # 7 and 14 day moving averages

    preprocess_dataset(
        filepath=Config.MISSIONS_FILE,
        time_col='data',
        target_col=Config.MISSIONS_TARGET,
        lags=missions_lags,
        rolling_windows=missions_windows,
        dataset_name='missions'
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print_section("PREPROCESSING SUMMARY")

    summary = """
✓ Preprocessing completed successfully for both datasets!

DATASETS PROCESSED:
==================
1. Voltage (minute-level time series)
   - Lags: 1, 5, 10, 60 (minutes)
   - Rolling windows: 7, 30 (periods)
   - Target: voltagem

2. Missions (daily time series)
   - Lags: 1, 7, 14, 28 (days)
   - Rolling windows: 7, 14 (periods)
   - Target: num_missoes

FEATURES CREATED:
=================
For each dataset:
✓ Calendar features (month, day_of_week, day_of_month, hour, is_weekend)
✓ Lag features (past values of target variable)
✓ Rolling window features (moving averages and standard deviations)

CRITICAL SAFEGUARDS IMPLEMENTED:
=================================
✓ TEMPORAL SPLIT: Test set is the most recent 20% of data
  - No shuffle! Chronological split ensures no data leakage
  - Simulates real-world forecasting scenario (predict future from past)

✓ SCALING ON TRAIN ONLY: StandardScaler fitted only on training data
  - Test data transformed using training statistics
  - Prevents data leakage through feature scaling
  - Matches production deployment scenario

✓ NaN REMOVAL: Rows with missing values from lag features removed
  - Ensures complete feature sets for modeling
  - Acceptable trade-off for better features

FILES SAVED:
============
For each dataset (voltage and missions):
  - X_train_scaled.pkl (scaled training features)
  - X_test_scaled.pkl (scaled test features)
  - y_train.pkl (training target)
  - y_test.pkl (test target)
  - scaler.pkl (fitted StandardScaler for production)
  - feature_names.pkl (list of feature column names)

Location: outputs/data_processed/voltage/ and outputs/data_processed/missions/

NEXT STEPS:
===========
1. Train forecasting models (ARIMA, SARIMA, Prophet, LSTM, etc.)
2. Evaluate on test set (realistic future performance)
3. Use saved scalers for production deployment
4. Monitor for data drift and retrain periodically

KEY CONCEPTS APPLIED:
=====================
✓ Time series split (temporal, not random)
✓ Data leakage prevention (fit on train only)
✓ Feature engineering (calendar, lags, rolling stats)
✓ Production-ready artifacts (scalers, feature names)
"""

    print(summary)

    print("=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
