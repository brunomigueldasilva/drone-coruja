#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
TIME SERIES FORECASTING - COMPREHENSIVE MODEL TRAINING
==============================================================================

Purpose: Train and evaluate specialized time series forecasting models

This script:
1. Loads preprocessed training and test data with temporal split
2. Trains multiple forecasting models specialized for each dataset:
   - VOLTAGE (High-Frequency Series): ARIMA, SARIMA, Holt-Winters
   - MISSIONS (Seasonal Daily Series): Prophet
3. Implements naive baseline (persistence model) for comparison
4. Generates predictions on held-out test set
5. Saves trained models and predictions for evaluation
6. Tracks training times and computational performance
7. Creates training summary with all model artifacts

Model Selection Rationale:
- Naive Baseline: Establishes minimum acceptable performance
- ARIMA: Classical approach for non-seasonal time series
- SARIMA: Handles explicit seasonality in data
- Holt-Winters: Efficient exponential smoothing with trend/seasonality
- Prophet: Facebook's tool optimized for business time series

Expected Datasets:
- Voltage: Preprocessed minute-level battery voltage data
- Missions: Preprocessed daily mission count data

Author: Bruno Silva
Date: 2025
==============================================================================
"""

import pandas as pd
import pickle
import warnings
from pathlib import Path
import time

# Time Series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prophet
try:
    from prophet import Prophet
    # Suppress Prophet's verbose output
    import logging
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("WARNING: Prophet not installed. Install with: pip install prophet")

warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Configuration parameters."""
    # Directories
    PROCESSED_DATA_DIR = Path('outputs/data_processed')
    MODELS_DIR = Path('outputs/models')
    PREDICTIONS_DIR = Path('outputs/predictions')

    # ARIMA/SARIMA parameters
    ARIMA_ORDER = (5, 1, 0)  # (p, d, q)
    SARIMA_ORDER = (1, 1, 1)  # (p, d, q)
    SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)  # (P, D, Q, s)

    # Holt-Winters parameters
    HW_SEASONAL_PERIODS_VOLTAGE = 60  # 60 minutes
    HW_SEASONAL_PERIODS_MISSIONS = 7  # 7 days


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def save_pickle(obj, filepath: Path) -> None:
    """Save object to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Path):
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_preprocessed_data(dataset_name='voltage'):
    """
    Load preprocessed and scaled data.

    Args:
        dataset_name: 'voltage' or 'missions'

    Returns:
        tuple: X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    """
    print_section(f"LOADING PREPROCESSED DATA: {dataset_name.upper()}")

    data_dir = Config.PROCESSED_DATA_DIR / dataset_name

    if not data_dir.exists():
        print(f"\n✗ ERROR: Directory not found: {data_dir}")
        print("\nPlease run the preprocessing script first:")
        print("  python 02_preprocessing.py")
        exit(1)

    print(f"Loading from: {data_dir}")

    # Load data
    X_train_scaled = load_pickle(data_dir / "X_train_scaled.pkl")
    print(f"✓ Loaded X_train_scaled: shape {X_train_scaled.shape}")

    X_test_scaled = load_pickle(data_dir / "X_test_scaled.pkl")
    print(f"✓ Loaded X_test_scaled: shape {X_test_scaled.shape}")

    y_train = load_pickle(data_dir / "y_train.pkl")
    print(f"✓ Loaded y_train: shape {y_train.shape}")

    y_test = load_pickle(data_dir / "y_test.pkl")
    print(f"✓ Loaded y_test: shape {y_test.shape}")

    feature_names = load_pickle(data_dir / "feature_names.pkl")
    print(f"✓ Loaded feature_names: {len(feature_names)} features")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


# ==============================================================================
# MODEL 1: NAIVE BASELINE
# ==============================================================================

def train_naive_baseline(y_train, y_test):
    """
    Train naive baseline (persistence model).
    Predicts that the next value equals the current value.
    """
    print("\n" + "-" * 80)
    print("MODEL 1: NAIVE BASELINE (Persistence Model)")
    print("-" * 80)

    start_time = time.time()

    # Prediction: shift test values by 1
    y_pred_naive = y_test.shift(1)
    y_pred_naive.iloc[0] = y_train.iloc[-1]

    training_time = time.time() - start_time

    print(f"✓ Naive baseline created in {training_time:.4f} seconds")

    return None, y_pred_naive.values, training_time


# ==============================================================================
# MODEL 2: ARIMA
# ==============================================================================

def train_arima(y_train, y_test, order=None):
    """
    Train ARIMA model.

    Args:
        y_train: Training data
        y_test: Test data
        order: ARIMA order (p, d, q). If None, uses Config.ARIMA_ORDER
    """
    print("\n" + "-" * 80)
    print("MODEL 2: ARIMA (AutoRegressive Integrated Moving Average)")
    print("-" * 80)

    if order is None:
        order = Config.ARIMA_ORDER

    print(f"ARIMA order (p, d, q): {order}")

    start_time = time.time()

    try:
        # Fit ARIMA model
        model = ARIMA(y_train, order=order)
        model_fit = model.fit()

        # Forecast
        predictions = model_fit.forecast(steps=len(y_test))

        training_time = time.time() - start_time

        print(f"✓ ARIMA trained in {training_time:.4f} seconds")
        print(f"  AIC: {model_fit.aic:.2f}")
        print(f"  BIC: {model_fit.bic:.2f}")

        return model_fit, predictions.values, training_time

    except Exception as e:
        print(f"✗ ARIMA training failed: {e}")
        print("  Returning naive predictions")
        return None, y_test.shift(1).fillna(y_train.iloc[-1]).values, 0.0


# ==============================================================================
# MODEL 3: SARIMA
# ==============================================================================

def train_sarima(y_train, y_test, order=None, seasonal_order=None):
    """
    Train SARIMA model (Seasonal ARIMA).

    Args:
        y_train: Training data
        y_test: Test data
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)
    """
    print("\n" + "-" * 80)
    print("MODEL 3: SARIMA (Seasonal ARIMA)")
    print("-" * 80)

    if order is None:
        order = Config.SARIMA_ORDER
    if seasonal_order is None:
        seasonal_order = Config.SARIMA_SEASONAL_ORDER

    print(f"SARIMA order (p, d, q): {order}")
    print(f"Seasonal order (P, D, Q, s): {seasonal_order}")

    start_time = time.time()

    try:
        # Fit SARIMA model
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)

        # Forecast
        predictions = model_fit.forecast(steps=len(y_test))

        training_time = time.time() - start_time

        print(f"✓ SARIMA trained in {training_time:.4f} seconds")
        print(f"  AIC: {model_fit.aic:.2f}")
        print(f"  BIC: {model_fit.bic:.2f}")

        return model_fit, predictions.values, training_time

    except Exception as e:
        print(f"✗ SARIMA training failed: {e}")
        print("  Returning naive predictions")
        return None, y_test.shift(1).fillna(y_train.iloc[-1]).values, 0.0


# ==============================================================================
# MODEL 4: HOLT-WINTERS
# ==============================================================================

def train_holt_winters(
        y_train,
        y_test,
        seasonal_periods=None,
        dataset_name='voltage'):
    """
    Train Holt-Winters Exponential Smoothing model.

    Args:
        y_train: Training data
        y_test: Test data
        seasonal_periods: Number of periods in a season
        dataset_name: 'voltage' or 'missions'
    """
    print("\n" + "-" * 80)
    print("MODEL 4: HOLT-WINTERS (Exponential Smoothing)")
    print("-" * 80)

    if seasonal_periods is None:
        if dataset_name == 'voltage':
            seasonal_periods = Config.HW_SEASONAL_PERIODS_VOLTAGE
        else:
            seasonal_periods = Config.HW_SEASONAL_PERIODS_MISSIONS

    print(f"Seasonal periods: {seasonal_periods}")

    start_time = time.time()

    try:
        # Check if we have enough data
        if len(y_train) < 2 * seasonal_periods:
            print(
                f"  Warning: Insufficient data for seasonality (need {
                    2 *
                    seasonal_periods}, have {
                    len(y_train)})")
            seasonal_periods = None

        # Fit Holt-Winters model
        if seasonal_periods:
            model = ExponentialSmoothing(
                y_train,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add',
                damped_trend=True
            )
        else:
            model = ExponentialSmoothing(
                y_train,
                trend='add',
                damped_trend=True
            )

        model_fit = model.fit()

        # Forecast
        predictions = model_fit.forecast(steps=len(y_test))

        training_time = time.time() - start_time

        print(f"✓ Holt-Winters trained in {training_time:.4f} seconds")

        return model_fit, predictions.values, training_time

    except Exception as e:
        print(f"✗ Holt-Winters training failed: {e}")
        print("  Returning naive predictions")
        return None, y_test.shift(1).fillna(y_train.iloc[-1]).values, 0.0


# ==============================================================================
# MODEL 5: PROPHET
# ==============================================================================

def train_prophet(y_train, y_test):
    """
    Train Prophet model (Facebook's forecasting tool).
    Best for daily data with strong seasonality.

    Args:
        y_train: Training data with datetime index
        y_test: Test data with datetime index
    """
    print("\n" + "-" * 80)
    print("MODEL 5: PROPHET (Facebook Forecasting)")
    print("-" * 80)

    if not PROPHET_AVAILABLE:
        print("✗ Prophet not available. Skipping.")
        print("  Install with: pip install prophet")
        return None, y_test.shift(1).fillna(y_train.iloc[-1]).values, 0.0

    start_time = time.time()

    try:
        # Prepare data in Prophet format (ds, y)
        df_prophet = pd.DataFrame({
            'ds': y_train.index,
            'y': y_train.values
        })

        # Ensure ds is datetime
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

        # Initialize Prophet with minimal verbosity
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive',
            interval_width=0.95
        )

        # Suppress verbose output
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        # Fit model
        model.fit(df_prophet)

        # Restore stdout
        sys.stdout = old_stdout

        # Create future dataframe for test period
        future_dates = pd.DataFrame({
            'ds': y_test.index
        })
        future_dates['ds'] = pd.to_datetime(future_dates['ds'])

        # Predict
        forecast = model.predict(future_dates)
        predictions = forecast['yhat'].values

        training_time = time.time() - start_time

        print(f"✓ Prophet trained in {training_time:.4f} seconds")
        print(
            f"  Predictions range: [{
                predictions.min():.2f}, {
                predictions.max():.2f}]")

        return model, predictions, training_time

    except Exception as e:
        print(f"✗ Prophet training failed: {e}")
        print("  Possible fixes:")
        print("  - Install cmdstanpy: pip install cmdstanpy")
        print("  - Or use: conda install -c conda-forge prophet")
        print("  Returning naive predictions")
        return None, y_test.shift(1).fillna(y_train.iloc[-1]).values, 0.0


# ==============================================================================
# SAVE FUNCTIONS
# ==============================================================================

def save_model_and_predictions(
        model,
        predictions,
        model_name,
        dataset_name='voltage'):
    """Save trained model and predictions."""
    models_dir = Config.MODELS_DIR / dataset_name
    predictions_dir = Config.PREDICTIONS_DIR / dataset_name

    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    if model is not None:
        model_path = models_dir / f"{model_name}.pkl"
        save_pickle(model, model_path)
        print(f"  ✓ Model saved to: {model_path}")
    else:
        print("  ℹ No model to save (naive baseline or failed model)")

    # Save predictions
    predictions_path = predictions_dir / f"{model_name}_predictions.pkl"
    save_pickle(predictions, predictions_path)
    print(f"  ✓ Predictions saved to: {predictions_path}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main(dataset_name='voltage'):
    """
    Main training pipeline.

    Args:
        dataset_name: 'voltage' or 'missions'
    """
    print("\n" + "=" * 80)
    print(f"TIME SERIES FORECASTING: MODEL TRAINING - {dataset_name.upper()}")
    print("=" * 80)

    # Load data
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = load_preprocessed_data(
        dataset_name)

    # Initialize summary
    training_summary = {
        'dataset_name': dataset_name,
        'model_names': [],
        'training_times': [],
        'predictions': {}
    }

    # -------------------------------------------------------------------------
    # MODEL 1: Naive Baseline (Both datasets)
    # -------------------------------------------------------------------------
    naive_model, naive_predictions, naive_time = train_naive_baseline(
        y_train, y_test)
    save_model_and_predictions(
        naive_model,
        naive_predictions,
        "naive_baseline",
        dataset_name)
    training_summary['model_names'].append('naive_baseline')
    training_summary['training_times'].append(naive_time)
    training_summary['predictions']['naive_baseline'] = naive_predictions

    # -------------------------------------------------------------------------
    # DATASET-SPECIFIC TIME SERIES MODELS
    # -------------------------------------------------------------------------

    if dataset_name == 'voltage':
        # PART A: VOLTAGE MODELS (High-Frequency Series)
        print_section("PART A: VOLTAGE TIME SERIES MODELS")

        # MODEL 2: ARIMA
        arima_model, arima_predictions, arima_time = train_arima(
            y_train, y_test)
        save_model_and_predictions(
            arima_model,
            arima_predictions,
            "arima",
            dataset_name)
        training_summary['model_names'].append('arima')
        training_summary['training_times'].append(arima_time)
        training_summary['predictions']['arima'] = arima_predictions

        # MODEL 3: SARIMA
        sarima_model, sarima_predictions, sarima_time = train_sarima(
            y_train, y_test)
        save_model_and_predictions(
            sarima_model,
            sarima_predictions,
            "sarima",
            dataset_name)
        training_summary['model_names'].append('sarima')
        training_summary['training_times'].append(sarima_time)
        training_summary['predictions']['sarima'] = sarima_predictions

        # MODEL 4: Holt-Winters
        hw_model, hw_predictions, hw_time = train_holt_winters(
            y_train, y_test, dataset_name=dataset_name)
        save_model_and_predictions(
            hw_model,
            hw_predictions,
            "holt_winters",
            dataset_name)
        training_summary['model_names'].append('holt_winters')
        training_summary['training_times'].append(hw_time)
        training_summary['predictions']['holt_winters'] = hw_predictions

    elif dataset_name == 'missions':
        # PART B: MISSIONS MODEL (Seasonal Daily Series)
        print_section("PART B: MISSIONS TIME SERIES MODEL")

        # MODEL 5: Prophet
        prophet_model, prophet_predictions, prophet_time = train_prophet(
            y_train, y_test)
        save_model_and_predictions(
            prophet_model,
            prophet_predictions,
            "prophet",
            dataset_name)
        training_summary['model_names'].append('prophet')
        training_summary['training_times'].append(prophet_time)
        training_summary['predictions']['prophet'] = prophet_predictions

    # -------------------------------------------------------------------------
    # Save Training Summary
    # -------------------------------------------------------------------------
    print_section("SAVING TRAINING SUMMARY")

    summary_dir = Config.MODELS_DIR / dataset_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "training_summary.pkl"

    save_pickle(training_summary, summary_path)
    print(f"✓ Training summary saved to: {summary_path}")

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print_section(f"TRAINING COMPLETE - {dataset_name.upper()}")
    print(f"\nTotal models trained: {len(training_summary['model_names'])}")
    print("\nTraining Times:")
    for model_name, train_time in zip(
            training_summary['model_names'], training_summary['training_times']):
        print(f"  {model_name:20s}: {train_time:8.4f} seconds")

    print(
        f"\nTotal training time: {sum(training_summary['training_times']):.4f} seconds")
    print("\nModels trained:")
    if dataset_name == 'voltage':
        print("  • Naive Baseline (persistence)")
        print("  • ARIMA (autoregressive integrated moving average)")
        print("  • SARIMA (seasonal ARIMA)")
        print("  • Holt-Winters (exponential smoothing)")
    else:
        print("  • Naive Baseline (persistence)")
        print("  • Prophet (Facebook's forecasting tool)")

    print("\nNext Steps:")
    print("  → Run 04_evaluate_metrics.py to compare model performance")
    print("=" * 80)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import sys

    # Process command line arguments
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
        if dataset not in ['voltage', 'missions', 'both']:
            print(
                f"Error: Invalid dataset '{dataset}'. Choose 'voltage', 'missions', or 'both'")
            sys.exit(1)
    else:
        dataset = 'both'  # Default: process both datasets

    # Process datasets
    if dataset == 'both':
        datasets_to_process = ['voltage', 'missions']
    else:
        datasets_to_process = [dataset]

    # Train models for each dataset
    for ds in datasets_to_process:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING DATASET: {ds.upper()}")
        print(f"{'=' * 80}")
        main(ds)
