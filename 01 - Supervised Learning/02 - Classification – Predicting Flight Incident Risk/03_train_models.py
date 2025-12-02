#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
AIRCRAFT INCIDENT CLASSIFICATION - MODEL TRAINING
==============================================================================

Purpose: Train multiple classification algorithms and save predictions

This script:
1. Loads preprocessed training and test data
2. Defines 5 classification models (Logistic Regression, KNN, SVM Linear, SVM RBF, Naive Bayes)
3. Trains each model on training data
4. Generates predictions and probabilities for test set
5. Saves trained models as pickle files
6. Saves all predictions for later evaluation
7. Records and reports training times for each model

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
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Model training configuration parameters."""
    PROCESSED_DATA_DIR = Path('outputs/data_processed')
    MODELS_DIR = Path('outputs/models')
    PREDICTIONS_DIR = Path('outputs/predictions')
    RANDOM_STATE = 42  # For reproducibility
    OUTPUT_CSV = Path('outputs/results/training_times.csv')
    KB_DIVISOR = 1024  # For file size conversion


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


def load_pickle(filepath: Path) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Any: Unpickled object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj: Any, filepath: Path) -> str:
    """
    Save object to pickle file and return file size.

    Args:
        obj: Object to pickle
        filepath: Path where to save the pickle file

    Returns:
        str: Human-readable file size
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


def load_preprocessed_data(
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load preprocessed train and test data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test

    Raises:
        SystemExit: If data directory or files not found
    """
    print_section("1. LOADING PREPROCESSED DATA")

    # Check if processed data directory exists
    if not Config.PROCESSED_DATA_DIR.exists():
        print(f"✗ ERROR: Directory not found: {Config.PROCESSED_DATA_DIR}")
        print("  Please run 02_preprocessing.py first.")
        exit(1)

    # Load data files
    print("Loading CSV files:")

    try:
        X_train = pd.read_csv(Config.PROCESSED_DATA_DIR / 'X_train.csv')
        X_test = pd.read_csv(Config.PROCESSED_DATA_DIR / 'X_test.csv')
        y_train = pd.read_csv(
            Config.PROCESSED_DATA_DIR /
            'y_train.csv').values.ravel()
        y_test = pd.read_csv(
            Config.PROCESSED_DATA_DIR /
            'y_test.csv').values.ravel()

        print(f"  ✓ X_train.csv        Shape: {X_train.shape}")
        print(f"  ✓ X_test.csv         Shape: {X_test.shape}")
        print(f"  ✓ y_train.csv        Shape: {y_train.shape}")
        print(f"  ✓ y_test.csv         Shape: {y_test.shape}")

    except FileNotFoundError as e:
        print(f"✗ ERROR: File not found: {e.filename}")
        print("  Please run 02_preprocessing.py first.")
        exit(1)

    # Verify data integrity
    print("\n✓ Data loaded successfully!")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Test samples: {X_test.shape[0]:,}")
    print(f"  Number of features: {X_train.shape[1]}")

    # Check class distribution
    print("\nClass distribution:")
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    print(f"  Train: {train_counts.to_dict()}")
    print(f"  Test:  {test_counts.to_dict()}")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# SECTION 4: MODEL DEFINITION
# ==============================================================================


def define_models() -> Dict[str, Tuple[Any, str]]:
    """
    Define all models to be trained.

    Returns:
        Dict[str, Tuple[Any, str]]: Dictionary mapping model names to
            (model_object, description) tuples
    """
    print_section("2. DEFINING MODELS")

    """
    MODEL SELECTION RATIONALE:

    We train 5 different algorithms to compare performance and understand
    which approach works best for our incident classification problem.
    Each model has different strengths and assumptions.
    """

    models = {
        'logistic_regression': (
            LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=Config.RANDOM_STATE
            ),
            """Linear model, interpretable, good baseline.
            Uses class_weight='balanced' to handle class imbalance.
            Provides probability estimates and feature importance.
            Good when: Features have linear relationships with target."""
        ),

        'knn': (
            KNeighborsClassifier(n_neighbors=5),
            """Non-parametric, instance-based learning.
            Classifies based on majority vote of 5 nearest neighbors.
            Sensitive to feature scale (data is already scaled).
            Good when: Data has local patterns, decision boundaries are irregular."""
        ),

        'svm_linear': (
            SVC(
                kernel='linear',
                probability=True,
                class_weight='balanced',
                random_state=Config.RANDOM_STATE
            ),
            """Finds optimal hyperplane to separate classes.
            Linear kernel: Efficient in high dimensions.
            class_weight='balanced' handles imbalanced data.
            Good when: Clear margin of separation exists between classes."""
        ),

        'svm_rbf': (
            SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=Config.RANDOM_STATE
            ),
            """SVM with Radial Basis Function (Gaussian) kernel.
            Allows non-linear decision boundaries.
            More flexible than linear SVM.
            Good when: Relationship between features and target is non-linear."""
        ),

        'naive_bayes': (
            GaussianNB(),
            """Probabilistic classifier based on Bayes theorem.
            Assumes feature independence and Gaussian distribution.
            Very fast to train, requires small training data.
            Good when: Features are relatively independent."""
        )
    }

    print(f"Models defined: {len(models)}")
    print("\nModel overview:")
    for i, (name, (model, description)) in enumerate(models.items(), 1):
        print(f"\n{i}. {name}")
        print(f"   Type: {type(model).__name__}")
        first_sentence = description.strip().split('\n')[0]
        print(f"   Note: {first_sentence}")

    return models


# ==============================================================================
# SECTION 5: MODEL TRAINING
# ==============================================================================


def train_models(
    models: Dict[str, Tuple[Any, str]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Train all models and save predictions and trained models.

    Args:
        models: Dictionary of model definitions
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Dict[str, float]: Training times for each model
    """
    print_section("3. TRAINING MODELS")

    training_times = {}

    for i, (model_name, (model, description)) in enumerate(models.items(), 1):
        print(f"\n{'-' * 80}")
        print(f"[{i}/{len(models)}] Training: {model_name}")
        print(f"{'-' * 80}")
        print(f"Algorithm: {type(model).__name__}")

        # Training phase
        print("\n► Training:")
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        end_time = time.perf_counter()
        training_time = end_time - start_time
        training_times[model_name] = training_time

        print(f"  ✓ Completed in {training_time:.4f} seconds")

        # Prediction phase
        print("\n► Predictions:")
        y_pred = model.predict(X_test)
        print(f"  ✓ Generated predictions for {len(y_pred):,} test samples")

        # Probability estimates
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            print(f"  ✓ Generated probabilities: {y_proba.shape}")
        else:
            y_proba = None
            print("  ⚠️  No probability estimates available")

        # Save model
        print("\n► Saving model:")
        model_filepath = Config.MODELS_DIR / f'{model_name}.pkl'
        size_str = save_pickle(model, model_filepath)
        print(f"  ✓ Saved: {model_filepath.name} ({size_str})")

        # Save predictions
        print("\n► Saving predictions:")
        predictions_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred
        })

        if y_proba is not None:
            for j, class_label in enumerate(model.classes_):
                predictions_df[f'proba_class_{class_label}'] = y_proba[:, j]

        predictions_filepath = Config.PREDICTIONS_DIR / \
            f'predictions_{model_name}.csv'
        predictions_df.to_csv(predictions_filepath, index=False)
        print(
            f"  ✓ Saved: {
                predictions_filepath.name} ({
                len(predictions_df):,} rows)")

    return training_times


# ==============================================================================
# SECTION 6: SUMMARY AND REPORTING
# ==============================================================================


def save_training_times(training_times: Dict[str, float]) -> None:
    """
    Save training times to CSV file.

    Args:
        training_times: Dictionary of model names and training times
    """
    times_df = pd.DataFrame([
        {'model': name, 'training_time_seconds': time_val}
        for name, time_val in training_times.items()
    ]).sort_values('training_time_seconds')

    Config.OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    times_df.to_csv(Config.OUTPUT_CSV, index=False)


def print_training_summary(training_times: Dict[str, float]) -> None:
    """
    Print formatted training summary.

    Args:
        training_times: Dictionary of model names and training times
    """
    print_section("4. TRAINING SUMMARY")

    print("\nTraining times by model:")
    print(f"\n{'Model':<25} {'Time (seconds)':<20} {'Formatted':<20}")
    print("-" * 65)

    sorted_times = sorted(training_times.items(), key=lambda x: x[1])

    for model_name, train_time in sorted_times:
        if train_time < 0.001:
            time_formatted = f"{train_time * 1000:.3f} ms"
        elif train_time < 1:
            time_formatted = f"{train_time * 1000:.1f} ms"
        elif train_time < 60:
            time_formatted = f"{train_time:.2f} s"
        else:
            minutes = int(train_time // 60)
            seconds = train_time % 60
            time_formatted = f"{minutes}m {seconds:.1f}s"

        print(f"{model_name:<25} {train_time:<20.6f} {time_formatted:<20}")

    total_time = sum(training_times.values())
    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Fastest model: {sorted_times[0][0]} ({sorted_times[0][1]:.4f}s)")
    print(f"Slowest model: {sorted_times[-1][0]} ({sorted_times[-1][1]:.4f}s)")


def print_final_summary(n_models: int) -> None:
    """
    Print final completion summary.

    Args:
        n_models: Number of models trained
    """
    print_section("TRAINING COMPLETED SUCCESSFULLY!", "=")

    print("\nOutput files:")
    print(f"  Models: {Config.MODELS_DIR}/")
    print(f"    → {len(list(Config.MODELS_DIR.glob('*.pkl')))} pickle files")

    print(f"\n  Predictions: {Config.PREDICTIONS_DIR}/")
    print(
        f"    → {len(list(Config.PREDICTIONS_DIR.glob('predictions_*.csv')))} CSV files")

    print(f"\n  Training times: {Config.OUTPUT_CSV}")

    print("\nNext steps:")
    print("  1. Run 04_avaliacao_metricas.py to evaluate model performance")
    print("  2. Compare metrics to select best model")
    print("  3. Analyze prediction patterns and errors")


# ==============================================================================
# SECTION 7: MAIN EXECUTION
# ==============================================================================


def main() -> None:
    """Main execution function."""
    print_section("AIRCRAFT INCIDENT CLASSIFICATION - MODEL TRAINING", "=")

    # Create output directories
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()

    # Define models
    models = define_models()

    # Train all models
    training_times = train_models(models, X_train, y_train, X_test, y_test)

    # Save and display results
    save_training_times(training_times)
    print_training_summary(training_times)
    print_final_summary(len(models))


if __name__ == '__main__':
    main()
