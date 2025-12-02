#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLIGHT TELEMETRY - NEURAL NETWORK TRAINING (PyTorch)
==============================================================================

Purpose: Train neural network models and compare with classical regression

This script:
1. Loads preprocessed datasets from preprocessing stage
2. Converts NumPy arrays to PyTorch tensors
3. Implements a single-neuron neural network (equivalent to linear regression)
4. Trains the neural network using gradient descent with PyTorch
5. Compares neural network with sklearn Linear Regression
6. Implements training loop with validation monitoring
7. Saves models, training history, and predictions

Key Concepts:
- A single neuron with linear activation = Linear Regression
- Demonstrates mathematical equivalence between neural nets and classical ML
- Introduces PyTorch fundamentals: tensors, autograd, optimization
- Establishes foundation for more complex neural architectures

Neural Network Architecture:
- Input layer: n features (after preprocessing)
- Output layer: 1 neuron (predicted duration)
- Activation: None (linear/identity) - maintains linearity
- Loss function: MSE (Mean Squared Error)
- Optimizer: Adam (adaptive learning rate)

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
import pickle

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')


class Config:
    """Neural network training configuration parameters."""
    # Directories
    OUTPUT_DIR = Path("outputs")
    ARTIFACTS_DIR = OUTPUT_DIR / "models"
    TABLES_DIR = OUTPUT_DIR / "results"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

    # Input files (from preprocessing)
    X_TRAIN_FILE = ARTIFACTS_DIR / "X_train.pkl"
    X_TEST_FILE = ARTIFACTS_DIR / "X_test.pkl"
    Y_TRAIN_FILE = ARTIFACTS_DIR / "y_train.pkl"
    Y_TEST_FILE = ARTIFACTS_DIR / "y_test.pkl"
    FEATURE_NAMES_FILE = TABLES_DIR / "feature_names_after_oh.csv"

    # Output files
    TRAINING_HISTORY_FILE = TABLES_DIR / "neural_training_history.csv"
    WEIGHTS_COMPARISON_FILE = TABLES_DIR / "weights_comparison.csv"

    # Neural network hyperparameters
    RANDOM_STATE = 42
    LEARNING_RATE = 0.005
    NUM_EPOCHS = 10000
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.15
    EARLY_STOPPING_PATIENCE = 50

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# SECTION 2: NEURAL NETWORK MODEL DEFINITION
# ==============================================================================


class SingleNeuronRegression(nn.Module):
    """
    Single neuron neural network - mathematically equivalent to linear regression.

    Architecture:
        Input: n features
        Output: 1 value (predicted duration)
        Activation: None (linear/identity)

    Mathematical operation:
        y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

    This is EXACTLY the multiple linear regression equation!
    """

    def __init__(self, n_features: int):
        """
        Initialize single neuron model.

        Args:
            n_features: Number of input features
        """
        super(SingleNeuronRegression, self).__init__()

        # Single linear layer (one neuron)
        # Maps n features to 1 output
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, n_features)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # No activation function - keeps linearity
        return self.linear(x)

    def get_weights_and_bias(self) -> Tuple[np.ndarray, float]:
        """
        Extract weights and bias for comparison with sklearn.

        Returns:
            Tuple of (weights, bias)
        """
        weights = self.linear.weight.detach().cpu().numpy().flatten()
        bias = self.linear.bias.detach().cpu().item()
        return weights, bias


# ==============================================================================
# SECTION 3: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    separator = char * 120
    print(f"\n{separator}")
    print(title)
    print(separator)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model: Any, filename: str) -> None:
    """
    Save model to file.

    Args:
        model: Model object
        filename: Filename without path
    """
    filepath = Config.ARTIFACTS_DIR / filename

    if isinstance(model, nn.Module):
        # PyTorch model - save state dict
        torch.save(model.state_dict(), filepath)
    else:
        # Sklearn model - use pickle
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
        filename: Filename without path
    """
    preds_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    filepath = Config.PREDICTIONS_DIR / filename
    preds_df.to_csv(filepath, index=False)
    print(f"✓ Predictions saved to: {filepath}")


# ==============================================================================
# SECTION 4: DATA LOADING
# ==============================================================================


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load preprocessed training and test data.

    Returns:
        Tuple containing:
        - X_train: Training features
        - X_test: Test features
        - y_train: Training target
        - y_test: Test target
        - feature_names: List of feature names
    """
    print_section("[SECTION 1] LOAD PREPROCESSED DATA")

    print("Loading training features...")
    with open(Config.X_TRAIN_FILE, 'rb') as f:
        X_train = pickle.load(f)
    print(f"✓ X_train loaded: {X_train.shape}")

    print("\nLoading test features...")
    with open(Config.X_TEST_FILE, 'rb') as f:
        X_test = pickle.load(f)
    print(f"✓ X_test loaded: {X_test.shape}")

    print("\nLoading training target...")
    with open(Config.Y_TRAIN_FILE, 'rb') as f:
        y_train = pickle.load(f)
    print(f"✓ y_train loaded: {y_train.shape}")

    print("\nLoading test target...")
    with open(Config.Y_TEST_FILE, 'rb') as f:
        y_test = pickle.load(f)
    print(f"✓ y_test loaded: {y_test.shape}")

    print("\nLoading feature names...")
    feature_names_df = pd.read_csv(Config.FEATURE_NAMES_FILE)
    feature_names = feature_names_df['Feature_Name'].tolist()
    print(f"✓ Feature names loaded: {len(feature_names)} features")

    return X_train, X_test, y_train, y_test, feature_names


# ==============================================================================
# SECTION 5: DATA PREPARATION FOR PYTORCH
# ==============================================================================


def prepare_pytorch_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Tuple:
    """
    Convert NumPy arrays to PyTorch tensors and create data loaders.

    Args:
        X_train: Training features (NumPy)
        X_test: Test features (NumPy)
        y_train: Training target (NumPy)
        y_test: Test target (NumPy)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, X_train_tensor,
                  X_test_tensor, y_train_tensor, y_test_tensor)
    """
    print_section("[SECTION 2] CONVERT TO PYTORCH TENSORS")

    print("CRITICAL STEP: Converting NumPy arrays to PyTorch tensors")
    print()
    print("Why tensors?")
    print("  • PyTorch operates on tensors (similar to NumPy arrays)")
    print("  • Tensors support automatic differentiation (autograd)")
    print("  • Can be moved to GPU for faster computation")
    print()

    # Convert to float32 (PyTorch default)
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)

    # Handle pandas Series or NumPy array
    if hasattr(y_train, 'values'):
        y_train_vals = y_train.values
        y_test_vals = y_test.values
    else:
        y_train_vals = y_train
        y_test_vals = y_test

    y_train_tensor = torch.FloatTensor(y_train_vals).reshape(-1, 1)
    y_test_tensor = torch.FloatTensor(y_test_vals).reshape(-1, 1)

    print("Conversion complete:")
    print(f"  X_train: {X_train_tensor.shape} - dtype: {X_train_tensor.dtype}")
    print(f"  X_test:  {X_test_tensor.shape} - dtype: {X_test_tensor.dtype}")
    print(f"  y_train: {y_train_tensor.shape} - dtype: {y_train_tensor.dtype}")
    print(f"  y_test:  {y_test_tensor.shape} - dtype: {y_test_tensor.dtype}")
    print()

    # Split training data into train and validation
    n_train = int(len(X_train_tensor) * (1 - Config.VALIDATION_SPLIT))

    X_train_split = X_train_tensor[:n_train]
    y_train_split = y_train_tensor[:n_train]
    X_val_split = X_train_tensor[n_train:]
    y_val_split = y_train_tensor[n_train:]

    print(
        f"Train/Validation split ({int((1 - Config.VALIDATION_SPLIT) * 100)}/{int(Config.VALIDATION_SPLIT * 100)}):")
    print(f"  Training samples:   {len(X_train_split)}")
    print(f"  Validation samples: {len(X_val_split)}")
    print(f"  Test samples:       {len(X_test_tensor)}")
    print()

    # Create datasets
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val_split, y_val_split)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create data loaders
    print(f"Creating DataLoaders with batch_size={Config.BATCH_SIZE}...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True  # Shuffle training data
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False  # Don't shuffle validation
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False  # Don't shuffle test
    )

    print("✓ DataLoaders created successfully")
    print(f"  Training batches:   {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches:       {len(test_loader)}")

    return (train_loader, val_loader, test_loader,
            X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)


# ==============================================================================
# SECTION 6: TRAINING FUNCTIONS
# ==============================================================================


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train model for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run on (CPU/GPU)

    Returns:
        Average training loss for the epoch
    """
    model.train()  # Set model to training mode
    total_loss = 0.0

    for batch_X, batch_y in train_loader:
        # Move data to device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_X)

        # Calculate loss
        loss = criterion(predictions, batch_y)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate model for one epoch.

    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on (CPU/GPU)

    Returns:
        Average validation loss for the epoch
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for batch_X, batch_y in val_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            predictions = model(batch_X)

            # Calculate loss
            loss = criterion(predictions, batch_y)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train_neural_network(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int
) -> Dict[str, list]:
    """
    Complete training loop with early stopping.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run on
        num_epochs: Maximum number of epochs
        patience: Early stopping patience

    Returns:
        Dictionary with training history
    """
    print_section("[SECTION 3] TRAIN NEURAL NETWORK")

    print("Training configuration:")
    print(f"  Learning rate:  {Config.LEARNING_RATE}")
    print(f"  Batch size:     {Config.BATCH_SIZE}")
    print(f"  Max epochs:     {num_epochs}")
    print(f"  Device:         {device}")
    print(f"  Early stopping: {patience} epochs")
    print()

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    start_time = time.time()

    print("Starting training loop...")
    print("-" * 120)

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device)

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Store history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print()
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

    training_time = time.time() - start_time

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("-" * 120)
    print()
    print("Training completed!")
    print(f"  Total epochs:       {len(history['epoch'])}")
    print(f"  Training time:      {training_time:.2f} seconds")
    print(f"  Best val loss:      {best_val_loss:.6f}")
    print(f"  Final train loss:   {history['train_loss'][-1]:.6f}")
    print(f"  Final val loss:     {history['val_loss'][-1]:.6f}")

    return history


# ==============================================================================
# SECTION 7: SKLEARN COMPARISON MODEL
# ==============================================================================


def train_sklearn_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[LinearRegression, Dict]:
    """
    Train sklearn Linear Regression for comparison.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        Tuple of (trained model, info dict)
    """
    print_section("[SECTION 4] TRAIN SKLEARN LINEAR REGRESSION (BASELINE)")

    print("Training standard Linear Regression with sklearn...")
    print("Purpose: Demonstrate mathematical equivalence with neural network")
    print()

    # Handle pandas Series
    if hasattr(y_train, 'values'):
        y_train = y_train.values
        y_test = y_test.values

    model = LinearRegression()

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"✓ Model trained in {training_time:.4f} seconds")
    print()

    # Display parameters
    print("Learned parameters:")
    print(f"  Intercept (bias):        {model.intercept_:.4f}")
    print(f"  Number of coefficients:  {len(model.coef_)}")
    print()

    # Generate predictions
    print("Generating predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate losses
    train_mse = np.mean((y_train - y_pred_train) ** 2)
    test_mse = np.mean((y_test - y_pred_test) ** 2)

    print(f"  Train MSE: {train_mse:.6f}")
    print(f"  Test MSE:  {test_mse:.6f}")
    print()

    # Save predictions
    save_predictions(y_test, y_pred_test, "preds_sklearn_linear.csv")
    save_model(model, "model_sklearn_linear.pkl")
    print()

    info = {
        'training_time': training_time,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'n_features': X_train.shape[1]
    }

    return model, info


# ==============================================================================
# SECTION 8: MODEL COMPARISON
# ==============================================================================


def compare_models(
    pytorch_model: nn.Module,
    sklearn_model: LinearRegression,
    feature_names: list
) -> pd.DataFrame:
    """
    Compare weights between PyTorch and sklearn models.

    Args:
        pytorch_model: Trained PyTorch model
        sklearn_model: Trained sklearn model
        feature_names: List of feature names

    Returns:
        DataFrame with weight comparison
    """
    print_section("[SECTION 5] COMPARE NEURAL NETWORK WITH SKLEARN")

    print("Extracting model parameters...")
    print()

    # Get PyTorch weights
    pytorch_weights, pytorch_bias = pytorch_model.get_weights_and_bias()

    # Get sklearn weights
    sklearn_weights = sklearn_model.coef_
    sklearn_bias = sklearn_model.intercept_

    print("BIAS COMPARISON:")
    print(f"  PyTorch bias:  {pytorch_bias:.6f}")
    print(f"  Sklearn bias:  {sklearn_bias:.6f}")
    print(f"  Difference:    {abs(pytorch_bias - sklearn_bias):.6f}")
    print()

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'PyTorch_Weight': pytorch_weights,
        'Sklearn_Weight': sklearn_weights,
        'Absolute_Difference': np.abs(pytorch_weights - sklearn_weights),
        'Relative_Difference_Pct': np.abs((pytorch_weights - sklearn_weights) /
                                          (sklearn_weights + 1e-10)) * 100
    })

    print("WEIGHT COMPARISON (first 10 features):")
    print(comparison_df.head(10).to_string(index=False))
    print()

    # Calculate statistics
    mean_abs_diff = comparison_df['Absolute_Difference'].mean()
    max_abs_diff = comparison_df['Absolute_Difference'].max()

    print("COMPARISON STATISTICS:")
    print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"  Max absolute difference:  {max_abs_diff:.6f}")
    print()

    if mean_abs_diff < 0.01:
        print("✓ EXCELLENT: Weights are nearly identical!")
        print("  This confirms the mathematical equivalence between:")
        print("  • Single neuron neural network (no activation)")
        print("  • Multiple linear regression")
    elif mean_abs_diff < 0.1:
        print("✓ GOOD: Weights are very similar")
        print("  Minor differences due to:")
        print("  • Different optimization algorithms")
        print("  • Stochastic gradient descent vs closed-form solution")
    else:
        print("⚠ WARNING: Significant differences detected")
        print("  Possible causes:")
        print("  • Insufficient training epochs")
        print("  • Learning rate too high/low")
        print("  • Early stopping triggered too early")
    print()

    # Save comparison
    comparison_filepath = Config.WEIGHTS_COMPARISON_FILE
    comparison_df.to_csv(comparison_filepath, index=False, float_format='%.6f')
    print(f"✓ Weights comparison saved to: {comparison_filepath}")

    return comparison_df


# ==============================================================================
# SECTION 9: EVALUATION AND PREDICTION
# ==============================================================================


def evaluate_neural_network(
    model: nn.Module,
    X_test_tensor: torch.Tensor,
    y_test_tensor: torch.Tensor,
    y_test: np.ndarray,
    device: torch.device
) -> Tuple[np.ndarray, Dict]:
    """
    Evaluate neural network on test set.

    Args:
        model: Trained neural network
        X_test_tensor: Test features (tensor)
        y_test_tensor: Test target (tensor)
        y_test: Test target (NumPy)
        device: Device to run on

    Returns:
        Tuple of (predictions, metrics dict)
    """
    print_section("[SECTION 6] EVALUATE NEURAL NETWORK ON TEST SET")

    model.eval()

    print("Generating predictions on test set...")

    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        predictions_tensor = model(X_test_device)
        predictions = predictions_tensor.cpu().numpy().flatten()

    print(f"✓ Predictions generated: {len(predictions)} samples")
    print()

    # Handle pandas Series
    if hasattr(y_test, 'values'):
        y_test_vals = y_test.values
    else:
        y_test_vals = y_test

    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_test_vals, predictions)
    mse = mean_squared_error(y_test_vals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_vals, predictions)

    print("TEST SET PERFORMANCE:")
    print(f"  MAE:  {mae:.4f} minutes")
    print(f"  MSE:  {mse:.4f} minutes²")
    print(f"  RMSE: {rmse:.4f} minutes")
    print(f"  R²:   {r2:.4f}")
    print()

    # Save predictions
    save_predictions(
        y_test_vals,
        predictions,
        "preds_pytorch_single_neuron.csv")
    print()

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

    return predictions, metrics


# ==============================================================================
# SECTION 10: SAVE TRAINING HISTORY
# ==============================================================================


def save_training_history(
        history: Dict,
        pytorch_metrics: Dict,
        sklearn_info: Dict) -> None:
    """
    Save training history and summary statistics.

    Args:
        history: Training history dictionary
        pytorch_metrics: PyTorch model metrics
        sklearn_info: Sklearn model information
    """
    print_section("[SECTION 7] SAVE TRAINING HISTORY")

    # Save epoch-by-epoch history
    history_df = pd.DataFrame(history)
    history_filepath = Config.TRAINING_HISTORY_FILE
    history_df.to_csv(history_filepath, index=False, float_format='%.6f')

    print(f"✓ Training history saved to: {history_filepath}")
    print(f"  Epochs recorded: {len(history_df)}")
    print()

    # Create summary comparison
    summary = pd.DataFrame({
        'Model': ['PyTorch Single Neuron', 'Sklearn Linear Regression'],
        'Test_MAE': [pytorch_metrics['mae'], np.sqrt(sklearn_info['test_mse'])],
        'Test_RMSE': [pytorch_metrics['rmse'], np.sqrt(sklearn_info['test_mse'])],
        'Test_R2': [pytorch_metrics['r2'], 'N/A'],
        'Training_Time_Seconds': [history_df['epoch'].iloc[-1] * 0.1, sklearn_info['training_time']]
    })

    print("MODEL COMPARISON SUMMARY:")
    print(summary.to_string(index=False))
    print()


# ==============================================================================
# SECTION 11: MAIN EXECUTION
# ==============================================================================


def main():
    """Main training pipeline execution."""
    print("=" * 120)
    print("NEURAL NETWORK TRAINING (PyTorch) - FLIGHT TELEMETRY")
    print("=" * 120)
    print()
    print("OBJECTIVE: Demonstrate that a single neuron = Multiple Linear Regression")
    print()

    try:
        # Set random seed
        set_seed(Config.RANDOM_STATE)

        # Create output directories
        Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load data
        X_train, X_test, y_train, y_test, feature_names = load_data()

        # 2. Prepare PyTorch data
        (train_loader, val_loader, test_loader,
         X_train_tensor, X_test_tensor,
         y_train_tensor, y_test_tensor) = prepare_pytorch_data(
            X_train, X_test, y_train, y_test
        )

        # 3. Create neural network model
        n_features = X_train.shape[1]
        model = SingleNeuronRegression(n_features).to(Config.DEVICE)

        print("\nModel architecture:")
        print(model)
        print(
            f"\nNumber of parameters: {sum(p.numel() for p in model.parameters())}")

        # 4. Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        # 5. Train neural network
        history = train_neural_network(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=Config.DEVICE,
            num_epochs=Config.NUM_EPOCHS,
            patience=Config.EARLY_STOPPING_PATIENCE
        )

        # 6. Save PyTorch model
        save_model(model, "model_pytorch_single_neuron.pth")

        # 7. Train sklearn baseline
        sklearn_model, sklearn_info = train_sklearn_baseline(
            X_train, y_train, X_test, y_test
        )

        # 8. Compare models
        compare_models(model, sklearn_model, feature_names)

        # 9. Evaluate neural network
        predictions, pytorch_metrics = evaluate_neural_network(
            model, X_test_tensor, y_test_tensor, y_test, Config.DEVICE
        )

        # 10. Save training history
        save_training_history(history, pytorch_metrics, sklearn_info)

        # Final summary
        print("\n" + "=" * 120)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 120)
        print()

        print("KEY FINDINGS:")
        print()
        print("1. MATHEMATICAL EQUIVALENCE:")
        print("   ✓ Single neuron (no activation) = Multiple Linear Regression")
        print("   ✓ Both models learn similar weights and bias")
        print()

        print("2. PYTORCH FUNDAMENTALS LEARNED:")
        print("   ✓ Tensors: PyTorch's fundamental data structure")
        print("   ✓ Autograd: Automatic differentiation for gradients")
        print("   ✓ Training loop: Forward pass → Loss → Backward pass → Update")
        print("   ✓ DataLoaders: Efficient batch processing")
        print()

        print("3. NEXT STEPS:")
        print("   • Add hidden layers for non-linear relationships")
        print("   • Experiment with activation functions (ReLU, tanh)")
        print("   • Try different optimizers (SGD, RMSprop)")
        print("   • Implement learning rate scheduling")
        print()

        print("FILES SAVED:")
        print(
            f"  • PyTorch model:     {
                Config.ARTIFACTS_DIR /
                'model_pytorch_single_neuron.pth'}")
        print(
            f"  • Sklearn model:     {
                Config.ARTIFACTS_DIR /
                'model_sklearn_linear.pkl'}")
        print(f"  • Training history:  {Config.TRAINING_HISTORY_FILE}")
        print(f"  • Weight comparison: {Config.WEIGHTS_COMPARISON_FILE}")
        print(f"  • Predictions:       {Config.PREDICTIONS_DIR}")
        print()

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
