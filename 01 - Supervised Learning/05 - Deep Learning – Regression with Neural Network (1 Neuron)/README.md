# Lab 06.5 - Deep Learning: Regression with a Single-Neuron Neural Network

**Topic:** Neural Networks with PyTorch — Equivalence with Linear Regression

---

## 1. Objective

The goal of this exercise is to demonstrate that **a neural network with only 1 neuron is mathematically equivalent to Multiple Linear Regression**. We build this neuron using **PyTorch** and compare it with classical models.

### Why is this exercise important?

- Understand that neural networks are a **generalization** of linear models
- Learn PyTorch fundamentals: tensors, autograd, and training loops
- Establish the foundation for more complex neural networks (future labs)

---

## 2. Dataset

The project uses the `voos_telemetria.csv` (flight telemetry) dataset containing:

| Feature | Description |
|---------|-------------|
| `distancia_planeada` | Planned distance (km) |
| `carga_util_kg` | Payload weight (kg) |
| `altitude_media_m` | Average altitude (meters) |
| `condicao_meteo` | Weather condition (categorical: `Bom`, `Moderado`, `Adverso`) |
| `duracao_voo_min` | Flight duration in minutes — **target variable** |

---

## 3. Project Structure

```
project/
├── inputs/
│   └── voos_telemetria.csv          # Raw flight telemetry data
├── outputs/
│   ├── graphics/                     # All visualizations
│   ├── results/                      # Metrics, training history, comparisons
│   ├── models/                       # Saved models and preprocessor
│   └── predictions/                  # Model predictions
├── 01_exploratory_analysis.py        # Exploratory Data Analysis (EDA)
├── 02_preprocessing.py               # Data preprocessing and transformation
├── 03_train_neural_models.py         # PyTorch neural network training
├── 04_visualize_neural_network.py    # Neural network visualizations
├── 05_final_report.py                # Final report generation
├── 06_orchestrator.py                # Pipeline automation tool
└── README.md                         # This file
```

---

## 4. Tasks

### 4.1. Data Preparation

1. **Preprocessing:**
   - Load and clean the data
   - Apply *one-hot encoding* to the `condicao_meteo` variable
   - **Normalize/Standardize** features (critical for neural networks!)
   - Convert data to **PyTorch Tensors**

2. **Data Split:**
   - Training set: 70%
   - Validation set: 15%
   - Test set: 15%

### 4.2. Neural Network Model Construction

**Architecture: 1 Linear Neuron**

| Component | Specification |
|-----------|---------------|
| **Input** | n features (after one-hot encoding) |
| **Output** | 1 value (predicted duration) |
| **Activation function** | None (identity) — maintains linearity |
| **Loss function** | MSE (Mean Squared Error) |
| **Optimizer** | Adam (or SGD) |

```python
import torch
import torch.nn as nn

class SingleNeuronRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return self.linear(x)
```

### 4.3. Model Training

The training loop implements:
1. Forward pass
2. Loss calculation
3. Backward pass (gradients)
4. Weight update

**Key points:**
- Use `model.train()` during training
- Use `model.eval()` and `torch.no_grad()` during validation
- Monitor training and validation loss per epoch
- Implement **early stopping** (optional)

### 4.4. Comparison with Sklearn

Train a **Multiple Linear Regression** model with sklearn and compare:
- **Coefficients** (weights) of both models — they should be similar!
- **Bias** (intercept)
- **Performance metrics** on the test set

### 4.5. Visualizations

The following visualizations are generated:

1. **Learning Curve:** Training vs. validation loss per epoch
2. **Actual vs. Predicted Values** on the test set
3. **Coefficient Comparison:** Bar plot comparing PyTorch vs. sklearn weights
4. **Residual Distribution**

---

## 5. Evaluation Metrics

For the neural network model, calculate:

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **R²** | Coefficient of Determination |

**Crucial question:** Are the results identical to the sklearn model? If not, why?

---

## 6. Key Concepts

### 6.1. Why 1 Neuron = Linear Regression?

A single neuron performs the operation:

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

This is **exactly** the multiple linear regression equation!

### 6.2. Hyperparameters to Explore

| Hyperparameter | Consideration |
|----------------|---------------|
| **Learning rate** | Too high → diverges; too low → slow training |
| **Batch size** | Size of batches for gradient calculation |
| **Number of epochs** | How many times to iterate over the entire dataset |

### 6.3. Feature Normalization

**Why is it critical?**
- Different scales can cause gradients to explode or vanish
- Accelerates convergence
- Makes the model more stable

---

## 7. Usage

### Running the Complete Pipeline

```bash
# Interactive mode
python 06_orchestrator.py

# Run complete pipeline automatically
python 06_orchestrator.py --all

# Run specific steps (e.g., training and visualization)
python 06_orchestrator.py --steps 3,4

# Run from step N onwards
python 06_orchestrator.py --from 3

# Clean all outputs
python 06_orchestrator.py --clean
```

### Running Individual Scripts

```bash
# Step 1: Exploratory Data Analysis
python 01_exploratory_analysis.py

# Step 2: Data Preprocessing
python 02_preprocessing.py

# Step 3: Train Neural Network Models
python 03_train_neural_models.py

# Step 4: Generate Visualizations
python 04_visualize_neural_network.py

# Step 5: Generate Final Report
python 05_final_report.py
```

---

## 8. Requirements

### Python Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
torch
colorama (optional, for colored terminal output)
```

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy torch colorama
```

For PyTorch with CUDA support, visit: https://pytorch.org/get-started/locally/

---

## 9. Expected Outputs

### Models
- `single_neuron_model.pth` — PyTorch neural network state dict
- `sklearn_linear_model.pkl` — Sklearn linear regression model
- `preprocessor.pkl` — Data preprocessing pipeline

### Results
- `neural_training_history.csv` — Training and validation loss per epoch
- `weights_comparison.csv` — Side-by-side comparison of model coefficients
- `neural_test_metrics.csv` — Performance metrics on test set

### Visualizations
- Learning curves (loss vs. epoch)
- Actual vs. predicted scatter plots
- Coefficient comparison bar charts
- Residual distribution plots

### Report
- `FINAL_REPORT.md` — Comprehensive analysis report

---

## 10. Final Deliverable

A complete Python project containing:

1. **Fully commented code** with:
   - Data preparation into tensors
   - Model class definition
   - Training loop implemented from scratch
   - Evaluation and comparison with sklearn

2. **Visualizations:**
   - Learning curve
   - Coefficient comparison
   - Prediction scatter plot
   - Residual analysis

3. **Written Analysis:**
   - Confirmation of mathematical equivalence
   - Discussion on hyperparameters
   - Conclusions on when to use PyTorch vs. sklearn

---

## Author

**Bruno Silva** — 2025

---

## License

This project is for educational purposes as part of a Deep Learning laboratory exercise.
