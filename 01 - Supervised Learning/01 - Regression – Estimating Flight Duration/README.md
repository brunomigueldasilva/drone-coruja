# Lab 06.1 — Regression: Estimating Flight Duration

**Topic:** Linear and Regularized Regression

---

## 1. Objective

The objective of this exercise is to build and evaluate regression models capable of estimating the **total flight duration** based on mission variables available before takeoff. We compare simple linear models, multiple linear models, and regularized models to understand their trade-offs.

---

## 2. Dataset

The project uses a tabular dataset (`voos_telemetria.csv`) containing the following variables for each flight:

| Variable | Description |
|----------|-------------|
| `distancia_planeada` | Planned distance (km) |
| `carga_util_kg` | Payload weight (kg) |
| `altitude_media_m` | Average altitude (meters) |
| `condicao_meteo` | Weather condition (categorical: `Bom`, `Moderado`, `Adverso`) |
| `duracao_voo_min` | Flight duration in minutes — **target variable** |

---

## 3. Tasks

### 3.1 Exploratory Data Analysis (EDA)
- Visualize the distribution of the target variable (`duracao_voo_min`)
- Analyze the correlation between numerical variables and flight duration
- Convert the `condicao_meteo` variable to numerical format (e.g., one-hot encoding)

### 3.2 Modeling
- **Baseline Model:** Train a **Simple Linear Regression** model using only `distancia_planeada`
- **Full Model:** Train a **Multiple Linear Regression** model with all predictor variables
- **Regularized Models:** Train **Lasso** and **Ridge** models to evaluate the impact of regularization
- **Non-linearity:** Explore **Polynomial Regression** (degree 2) to capture non-linear relationships

### 3.3 Evaluation and Interpretation
- For each model, calculate the following metrics on the test set: **MAE**, **RMSE**, and **R²**
- Compare model performance — which one generalizes best?
- Interpret the coefficients of the Multiple Linear Regression model — what do they tell us about feature importance?
- Analyze the Lasso model coefficients — were any variables considered irrelevant?

---

## 4. Challenges and Lessons

- **Data Scaling:** Evaluate the importance of scaling features before training models, especially for Lasso and Ridge
- **Interpretation vs. Prediction:** Reflect on the trade-off between simplicity/interpretability of a linear model and the potential higher accuracy of a more complex model
- **Data Leakage:** Ensure that no future information (unavailable before the flight) is used to train the model

---

## 5. Project Structure

```
├── inputs/
│   └── voos_telemetria.csv          # Input dataset
├── outputs/
│   ├── graphics/                    # Visualizations and plots
│   ├── models/                      # Trained model artifacts (.pkl)
│   ├── predictions/                 # Model predictions (.csv)
│   ├── results/                     # Metrics and summaries
│   └── FINAL_REPORT.md              # Comprehensive final report
├── 01_exploratory_analysis.py       # EDA and feature engineering
├── 02_preprocessing.py              # Data preparation and splitting
├── 03_train_models.py               # Model training
├── 04_evaluate_metrics.py           # Performance evaluation
├── 05_plot_predicted_vs_actual.py   # Scatter plot visualization
├── 06_residual_analysis.py          # Residual diagnostics
├── 07_final_report.py               # Final report generation
├── 08_orchestrator.py               # Pipeline orchestrator
└── README.md                        # This file
```

---

## 6. Models Trained

| Model | Description | Key Characteristics |
|-------|-------------|---------------------|
| **Simple Linear Regression** | Uses only `distancia_planeada` | Baseline, most interpretable |
| **Multiple Linear Regression** | Uses all features | Risk of multicollinearity |
| **Ridge Regression** | L2 regularization (α=1.0) | Shrinks coefficients, handles multicollinearity |
| **Lasso Regression** | L1 regularization (α=0.001) | Can zero out coefficients, automatic feature selection |
| **Polynomial Regression** | Degree 2 on numeric features | Captures non-linear relationships, risk of overfitting |

---

## 7. How to Run

### Prerequisites

Ensure the following Python libraries are installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Running the Complete Pipeline

**Interactive Mode:**
```bash
python 08_orchestrator.py
```

**Run Complete Pipeline (non-interactive):**
```bash
python 08_orchestrator.py --all
```

**Run Specific Steps:**
```bash
python 08_orchestrator.py --steps 1,3,4    # Run EDA, training, and evaluation
python 08_orchestrator.py --from 4         # Run from step 4 onwards
```

**Clean Outputs:**
```bash
python 08_orchestrator.py --clean
```

### Running Individual Scripts

```bash
python 01_exploratory_analysis.py
python 02_preprocessing.py
python 03_train_models.py
python 04_evaluate_metrics.py
python 05_plot_predicted_vs_actual.py
python 06_residual_analysis.py
python 07_final_report.py
```

---

## 8. Expected Outputs

### Visualizations (`outputs/graphics/`)
- `target_hist.png` — Histogram of target variable distribution
- `target_boxplot.png` — Boxplot of target variable
- `heatmap_correlations.png` — Correlation heatmap
- `box_weather_vs_target.png` — Weather condition vs. flight duration
- `predicted_vs_actual.png` — Predicted vs. actual values scatter plot
- `residuals_hist.png` — Residuals histogram
- `residuals_vs_predictions.png` — Residuals vs. predictions plot

### Results (`outputs/results/`)
- `target_statistics.csv` — Descriptive statistics of the target
- `eda_summary.md` — EDA summary report
- `feature_names_after_oh.csv` — Feature names after one-hot encoding
- `training_times.csv` — Training times for each model
- `model_comparison.csv` — Performance metrics comparison table
- `model_comparison.md` — Markdown summary of model comparison

### Models (`outputs/models/`)
- `preprocessor.pkl` — Fitted preprocessor
- `model_linear_simple.pkl` — Simple Linear Regression model
- `model_linear_multiple.pkl` — Multiple Linear Regression model
- `model_ridge.pkl` — Ridge Regression model
- `model_lasso.pkl` — Lasso Regression model
- `model_polynomial_deg2.pkl` — Polynomial Regression model
- `poly_transformer_deg2.pkl` — Polynomial feature transformer

### Final Report
- `outputs/FINAL_REPORT.md` — Comprehensive report with analysis, metrics, and conclusions

---

## 9. Evaluation Metrics

The models are evaluated using the following metrics on the test set:

| Metric | Description |
|--------|-------------|
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and actual values |
| **RMSE** (Root Mean Squared Error) | Square root of average squared differences — penalizes larger errors |
| **R²** (Coefficient of Determination) | Proportion of variance explained by the model (0 to 1) |

---

## 10. Final Deliverable

A comparative report presenting:
- Exploratory data analysis
- Training code for each model
- A summary table of performance metrics
- Conclusion on the best model for the task and interpretation of results

---

## Author

**Bruno Silva** — 2025
