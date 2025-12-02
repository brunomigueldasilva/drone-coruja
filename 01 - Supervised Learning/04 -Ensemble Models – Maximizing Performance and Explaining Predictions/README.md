# Lab 06.4 - Ensemble Models: Maximize Performance and Explain Predictions

**Topic:** Ensemble Methods (Bagging and Boosting)

## 1. Objective

This laboratory has a dual objective:

1. Use **ensemble models** (such as Random Forest and Gradient Boosting) to improve prediction performance in a classification task.
2. Go beyond simple prediction and use these models to **interpret the data**, identifying the most important variables and how they influence the outcome.

## 2. Dataset

This project uses the flight incident classification dataset (`voos_pre_voo.csv`), focused on predicting `incidente_reportado` (reported incident: 0 or 1). This allows for a direct comparison of the predictive power of ensemble methods against simpler models.

### Features

| Feature | Description |
|---------|-------------|
| `idade_aeronave_anos` | Aircraft age in years |
| `horas_voo_desde_ultima_manutencao` | Flight hours since last maintenance |
| `previsao_turbulencia` | Turbulence forecast |
| `tipo_missao` | Mission type |
| `experiencia_piloto_anos` | Pilot experience in years |

### Target Variable

- `incidente_reportado` — Binary classification (0 = No incident, 1 = Incident reported)

## 3. Tasks

### 3.1 Baseline Model: Decision Tree

- Train a single **Decision Tree** (without depth limit) to serve as a *baseline*.
- Visualize the tree and observe how it can easily suffer from *overfitting*.
- Evaluate its performance on the test set (using F1-Score and ROC-AUC).

### 3.2 Ensemble Modeling

#### Random Forest (Bagging)
- Train a `RandomForestClassifier` model.
- Analyze the **Out-of-Bag (OOB) score** and compare it to the test set score.

#### XGBoost (Boosting)
- Train an `XGBClassifier` model.
- Use cross-validation to find an optimal number of estimators (`n_estimators`).

#### Gradient Boosting
- Train a `GradientBoostingClassifier` for additional comparison.

#### (Optional) CatBoost
- Train a `CatBoostClassifier`, known for handling categorical variables natively.
- Compare its convenience and performance.

### 3.3 Evaluation and Comparison

- Compare performance metrics (F1-Score, ROC-AUC) across all models:
  - Decision Tree
  - Random Forest
  - XGBoost
  - Gradient Boosting
- Determine which model achieved the best results.
- Assess whether there was significant improvement compared to simpler models.

### 3.4 Explainability (XAI - Explainable AI)

#### Feature Importance
- Extract and plot feature importance from Random Forest and XGBoost.
- Verify if both rankings are consistent.
- Identify the top 3 most influential variables.

#### Partial Dependence Plots (PDP)
- Generate PDPs for the 1-2 most important variables.
- Analyze what the PDP reveals about the *relationship* between each variable and the probability of an incident.
- Example question: Does the probability increase or decrease with `experiencia_piloto_anos` (pilot experience)?

## 4. Key Concepts and Lessons

### Overfitting vs. Generalization
Compare the performance of a single decision tree (likely overfitting) with Random Forest. Discuss how *bagging* helps reduce variance and improve generalization.

### Bias-Variance Trade-off
Analyze the models in terms of this trade-off:
- A single tree has **high variance**
- Random Forest **reduces variance** (through bagging)
- Boosting **reduces bias** (sequentially)

### Explainability of "Black Box" Models
Ensemble models are often seen as "black boxes." Discuss how techniques like *Feature Importance* and *Partial Dependence Plots* allow us to open this box and gain confidence in the predictions.

## 5. Project Structure

```
project/
├── inputs/
│   └── voos_pre_voo.csv              # Input dataset
│
├── outputs/
│   ├── data_processed/               # Processed data and artifacts
│   │   ├── X_train_scaled.csv
│   │   ├── X_test_scaled.csv
│   │   ├── y_train.csv
│   │   ├── y_test.csv
│   │   ├── scaler.pkl
│   │   └── feature_names.pkl
│   │
│   ├── models/                       # Trained model artifacts
│   │   ├── decisiontree_model.pkl
│   │   ├── randomforest_model.pkl
│   │   ├── xgboost_model.pkl
│   │   └── gradientboosting_model.pkl
│   │
│   ├── predictions/                  # Model predictions
│   │   ├── decisiontree_predictions.csv
│   │   ├── randomforest_predictions.csv
│   │   ├── xgboost_predictions.csv
│   │   └── gradientboosting_predictions.csv
│   │
│   ├── graphics/                     # Visualizations
│   │   ├── target_distribution.png
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves_comparison.png
│   │   ├── feature_importance_rf.png
│   │   ├── feature_importance_xgb.png
│   │   ├── feature_importance_comparison.png
│   │   ├── partial_dependence_top3.png
│   │   └── partial_dependence_interaction.png
│   │
│   ├── results/                      # Tables and metrics
│   │   ├── descriptive_statistics.csv
│   │   ├── correlation_with_target.csv
│   │   └── class_distribution.csv
│   │
│   ├── metrics_comparison.csv        # Comparative metrics table
│   ├── metrics_comparison.md         # Metrics in Markdown format
│   └── FINAL_REPORT.md               # Comprehensive final report
│
├── 01_exploratory_analysis.py        # EDA and feature engineering
├── 02_preprocessing.py               # Data preparation and encoding
├── 03_train_models.py                # Ensemble model training
├── 04_evaluate_metrics.py            # Performance evaluation
├── 05_roc_and_importance.py          # ROC curves and feature importance
├── 06_partial_dependence.py          # XAI with partial dependence plots
├── 07_final_report.py                # Final report generation
├── 08_orchestrator.py                # Pipeline orchestrator
└── README.md                         # This file
```

## 6. Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `01_exploratory_analysis.py` | Exploratory Data Analysis (EDA) and feature engineering |
| `02_preprocessing.py` | Data preparation, encoding, and scaling |
| `03_train_models.py` | Training of ensemble models (Decision Tree, Random Forest, XGBoost, Gradient Boosting) |
| `04_evaluate_metrics.py` | Performance evaluation with multiple metrics |
| `05_roc_and_importance.py` | ROC curve generation and feature importance analysis |
| `06_partial_dependence.py` | Partial Dependence Plots for model explainability |
| `07_final_report.py` | Comprehensive final report generation |
| `08_orchestrator.py` | Pipeline automation and orchestration |

## 7. Usage

### Running the Complete Pipeline

```bash
# Interactive mode
python 08_orchestrator.py

# Run complete pipeline (non-interactive)
python 08_orchestrator.py --all

# Run specific steps (e.g., steps 1, 3, and 4)
python 08_orchestrator.py --steps 1,3,4

# Run from a specific step onwards
python 08_orchestrator.py --from 4

# Clean all outputs
python 08_orchestrator.py --clean
```

### Running Individual Scripts

```bash
python 01_exploratory_analysis.py
python 02_preprocessing.py
python 03_train_models.py
python 04_evaluate_metrics.py
python 05_roc_and_importance.py
python 06_partial_dependence.py
python 07_final_report.py
```

## 8. Requirements

### Required Libraries

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`
- `scipy`
- `colorama` (optional, for colored terminal output)

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost scipy colorama
```

## 9. Expected Deliverables

The final report should include:

1. **Model Training and Evaluation** — Training and evaluation of all models (Decision Tree, Random Forest, XGBoost, Gradient Boosting)

2. **Comparative Metrics Table** — Side-by-side comparison of performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)

3. **Feature Importance Ranking** — Bar chart showing the ranking of variable importance from ensemble models

4. **Partial Dependence Plots** — 1-2 PDPs with interpretation of how key features affect incident probability

5. **Conclusions** — Discussion on the power of ensemble models for both prediction and knowledge extraction

## 10. Author

**Bruno Silva**  
2025

---

*This project is part of Lab 06.4 - Ensemble Models: Maximize Performance and Explain Predictions*
