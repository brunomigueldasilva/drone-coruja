# Lab 06.2 - Classification: Predicting Flight Incident Risk

**Topic:** Binary Classification and Model Evaluation

---

## 1. Objective

The goal of this lab is to develop a classification model capable of predicting the probability of a **safety incident** (`risk = 1`) occurring during a flight, using data available *before* takeoff. The focus is on evaluating and comparing different algorithms and interpreting their predictions.

---

## 2. Dataset

The project uses a tabular dataset (`voos_pre_voo.csv`) with the following features:

| Feature | Description |
|---------|-------------|
| `idade_aeronave_anos` | Aircraft age in years |
| `horas_voo_desde_ultima_manutencao` | Flight hours since last maintenance |
| `previsao_turbulencia` | Turbulence forecast (Low, Medium, High) |
| `tipo_missao` | Mission type (Surveillance, Cargo, Transport) |
| `experiencia_piloto_anos` | Pilot experience in years |
| `incidente_reportado` | Reported incident (0 or 1) — **target variable** |

> ⚠️ **Note:** This is an **imbalanced dataset** — the majority of flights do not have incidents.

---

## 3. Tasks

### 3.1 Analysis and Preprocessing

- Verify the imbalance of the target variable. How many flights belong to each class?
- Analyze the relationship between predictor variables and incident occurrence.
- Convert categorical variables (`previsao_turbulencia`, `tipo_missao`) to numeric format.
- Scale numeric variables, as models like KNN and SVM are sensitive to scale.

### 3.2 Modeling

Train and evaluate the following classification models:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline model |
| **K-Nearest Neighbors (KNN)** | Instance-based learning |
| **Support Vector Machine (SVM)** | Linear kernel |
| **Kernel SVM** | RBF (Radial Basis Function) kernel |
| **Naïve Bayes** | Gaussian variant |

### 3.3 Detailed Evaluation

For each model, calculate the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

Additionally:

- Generate the **Confusion Matrix** for the best model (based on F1-Score or ROC-AUC). Discuss False Positives and False Negatives in the context of the problem.
- Plot the **ROC Curve** for all models on the same graph for visual performance comparison.

---

## 4. Challenges and Lessons

### Imbalance Management

- How does class imbalance affect the metrics?
- Discuss the relevance of *Accuracy* in this scenario and why *Recall* and *F1-Score* are more important.
- **Optional:** Experiment with techniques such as `class_weight='balanced'` or over-sampling/under-sampling.

### Threshold Selection

- Logistic Regression and SVM provide probabilities.
- How can choosing a decision *threshold* (different from the default 0.5) optimize the trade-off between Precision and Recall?

### Scale Sensitivity

- Compare the performance of KNN or SVM with and without data scaling to demonstrate its importance.

---

## 5. Project Structure

```
├── inputs/
│   └── voos_pre_voo.csv              # Input dataset
├── outputs/
│   ├── data_processed/               # Processed train/test splits
│   ├── models/                       # Trained model artifacts (.pkl)
│   ├── predictions/                  # Model predictions
│   ├── results/                      # Metrics tables and summaries
│   ├── graphics/                     # Visualizations (PNG/PDF)
│   └── FINAL_REPORT.md               # Comprehensive final report
├── 01_exploratory_analysis.py        # EDA and feature engineering
├── 02_preprocessing.py               # Data preparation and scaling
├── 03_train_models.py                # Model training
├── 04_evaluate_metrics.py            # Performance evaluation
├── 05_confusion_matrix.py            # Confusion matrix analysis
├── 06_roc_curves.py                  # ROC and PR curves
├── 07_final_report.py                # Report generation
├── 08_orchestrator.py                # Pipeline orchestrator
└── README.md                         # This file
```

---

## 6. Usage

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
python 01_exploratory_analysis.py     # Exploratory Data Analysis
python 02_preprocessing.py            # Data Preprocessing
python 03_train_models.py             # Model Training
python 04_evaluate_metrics.py         # Metrics Evaluation
python 05_confusion_matrix.py         # Confusion Matrix Analysis
python 06_roc_curves.py               # ROC & PR Curves
python 07_final_report.py             # Final Report Generation
```

---

## 7. Requirements

The following Python libraries are required:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `scipy`
- `colorama` (optional, for colored terminal output)

Install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy colorama
```

---

## 8. Final Deliverable

The final product is a comprehensive report containing:

- ✅ Analysis of data imbalance
- ✅ Preprocessing pipeline description
- ✅ Model training and evaluation code
- ✅ Comparative metrics table
- ✅ Commented Confusion Matrix and ROC Curve
- ✅ Justification for the "best" model selection for this specific task

---

## 9. Key Considerations

### Why Accuracy Can Be Misleading

In imbalanced datasets, a model that always predicts the majority class (no incident) can achieve high accuracy while being useless for detecting actual incidents. This is why **Recall** (ability to find all positive cases) and **F1-Score** (harmonic mean of Precision and Recall) are more meaningful metrics.

### Business Context

In aviation safety:
- **False Negatives** (missing a real incident) are potentially catastrophic
- **False Positives** (predicting an incident that doesn't happen) lead to unnecessary precautions but are safer

This asymmetry should guide model selection and threshold tuning.

---

## Author

Bruno Silva — 2025

---

## License

This project is for educational purposes as part of a Machine Learning laboratory assignment.
