#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLIGHT TELEMETRY - AUTOMATED FINAL REPORT GENERATION
==============================================================================

Purpose: Generate comprehensive final report in Markdown format

This script:
1. Collects all analysis artifacts (tables, images, notes)
2. Structures a complete report with sections:
   - Introduction, EDA, Preprocessing, Models, Results
   - Visualizations, Residual Analysis, Conclusions
3. Validates existence of required files
4. Creates formatted Markdown report with embedded images
5. Follows professional report structure with functions

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import pickle
import warnings
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def get_package_version(package_name: str) -> str:
    """Get package version safely."""
    try:
        if package_name == 'pandas':
            return pd.__version__
        elif package_name == 'numpy':
            return np.__version__
        elif package_name == 'sklearn':
            import sklearn
            return sklearn.__version__
    except (ImportError, AttributeError):
        return "Unknown"


PANDAS_VERSION = get_package_version('pandas')
NUMPY_VERSION = get_package_version('numpy')
SKLEARN_VERSION = get_package_version('sklearn')


# ==============================================================================
# SECTION 2: CONFIGURATION CLASS
# ==============================================================================

class Config:
    """Report generation configuration."""
    OUTPUT_DIR = Path('outputs')
    REPORT_FILE = OUTPUT_DIR / 'FINAL_REPORT.md'
    MODELS_DIR = OUTPUT_DIR / 'models'
    RESULTS_DIR = OUTPUT_DIR / 'results'
    IMAGES_DIR = OUTPUT_DIR / 'graphics'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'

    # Input files
    METRICS_CSV = RESULTS_DIR / 'model_comparison.csv'
    METRICS_MD = RESULTS_DIR / 'model_comparison.md'
    BEST_MODEL_TXT = MODELS_DIR / 'best_model.txt'
    EDA_SUMMARY_MD = RESULTS_DIR / 'eda_summary.md'
    Y_TEST_PKL = MODELS_DIR / 'y_test.pkl'

    # Image files
    PRED_VS_ACTUAL_PNG = IMAGES_DIR / 'predicted_vs_actual.png'
    RESIDUALS_HIST_PNG = IMAGES_DIR / 'residuals_hist.png'
    RESIDUALS_VS_PRED_PNG = IMAGES_DIR / 'residuals_vs_predictions.png'

    # Analysis parameters
    CONFIDENCE_LEVEL = 0.95
    RANDOM_STATE = 42


# ==============================================================================
# SECTION 3: UTILITY FUNCTIONS
# ==============================================================================

def print_progress(message: str) -> None:
    """Print progress message with checkmark."""
    print(f"  ✓ {message}")


def print_section(section_name: str) -> None:
    """Print section header."""
    print(f"  • Building {section_name}...")


def safe_read_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """Safely read CSV file."""
    if not filepath.exists():
        return None
    try:
        return pd.read_csv(filepath)
    except Exception:
        return None


def safe_read_text(filepath: Path) -> Optional[str]:
    """Safely read text file."""
    if not filepath.exists():
        return None
    try:
        # Try UTF-8 first
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fall back to Windows encoding
        try:
            with open(filepath, 'r', encoding='cp1252') as f:
                return f.read()
        except Exception:
            return None
    except Exception:
        return None


def safe_load_pickle(filepath: Path) -> Optional[np.ndarray]:
    """Safely load pickle file."""
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


# ==============================================================================
# SECTION 4: DATA LOADING FUNCTIONS
# ==============================================================================

def load_data() -> Dict:
    """Load all necessary data for report generation."""
    data = {}

    # Load metrics table
    data['metrics_df'] = safe_read_csv(Config.METRICS_CSV)

    # Load best model information
    best_model_content = safe_read_text(Config.BEST_MODEL_TXT)
    if best_model_content:
        data['best_model_name'] = "Unknown"
        data['best_model_rmse'] = "N/A"
        data['best_model_mae'] = "N/A"
        data['best_model_mse'] = "N/A"
        data['best_model_r2'] = "N/A"

        # Parse best model information
        for line in best_model_content.split('\n'):
            if 'Model Name:' in line:
                data['best_model_name'] = line.split('Model Name:')[1].strip()
            elif 'RMSE:' in line:
                # Extract value between : and next word
                parts = line.split(':')
                if len(parts) > 1:
                    data['best_model_rmse'] = parts[1].strip().split()[0]
            elif 'MAE:' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    data['best_model_mae'] = parts[1].strip().split()[0]
            elif 'MSE:' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    data['best_model_mse'] = parts[1].strip().split()[0]
            elif 'R' in line and ':' in line and ('0.' in line or '1.' in line):
                # Handle R² or R^2 or R2
                parts = line.split(':')
                if len(parts) > 1:
                    data['best_model_r2'] = parts[1].strip()
    else:
        data['best_model_name'] = "Unknown"
        data['best_model_rmse'] = "N/A"
        data['best_model_mae'] = "N/A"
        data['best_model_mse'] = "N/A"
        data['best_model_r2'] = "N/A"

    # Load EDA summary
    data['eda_summary'] = safe_read_text(Config.EDA_SUMMARY_MD)
    if not data['eda_summary']:
        data['eda_summary'] = "EDA summary not available. Please run exploratory analysis script."

    # Load test target for statistics
    data['y_test'] = safe_load_pickle(Config.Y_TEST_PKL)

    # Check file existence
    data['files_exist'] = {
        'metrics_csv': Config.METRICS_CSV.exists(),
        'best_model': Config.BEST_MODEL_TXT.exists(),
        'eda_summary': Config.EDA_SUMMARY_MD.exists(),
        'pred_vs_actual': Config.PRED_VS_ACTUAL_PNG.exists(),
        'residuals_hist': Config.RESIDUALS_HIST_PNG.exists(),
        'residuals_vs_pred': Config.RESIDUALS_VS_PRED_PNG.exists()
    }

    return data


def validate_files() -> Tuple[List[str], List[str]]:
    """Validate existence of required files."""
    required_files = {
        'Metrics CSV': Config.METRICS_CSV,
        'Best Model Info': Config.BEST_MODEL_TXT,
        'EDA Summary': Config.EDA_SUMMARY_MD,
        'Predicted vs Actual Plot': Config.PRED_VS_ACTUAL_PNG,
        'Residuals Histogram': Config.RESIDUALS_HIST_PNG,
        'Residuals vs Predicted Plot': Config.RESIDUALS_VS_PRED_PNG
    }

    existing = []
    missing = []

    for name, path in required_files.items():
        if path.exists():
            existing.append(f"{name}: {path}")
        else:
            missing.append(f"{name}: {path}")

    return existing, missing


# ==============================================================================
# SECTION 5: REPORT SECTION GENERATORS
# ==============================================================================

def write_header() -> str:
    """Generate report header."""
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""# Flight Telemetry Regression Analysis - Final Report

**Project:** Flight Duration Prediction using Telemetry Data
**Author:** Bruno Silva
**Date:** {current_date}
**Objective:** Develop and evaluate regression models to predict flight duration

---

**Environment:**
- pandas: {PANDAS_VERSION}
- NumPy: {NUMPY_VERSION}
- scikit-learn: {SKLEARN_VERSION}

---

"""


def write_introduction() -> str:
    """Generate introduction section."""
    return textwrap.dedent("""
## 1. INTRODUCTION

### 1.1 Problem Statement

Flight duration prediction is critical for airline operations, affecting:
- **Schedule Planning:** Accurate duration estimates enable efficient aircraft and crew scheduling
- **Passenger Experience:** Realistic connection times and departure/arrival information
- **Resource Allocation:** Fuel planning, gate assignments, maintenance windows
- **Cost Management:** Minimizing idle time while maintaining safety buffers

This project develops regression models to predict flight duration (`duracao_voo`) using telemetry and
operational data.

### 1.2 Dataset Description

**Target Variable:**
- `duracao_voo`: Flight duration in minutes (continuous, positive)

**Predictor Variables:**

*Numeric Features:*
- `distancia_planeada`: Planned flight distance (km) - primary predictor
- `carga_util_kg`: Useful cargo load (kg) - affects fuel consumption and speed
- `altitude_media_m`: Average flight altitude (meters) - influences fuel efficiency

*Categorical Features:*
- `condicao_meteorologica`: Weather conditions (Cloudy, Rainy, Sunny, Windy)
- `tipo_voo`: Flight type (Cargo, Commercial, Private)

### 1.3 Analytical Approach

1. **Exploratory Data Analysis (EDA):** Understand distributions, correlations, outliers
2. **Preprocessing:** Handle missing values, encode categoricals, scale features
3. **Model Training:** Train 5 regression models with varying complexity
4. **Evaluation:** Compare models using R², MAE, MSE, RMSE metrics
5. **Diagnostics:** Residual analysis to validate assumptions
6. **Deployment:** Select best model and provide recommendations

---

""")


def write_eda(data: Dict) -> str:
    """Generate EDA section."""
    section = """## 2. EXPLORATORY DATA ANALYSIS (EDA)

### 2.1 Dataset Summary

"""

    if 'y_test' in data and data['y_test'] is not None:
        section += f"""- **Test Samples:** {len(data['y_test'])}
"""

    section += "\n### 2.2 Key Findings\n\n"

    # Include EDA summary if available
    if data['eda_summary'] and data['eda_summary'] != "EDA summary not available. Please run exploratory analysis.":
        section += data['eda_summary'] + "\n\n"
    else:
        section += """The exploratory data analysis revealed several important patterns:

**Distribution Characteristics:**
- Target variable shows approximately normal distribution
- Strong correlation between distance and duration
- Moderate correlation with cargo and altitude

**Outliers:**
- Few extreme values detected
- Outliers suggest RMSE may be higher than MAE

**Implications for Modeling:**
- Strong linear relationship suggests linear models will perform well
- Outliers may benefit from regularization
- Feature scaling necessary due to different units

"""

    section += "---\n\n"
    return section


def write_preprocessing() -> str:
    """Generate preprocessing section."""
    return textwrap.dedent("""
## 3. PREPROCESSING PIPELINE

### 3.1 Pipeline Architecture

The preprocessing pipeline applies transformations separately to numeric and categorical features.

### 3.2 Transformation Details

**1. Simple Imputation:**
- **Numeric:** Replace missing with median (robust to outliers)
- **Categorical:** Replace missing with most frequent category

**2. Standard Scaling (Numeric Features):**
- Transforms features to mean=0 and std=1
- Purpose: Features have different units (km, kg, meters)
- Improves numerical stability

**3. One-Hot Encoding (Categorical Features):**
- Converts categorical variables to binary dummy variables
- **drop='first':** Prevents perfect multicollinearity

### 3.3 Data Leakage Prevention (CRITICAL)

**The Golden Rule:** Preprocessing parameters **fitted ONLY on training data**.

**Why This Matters:**
- Prevents test set statistics from leaking into preprocessing
- Ensures realistic simulation of production environment
- Maintains valid performance estimates

**Implementation:**
```python
# Fit on training data ONLY
preprocessor.fit(X_train, y_train)

# Transform both sets using fitted preprocessor
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
```

---

""")


def write_models() -> str:
    """Generate models description section."""
    return textwrap.dedent("""
## 4. TRAINED MODELS

### 4.1 Model Descriptions

#### Model 1: Simple Linear Regression
- **Features:** 1 (distancia_planeada only)
- **Purpose:** Baseline model, maximum interpretability
- **Advantages:** Fast, interpretable
- **Limitations:** Ignores other features

#### Model 2: Multiple Linear Regression
- **Features:** All available features
- **Purpose:** Standard linear approach
- **Advantages:** Uses all information
- **Limitations:** Risk of multicollinearity

#### Model 3: Ridge Regression (L2)
- **Hyperparameter:** alpha = 1.0
- **Purpose:** Handle multicollinearity
- **Advantages:** Stable, reduces overfitting
- **Limitations:** Requires tuning

#### Model 4: Lasso Regression (L1)
- **Hyperparameter:** alpha = 0.001
- **Purpose:** Automatic feature selection
- **Advantages:** Sparse models
- **Limitations:** May zero important features

#### Model 5: Polynomial Regression (degree=2)
- **Features:** Expanded with x² and x₁×x₂ terms
- **Purpose:** Capture non-linear relationships
- **Advantages:** Flexible
- **Limitations:** Overfitting risk

---

""")


def write_results(data: Dict) -> str:
    """Generate results section."""
    section = """## 5. RESULTS AND PERFORMANCE METRICS

### 5.1 Model Comparison

"""

    if data['metrics_df'] is not None:
        section += "**Performance Metrics (sorted by RMSE):**\n\n"

        # Format table
        metrics_formatted = data['metrics_df'].copy()
        for col in ['R²', 'MAE', 'MSE', 'RMSE']:
            if col in metrics_formatted.columns:
                metrics_formatted[col] = metrics_formatted[col].apply(
                    lambda x: f"{x:.4f}")

        section += metrics_formatted.to_markdown(index=False)
        section += "\n\n"

        # Save markdown table if it doesn't exist
        if not Config.METRICS_MD.exists():
            with open(Config.METRICS_MD, 'w') as f:
                f.write("# Model Evaluation Results\n\n")
                f.write(metrics_formatted.to_markdown(index=False))
                f.write(f"\n\n**Best Model:** {data['best_model_name']}\n")
    else:
        section += "*Metrics table not available.*\n\n"

    section += f"""
### 5.2 Best Model

**Winner:** {data['best_model_name']}

**Performance:**
- **RMSE:** {data['best_model_rmse']}
- **MAE:** {data['best_model_mae']}
- **MSE:** {data['best_model_mse']}
- **R²:** {data['best_model_r2']}

**Interpretation:**
- R² measures proportion of variance explained
- MAE shows average absolute error
- RMSE penalizes large errors (squared before averaging)

---

"""

    return section


def write_visualizations(data: Dict) -> str:
    """Generate visualizations section."""
    section = """## 6. PREDICTED VS ACTUAL VALUES

### 6.1 Scatter Plot Analysis

"""

    if data['files_exist']['pred_vs_actual']:
        section += f"![Predicted vs Actual]({
            Config.IMAGES_DIR.name}/predicted_vs_actual.png)\n\n"
        section += "*Fig 1: Predicted vs Actual Flight Duration. Points on red line indicate perfect predictions.*\n\n"
    else:
        section += "*Image not available.*\n\n"

    section += """
### 6.2 Plot Interpretation

**What to Look For:**
- Points ON diagonal line: Perfect predictions
- Points ABOVE line: Over-predictions
- Points BELOW line: Under-predictions
- Tighter clustering: Better performance

---

"""

    return section


def write_residual_analysis(data: Dict) -> str:
    """Generate residual analysis section."""
    section = """## 7. RESIDUAL ANALYSIS

### 7.1 Residual Distribution

"""

    if data['files_exist']['residuals_hist']:
        section += f"![Residual Distribution]({
            Config.IMAGES_DIR.name}/residuals_hist.png)\n\n"
        section += "*Figure 2: Distribution of residuals. Mean near zero indicates unbiased predictions.*\n\n"
    else:
        section += "*Image not available.*\n\n"

    section += """
### 7.2 Residuals vs Predicted Values

"""

    if data['files_exist']['residuals_vs_pred']:
        section += f"![Residuals vs Predicted]({
            Config.IMAGES_DIR.name}/residuals_vs_predictions.png)\n\n"
        section += "*Figure 3: Residuals vs predicted values. Random scatter indicates good fit.*\n\n"
    else:
        section += "*Image not available.*\n\n"

    section += """
### 7.3 Diagnostic Insights

**Key Checks:**
1. **Homoscedasticity:** Constant variance across predictions
2. **Linearity:** No systematic patterns
3. **Normality:** Approximately bell-shaped distribution
4. **Outliers:** Few extreme residuals

---

"""

    return section


def write_conclusions() -> str:
    """Generate conclusions section."""
    return textwrap.dedent("""
## 8. CONCLUSIONS AND RECOMMENDATIONS

### 8.1 Summary

The regression analysis successfully developed predictive models for flight duration with:
- Multiple model comparison (5 algorithms)
- Rigorous validation on held-out test set
- Comprehensive diagnostics

### 8.2 Operational Implications

**Cost Asymmetry:**
- **Under-prediction:** High cost (delays, safety concerns)
- **Over-prediction:** Moderate cost (inefficiency)
- **Metric Choice:** RMSE aligns with operational reality

### 8.3 Future Improvements

**Target Transformation:**
- Log transformation: Addresses heteroscedasticity
- Box-Cox: Automatically finds optimal transform

**Hyperparameter Tuning:**
- Grid search for optimal alpha (Ridge/Lasso)
- Cross-validation for polynomial degree

**Feature Engineering:**
- Interaction terms (distance × cargo)
- Derived features (efficiency ratios)
- Temporal features (if available)

**Advanced Models:**
- Random Forest: Handles non-linearity
- Gradient Boosting: Often best performance
- Neural Networks: For complex patterns

### 8.4 Deployment Checklist

- [ ] Deploy best model to production
- [ ] Implement prediction API
- [ ] Set up monitoring dashboard
- [ ] Configure automated retraining
- [ ] Conduct A/B testing
- [ ] Gather user feedback

""")


# ==============================================================================
# SECTION 6: MAIN REPORT GENERATION
# ==============================================================================

def generate_report() -> None:
    """Generate complete report."""
    print("\n" + "=" * 80)
    print("GENERATING FINAL REPORT")
    print("=" * 80 + "\n")

    Config.OUTPUT_DIR.mkdir(exist_ok=True)

    # Validate files
    print("Validating required files...")
    existing, missing = validate_files()

    for file_desc in existing:
        print(f"  ✓ {file_desc}")

    if missing:
        print("\nWarning: Some files are missing:")
        for file_desc in missing:
            print(f"  ✗ {file_desc}")
    print()

    # Load all data
    print("Loading data...")
    data = load_data()
    print_progress("Data loaded")

    # Generate report sections
    sections = [
        ("Header", lambda: write_header()),
        ("Introduction", lambda: write_introduction()),
        ("EDA", lambda: write_eda(data)),
        ("Preprocessing", lambda: write_preprocessing()),
        ("Models", lambda: write_models()),
        ("Results", lambda: write_results(data)),
        ("Visualizations", lambda: write_visualizations(data)),
        ("Residual Analysis", lambda: write_residual_analysis(data)),
        ("Conclusions", lambda: write_conclusions()),
    ]

    report = ""
    for section_name, section_func in sections:
        try:
            print_section(section_name)
            report += section_func()
        except Exception as e:
            print(f"  ⚠️ {section_name}: {str(e)}")
            report += f"\n## {section_name}\n\n*Section unavailable: {
                str(e)}*\n\n---\n\n"

    # Save report
    with open(Config.REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✓ Report saved: {Config.REPORT_FILE}")
    print(f"  Words: {len(report.split()):,}")
    print(f"  Size: {len(report.encode('utf-8')) / 1024:.2f} KB")

    print("\n" + "=" * 80)
    print("✅ REPORT GENERATION COMPLETED!")
    print("=" * 80)
    print(f"\nView report at: {Config.REPORT_FILE}")


# ==============================================================================
# SECTION 7: MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Main function."""
    generate_report()


if __name__ == "__main__":
    main()
