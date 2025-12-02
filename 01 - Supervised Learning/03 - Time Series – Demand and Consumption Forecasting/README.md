# Lab 06.3 - Time Series: Demand and Consumption Forecasting

## 📋 Overview

This laboratory focuses on **modeling and forecasting two distinct time series** using classical statistical methods and modern approaches. The project implements a complete end-to-end machine learning pipeline for time series analysis, from exploratory data analysis to final report generation.

### Target Series

1. **Battery Voltage Consumption** - High-frequency series measured minute-by-minute to anticipate peaks or drops in power consumption.

2. **Daily Flight Missions** - Daily mission counts to optimize resource planning and operational scheduling.

---

## 🎯 Objectives

- Perform comprehensive exploratory data analysis on time series data
- Test for stationarity and apply appropriate transformations
- Analyze autocorrelation patterns (ACF/PACF) to identify model parameters
- Train and compare multiple forecasting models (classical and modern)
- Evaluate model performance using standard metrics
- Generate visualizations with confidence intervals
- Produce a comprehensive final report with insights and recommendations

---

## 📁 Project Structure

```
├── inputs/
│   ├── voltagem_bateria.csv      # Battery voltage data (timestamp, voltagem)
│   └── missoes_diarias.csv       # Daily missions data (data, num_missoes)
│
├── outputs/
│   ├── graphics/                 # All generated visualizations
│   ├── data_processed/           # Preprocessed datasets
│   │   ├── voltage/              # Processed voltage data
│   │   └── missions/             # Processed missions data
│   ├── models/                   # Trained model artifacts
│   │   ├── voltage/
│   │   └── missions/
│   ├── predictions/              # Model predictions
│   │   ├── voltage/
│   │   └── missions/
│   ├── voltage_metrics_comparison.csv
│   ├── missions_metrics_comparison.csv
│   └── FINAL_REPORT.md
│
├── 01_exploratory_analysis.py    # EDA and stationarity tests
├── 02_preprocessing.py           # Feature engineering & temporal split
├── 03_train_models.py            # Model training
├── 04_evaluate_metrics.py        # Performance evaluation
├── 05_plot_predicted_vs_actual.py # Forecast visualizations
├── 06_residual_analysis.py       # Residual diagnostics
├── 07_final_report.py            # Report generation
├── 08_orchestrator.py            # Pipeline orchestrator
└── README.md
```

---

## 🔧 Pipeline Components

### 1. Exploratory Data Analysis (`01_exploratory_analysis.py`)

- Time series visualization and pattern identification
- **Stationarity testing** using Augmented Dickey-Fuller (ADF) test
- **Seasonal decomposition** into trend, seasonal, and residual components
- **ACF/PACF analysis** to identify ARIMA parameters (p, d, q)
- Gap detection and data validation

### 2. Data Preprocessing (`02_preprocessing.py`)

- **Calendar features**: month, day of week, hour, is_weekend
- **Lag features**: past values to inform predictions
- **Rolling window features**: moving averages and standard deviations
- **Temporal train-test split** (chronological, no shuffle - critical for time series!)
- Feature scaling with StandardScaler (fit only on training data to prevent leakage)

### 3. Model Training (`03_train_models.py`)

**For Voltage (High-Frequency Series):**
- **ARIMA** - Autoregressive Integrated Moving Average
- **SARIMA** - Seasonal ARIMA for explicit seasonality
- **Holt-Winters** - Exponential smoothing with trend and seasonality

**For Missions (Seasonal Daily Series):**
- **Prophet** - Facebook's forecasting tool for business time series
- **Naive Baseline** - Persistence model for comparison

### 4. Metrics Evaluation (`04_evaluate_metrics.py`)

Compares all models using:
- **MAE** - Mean Absolute Error
- **RMSE** - Root Mean Squared Error
- **MAPE** - Mean Absolute Percentage Error

### 5. Forecast Visualization (`05_plot_predicted_vs_actual.py`)

- Predicted vs. actual values plots
- **Confidence/prediction intervals**
- Best model selection visualization

### 6. Residual Analysis (`06_residual_analysis.py`)

- Residual distribution plots
- Autocorrelation of residuals
- Q-Q plots for normality assessment
- Diagnostic validation

### 7. Final Report (`07_final_report.py`)

Generates a comprehensive Markdown report including:
- Analysis summary and key findings
- Model comparison tables
- Embedded visualizations
- Recommendations and conclusions

---

## 🚀 How to Run

### Option 1: Run Complete Pipeline

```bash
python 08_orchestrator.py --all
```

### Option 2: Interactive Mode

```bash
python 08_orchestrator.py
```

This opens an interactive menu where you can:
1. Run complete pipeline (all 7 scripts)
2. Run specific steps
3. Run from step N onwards
4. Clean outputs and restart
5. View pipeline status

### Option 3: Run Individual Scripts

```bash
python 01_exploratory_analysis.py
python 02_preprocessing.py
python 03_train_models.py
python 04_evaluate_metrics.py
python 05_plot_predicted_vs_actual.py
python 06_residual_analysis.py
python 07_final_report.py
```

### Utility Commands

```bash
# View pipeline status
python 08_orchestrator.py --status

# Clean all outputs
python 08_orchestrator.py --clean

# Skip dependency checks
python 08_orchestrator.py --all --skip-checks
```

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
statsmodels
prophet
colorama (optional, for colored output)
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy statsmodels prophet
```

---

## 📊 Expected Outputs

### Visualizations (in `outputs/graphics/`)

| File | Description |
|------|-------------|
| `voltage_timeseries.png` | Voltage time series plot |
| `voltage_decomposition.png` | Seasonal decomposition |
| `voltage_acf.png` / `voltage_pacf.png` | Autocorrelation plots |
| `voltage_best_model_forecast.png` | Best model predictions |
| `voltage_residual_analysis.png` | Residual diagnostics |
| `missions_timeseries.png` | Missions time series plot |
| `missions_decomposition.png` | Seasonal decomposition |
| `missions_acf.png` / `missions_pacf.png` | Autocorrelation plots |
| `missions_best_model_forecast.png` | Best model predictions |
| `missions_residual_analysis.png` | Residual diagnostics |

### Reports (in `outputs/`)

- `voltage_metrics_comparison.csv` - Metrics table for voltage models
- `missions_metrics_comparison.csv` - Metrics table for mission models
- `FINAL_REPORT.md` - Comprehensive analysis report

---

## 📚 Key Concepts

### Stationarity
A fundamental assumption for ARIMA models. A stationary series has constant mean and variance over time. The Augmented Dickey-Fuller test helps verify this assumption.

### Temporal Validation
Unlike traditional ML, time series requires **chronological splitting** (train = past, test = future). Random splitting would cause data leakage by allowing the model to "see the future" during training.

### Data Leakage Prevention
- Temporal split ensures no future information is used in training
- Scaler is fit ONLY on training data
- Lag features are carefully constructed to avoid look-ahead bias

### Model Selection Guidelines

| Scenario | Recommended Model |
|----------|-------------------|
| Simple trend, no seasonality | ARIMA |
| Clear seasonal patterns | SARIMA, Holt-Winters |
| Multiple seasonalities, holidays | Prophet |
| High-frequency data | ARIMA, Exponential Smoothing |

---

## 📈 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | Mean(\|actual - predicted\|) | Average absolute error in original units |
| **RMSE** | √Mean((actual - predicted)²) | Penalizes large errors more heavily |
| **MAPE** | Mean(\|actual - predicted\| / \|actual\|) × 100 | Percentage error, scale-independent |

---

## 🔬 Challenges and Lessons

1. **Stationarity**: Understanding why it's fundamental for ARIMA models and how transformations (differencing, log) help achieve it.

2. **Temporal Validation**: Traditional cross-validation doesn't work for time series. Use walk-forward validation or chronological train-test split.

3. **Model Trade-offs**: 
   - Classical models (ARIMA) require more manual tuning but are interpretable
   - Modern tools (Prophet) are more automated but may be less transparent

---

## 👤 Author

**Bruno Silva** - 2025

---

## 📄 License

This project is part of an academic laboratory exercise.
