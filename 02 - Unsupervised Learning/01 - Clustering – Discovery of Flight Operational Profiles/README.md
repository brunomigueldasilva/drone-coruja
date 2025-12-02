# Lab 07.1 - Clustering: Discovery of Flight Operational Profiles

## Overview

This laboratory project applies **unsupervised learning** techniques to identify and analyze different **flight operational profiles** from telemetry data. Instead of predicting a known value, the algorithms group flights with similar characteristics, discovering patterns that are not obvious at first glance.

**Theme:** Unsupervised Learning - Clustering

## Objective

Use clustering algorithms to identify distinct flight profiles based on telemetry metrics. The goal is to let the algorithm discover natural groupings in the data, revealing hidden operational patterns across different flights.

## Dataset

The project uses `voos_telemetria_completa.csv`, containing aggregated metrics per flight:

| Feature | Description |
|---------|-------------|
| `duracao_voo_min` | Flight duration (minutes) |
| `distancia_percorrida_km` | Distance traveled (km) |
| `altitude_maxima_m` | Maximum altitude (meters) |
| `velocidade_media_kmh` | Average speed (km/h) |
| `consumo_combustivel_litros` | Fuel consumption (liters) |
| `variacao_vertical_total_m` | Total vertical variation (meters) |

> **Note:** There is no target variable. The task is to create the groups ourselves.

## Project Structure

```
.
├── inputs/
│   └── voos_telemetria_completa.csv    # Raw flight telemetry data
├── outputs/
│   ├── data_processed/                  # Processed data and summaries
│   ├── graphics/                        # Visualizations (PNG/PDF)
│   ├── models/                          # Trained model artifacts
│   ├── results/                         # Analysis notes and reports
│   └── FINAL_REPORT.md                  # Comprehensive final report
├── 01_exploratory_analysis.py           # EDA and feature analysis
├── 02_preprocessing.py                  # Data preparation and scaling
├── 03_elbow_method.py                   # Optimal k selection
├── 04_training_evaluation.py            # Model training and comparison
├── 05_pca_visualization.py              # Cluster visualization
├── 06_dbscan_profile.py                 # Outlier detection and profiling
├── 07_final_report.py                   # Report generation
├── 08_orchestrator.py                   # Pipeline orchestration
└── README.md
```

## Pipeline Stages

### 1. Exploratory Data Analysis (`01_exploratory_analysis.py`)

- Analyzes the distribution and scale of each variable
- Creates comprehensive visualizations (histograms, boxplots, pairplots, correlation heatmap)
- Generates schema summary and analytical recommendations
- Identifies potential outliers and data quality issues

### 2. Preprocessing (`02_preprocessing.py`)

- **Data Normalization/Scaling** using `StandardScaler`
- Handles missing values and data quality issues
- Prepares feature matrix for clustering algorithms

> **Why is scaling crucial?** Distance-based algorithms like K-Means are sensitive to feature scales. Without normalization, features with larger ranges would dominate the distance calculations, leading to biased cluster assignments.

### 3. Elbow Method (`03_elbow_method.py`)

- Implements the **Elbow Method** to find optimal number of clusters (k)
- Visualizes inertia (WCSS - Within-Cluster Sum of Squares) vs. number of clusters
- Computes **Silhouette Score** for different k values
- Provides data-driven recommendation for optimal k

### 4. Model Training & Evaluation (`04_training_evaluation.py`)

**K-Means Clustering:**
- Trains K-Means with optimal k determined from elbow analysis
- Assigns cluster labels to each flight

**Hierarchical Agglomerative Clustering:**
- Creates **dendrogram** to visualize hierarchical structure
- Uses Ward linkage for cluster formation
- Compares results with K-Means

**DBSCAN (Density-Based):**
- Detects outliers and noise points
- Grid search for optimal `eps` and `min_samples` parameters
- Provides robust clustering without requiring k specification

**Evaluation Metrics:**
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

### 5. PCA Visualization (`05_pca_visualization.py`)

- Reduces dimensionality using **Principal Component Analysis (PCA)**
- Creates 2D scatter plots colored by cluster assignments
- Visualizes cluster separation and structure
- Reports variance explained by principal components

### 6. Cluster Profiling (`06_dbscan_profile.py`)

- Analyzes characteristics of each cluster
- Computes mean values of each feature per cluster
- Creates **flight profile personas**, such as:
  - *"Short Flights at Low Altitude"*
  - *"Long-Distance High-Speed Missions"*
  - *"High Vertical Maneuver Flights"*

### 7. Final Report Generation (`07_final_report.py`)

Compiles comprehensive report including:
- Preprocessing methodology with normalization justification
- Elbow Method and Silhouette Score visualizations
- Comparison of K-Means vs. Hierarchical Clustering results
- Detailed cluster profile descriptions and interpretations

## Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

Optional (for colored output):
```bash
pip install colorama
```

### Running the Pipeline

**Interactive Mode:**
```bash
python 08_orchestrator.py
```

**Run Complete Pipeline:**
```bash
python 08_orchestrator.py --all
```

**Run Specific Steps:**
```bash
python 08_orchestrator.py --steps 1,3,4    # Run EDA, elbow method, and training
```

**Run from Specific Step:**
```bash
python 08_orchestrator.py --from 4         # Run from training onwards
```

**Clean All Outputs:**
```bash
python 08_orchestrator.py --clean
```

### Running Individual Scripts

Each script can also be executed independently:

```bash
python 01_exploratory_analysis.py
python 02_preprocessing.py
python 03_elbow_method.py
python 04_training_evaluation.py
python 05_pca_visualization.py
python 06_dbscan_profile.py
python 07_final_report.py
```

## Key Challenges and Lessons

### 1. The Importance of Normalization

Compare K-Means results with and without normalization to observe the dramatic impact that variable scales can have on clustering outcomes.

### 2. The Subjectivity of 'k'

The choice of number of clusters (k) doesn't always have a unique answer. It may depend on:
- Business objectives
- Domain knowledge
- Interpretability requirements

### 3. Handling Outliers

Outliers can significantly affect K-Means centroids. Strategies include:
- Outlier removal before clustering
- Using robust algorithms like **DBSCAN**
- Winsorization or capping extreme values

## Expected Outputs

### Visualizations
- `histograms.png` - Feature distributions
- `boxplots.png` - Outlier detection
- `pairplot.png` - Feature relationships
- `correlation_heatmap.png` - Feature correlations
- `elbow_wcss.png` - Elbow method plot
- `dendrogram.png` - Hierarchical clustering structure
- `pca_kmeans.png` - K-Means clusters in PCA space
- `pca_dbscan.png` - DBSCAN results in PCA space

### Data Files
- `processed_X.npy` - Scaled feature matrix
- `model_comparison.csv` - Algorithm performance metrics
- `cluster_profile_means.csv` - Cluster centroids
- `elbow_wcss.csv` - Inertia values for different k

### Reports
- `FINAL_REPORT.md` - Comprehensive analysis report
- Individual notes files for each pipeline stage

## Author

Bruno Silva, 2025

## License

This project is part of an academic laboratory exercise on unsupervised machine learning.
