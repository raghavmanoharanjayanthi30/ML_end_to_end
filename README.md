# Meridian Health Churn Prediction Pipeline

A production-ready ML pipeline for predicting member churn at Meridian Health, built with strict data governance and MLOps best practices.

## Loom Walkthrough

[Watch the project walkthrough](https://www.loom.com/share/f28e344274184bafbf7b277d8744a4c4)

## Project Overview

This pipeline predicts which health insurance members are likely to churn (not renew their coverage), enabling proactive retention interventions. The system is designed with healthcare data governance requirements in mind, excluding sensitive demographic features and preventing data leakage.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Raw Data          Validation         Feature Eng.        Training     │
│   ─────────         ──────────         ────────────        ────────     │
│   train.csv    →    Great Exp.    →    sklearn       →    GridSearchCV  │
│   test.csv          + custom           Pipeline           + MLflow      │
│                     checks                                               │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                           SERVING & MONITORING                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   FastAPI           Batch CLI          Monitoring                        │
│   ───────           ─────────          ──────────                        │
│   /predict          predict_batch()    Data drift (KS, Chi-sq)          │
│   /predict/batch                       Prediction drift                  │
│   /upload                              Concept drift                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
module_5_project/
├── data/
│   ├── raw/                    # Original data (DVC tracked)
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── train.csv.dvc       # DVC pointer files
│   │   └── test.csv.dvc
│   └── processed/              # Feature-engineered data
├── src/
│   ├── ingest.py               # Data loading & validation
│   ├── features.py             # Feature engineering pipeline
│   ├── train.py                # Model training with MLflow
│   ├── evaluate.py             # Model evaluation & metrics
│   ├── serve.py                # FastAPI + batch inference
│   └── monitor.py              # Drift detection & alerts
├── tests/                      # pytest test suite
├── notebooks/
│   └── exploration.ipynb       # EDA notebook
├── models/                     # Saved models & pipelines
├── mlruns/                     # MLflow experiment tracking
├── .github/workflows/
│   └── ci.yml                  # CI/CD pipeline
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd module_5_project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull data files (if using DVC remote)
dvc pull
```

## Usage

### 1. Data Validation

```bash
python -c "from src.ingest import validate_data_simple; validate_data_simple('data/raw/train.csv')"
```

### 2. Train Model

```bash
python src/train.py
```

This will:
- Train Logistic Regression, Random Forest, and Gradient Boosting models
- Perform hyperparameter tuning with GridSearchCV
- Log all experiments to MLflow
- Save the best model based on precision@10%

### 3. Evaluate Model

```bash
python src/evaluate.py
```

Generates:
- Aggregate metrics (AUC, precision@10%, recall@10%, F1, calibration error)
- Subgroup metrics by plan type and state
- Calibration curves and ROC/PR plots

### 4. Serve Predictions

**FastAPI Server:**
```bash
python src/serve.py --mode api
# Server runs at http://localhost:8000
```

API Endpoints:
- `GET /health` - Health check
- `GET /model/metadata` - Model info
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions (JSON)
- `POST /upload` - Batch predictions (CSV upload)

**Batch CLI:**
```bash
python src/serve.py --mode batch --input data/raw/test.csv --output predictions.csv
```

### 5. Run Monitoring

```bash
python src/monitor.py
```

Detects:
- **Data drift** - Feature distribution changes (KS test, Chi-square)
- **Prediction drift** - Output distribution changes
- **Concept drift** - Performance degradation over time

## Data Governance

### Blocked Features (Fairness)
The following features are **excluded** from the model to prevent discriminatory predictions:
- `gender` - Protected attribute
- `race` - Protected attribute
- `mental_health_diagnosis` - Sensitive health information
- `dual_eligible` - Proxy for socioeconomic status

### Leakage Features (Integrity)
The following features are **excluded** to prevent data leakage:
- `days_to_renewal_decision` - Future information
- `cancellation_form_started` - Direct indicator of outcome
- `renewal_decision_date` - Future information

### Allowed Features
```python
ALLOWED_FEATURES = [
    'age',                      # Demographics (non-protected)
    'state',                    # Geographic
    'is_rural',                 # Geographic
    'plan_type',                # Product
    'tenure_years',             # Engagement
    'chronic_conditions',       # Health (non-sensitive)
    'claims_last_12mo',         # Utilization
    'denied_claims_last_12mo',  # Experience
    'premium_increase_pct',     # Financial
    'service_calls_last_12mo',  # Engagement
    'last_satisfaction_score',  # Sentiment
]
```

## Model Performance

| Metric | Value |
|--------|-------|
| AUC | ~0.75-0.80 |
| Precision@10% | ~0.55-0.65 |
| Recall@10% | ~0.20-0.25 |
| Calibration Error | <0.05 |

*Note: Actual values depend on training run. Check MLflow for specific experiment results.*

## Monitoring Strategy

### Data Drift Detection
- **Numeric features**: Kolmogorov-Smirnov test (alert if p < 0.01)
- **Categorical features**: Chi-square test (alert if p < 0.01)
- Runs on each feature independently

### Prediction Drift Detection
- KS test on predicted probabilities
- Mean/std shift detection
- Alert if distribution significantly changes

### Concept Drift Detection
- Compare AUC and Precision@10% between periods
- Alert if performance drops >5%
- Monitor base churn rate shifts

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ --cov=src
```

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR:
1. **test** - Runs pytest and data validation smoke test
2. **lint** - Checks code formatting (black, isort, flake8)
3. **validate-features** - Ensures no blocked/leakage features in pipeline

## Data Versioning (DVC)

Data files are tracked with DVC:

```bash
# After modifying data
dvc add data/raw/train.csv data/raw/test.csv
git add data/raw/*.dvc
git commit -m "Update training data"

# Push data to remote storage (if configured)
dvc push

# Pull data on another machine
dvc pull
```

## MLflow Experiment Tracking

View experiment results:
```bash
mlflow ui
# Open http://localhost:5000
```

## License

Internal use only - Meridian Health
