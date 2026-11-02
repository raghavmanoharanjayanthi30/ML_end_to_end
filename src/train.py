"""
Model training module with MLflow experiment tracking.

This module trains multiple models, performs hyperparameter tuning via
cross-validation, and logs all experiments to MLflow.

Usage:
    python -m src.train
    python -m src.train --experiment-name "churn_v2"
    python -m src.train --quick  # Fast mode for testing
"""

import argparse
import sys
import os
import warnings
from pathlib import Path
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    make_scorer, confusion_matrix
)

from src.features import (
    create_feature_pipeline, prepare_features, 
    FEATURE_COLUMNS, TARGET, get_feature_names
)
from src.ingest import validate_data_simple

warnings.filterwarnings('ignore')


# =============================================================================
# CUSTOM METRICS
# =============================================================================

def precision_at_k(y_true, y_proba, k=0.1):
    """
    Calculate precision at top k% of predictions.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        k: Fraction of top predictions to consider (default 0.1 = top 10%)
    
    Returns:
        Precision at k
    """
    n_top = int(len(y_true) * k)
    if n_top == 0:
        n_top = 1
    
    # Get indices of top k predictions
    top_indices = np.argsort(y_proba)[-n_top:]
    
    # Calculate precision in top k
    y_true_top = np.array(y_true)[top_indices]
    precision = np.mean(y_true_top)
    
    return precision


def recall_at_k(y_true, y_proba, k=0.1):
    """
    Calculate recall at top k% of predictions.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        k: Fraction of top predictions to consider (default 0.1 = top 10%)
    
    Returns:
        Recall at k (fraction of true positives captured in top k)
    """
    n_top = int(len(y_true) * k)
    if n_top == 0:
        n_top = 1
    
    # Get indices of top k predictions
    top_indices = np.argsort(y_proba)[-n_top:]
    
    # Calculate recall
    y_true_top = np.array(y_true)[top_indices]
    total_positives = np.sum(y_true)
    
    if total_positives == 0:
        return 0.0
    
    recall = np.sum(y_true_top) / total_positives
    
    return recall


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

def get_model_configs(quick_mode=False):
    """
    Get model configurations for experiments.
    
    Args:
        quick_mode: If True, use smaller parameter grids for faster testing
    
    Returns:
        Dict of model configurations
    """
    if quick_mode:
        # Minimal configs for quick testing
        return {
            'logistic_regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
                'param_grid': {
                    'C': [1.0],
                    'class_weight': [None]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': [100],
                    'max_depth': [10]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': [100],
                    'max_depth': [3]
                }
            }
        }
    else:
        # Full configs for production
        return {
            'logistic_regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
                'param_grid': {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'class_weight': [None, 'balanced']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, 20],
                    'class_weight': [None, 'balanced']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1]
                }
            }
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_and_evaluate_model(
    model_name: str,
    model,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5
) -> dict:
    """
    Train a model with hyperparameter tuning and cross-validation.
    
    Args:
        model_name: Name of the model
        model: sklearn model instance
        param_grid: Hyperparameter grid for GridSearchCV
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of CV folds
    
    Returns:
        Dict with best model, best params, and CV metrics
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Grid search with AUC as primary metric
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    print(f"Running grid search with {len(param_grid)} parameter(s)...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    # Calculate additional CV metrics with best model
    print("\nCalculating additional CV metrics...")
    
    # Get CV predictions for precision@k and recall@k
    cv_metrics = {
        'cv_auc_mean': grid_search.best_score_,
        'cv_auc_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    }
    
    # Cross-val predict for precision@k and recall@k
    from sklearn.model_selection import cross_val_predict
    y_proba_cv = cross_val_predict(best_model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    
    cv_metrics['cv_precision_at_10'] = precision_at_k(y_train, y_proba_cv, k=0.1)
    cv_metrics['cv_recall_at_10'] = recall_at_k(y_train, y_proba_cv, k=0.1)
    
    # F1 score
    y_pred_cv = (y_proba_cv >= 0.5).astype(int)
    cv_metrics['cv_f1'] = f1_score(y_train, y_pred_cv)
    
    print(f"CV Precision@10%: {cv_metrics['cv_precision_at_10']:.4f}")
    print(f"CV Recall@10%: {cv_metrics['cv_recall_at_10']:.4f}")
    print(f"CV F1: {cv_metrics['cv_f1']:.4f}")
    
    return {
        'model': best_model,
        'best_params': best_params,
        'cv_metrics': cv_metrics,
        'grid_search': grid_search
    }


def run_experiment(
    model_name: str,
    model_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_pipeline,
    cv_folds: int = 5
) -> str:
    """
    Run a single experiment and log to MLflow.
    
    Args:
        model_name: Name of the model
        model_config: Dict with 'model' and 'param_grid'
        X_train: Training features
        y_train: Training labels
        feature_pipeline: Fitted feature pipeline
        cv_folds: Number of CV folds
    
    Returns:
        MLflow run ID
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Log tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("stage", "training")
        
        # Train and evaluate
        results = train_and_evaluate_model(
            model_name=model_name,
            model=model_config['model'],
            param_grid=model_config['param_grid'],
            X_train=X_train,
            y_train=y_train,
            cv_folds=cv_folds
        )
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("n_samples", len(y_train))
        mlflow.log_param("n_features", X_train.shape[1])
        
        for param_name, param_value in results['best_params'].items():
            mlflow.log_param(f"best_{param_name}", param_value)
        
        # Log metrics
        for metric_name, metric_value in results['cv_metrics'].items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Refit best model on all training data
        print("\nRefitting best model on all training data...")
        best_model = results['model']
        best_model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(
            best_model,
            "model",
            registered_model_name=f"churn_{model_name}"
        )
        
        # Log feature pipeline
        pipeline_path = "artifacts/feature_pipeline.pkl"
        os.makedirs("artifacts", exist_ok=True)
        with open(pipeline_path, 'wb') as f:
            pickle.dump(feature_pipeline, f)
        mlflow.log_artifact(pipeline_path)
        
        # Log feature names
        feature_names = get_feature_names(feature_pipeline)
        with open("artifacts/feature_names.txt", 'w') as f:
            f.write('\n'.join(feature_names))
        mlflow.log_artifact("artifacts/feature_names.txt")
        
        print(f"\n✓ Experiment logged with run_id: {run.info.run_id}")
        
        return run.info.run_id


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(
    data_path: str = "data/raw/train.csv",
    experiment_name: str = "churn_prediction",
    quick_mode: bool = False,
    cv_folds: int = 5
) -> dict:
    """
    Main training function.
    
    Args:
        data_path: Path to training data
        experiment_name: MLflow experiment name
        quick_mode: If True, use smaller parameter grids
        cv_folds: Number of CV folds
    
    Returns:
        Dict with run IDs and best model info
    """
    print("="*60)
    print("CHURN PREDICTION MODEL TRAINING")
    print("="*60)
    print(f"\nData path: {data_path}")
    print(f"Experiment: {experiment_name}")
    print(f"Quick mode: {quick_mode}")
    print(f"CV folds: {cv_folds}")
    
    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Load and validate data
    print("\n" + "-"*60)
    print("Loading and validating data...")
    print("-"*60)
    df = validate_data_simple(data_path, verbose=False)
    print(f"✓ Loaded {len(df):,} rows")
    
    # Prepare features
    print("\n" + "-"*60)
    print("Preparing features...")
    print("-"*60)
    feature_pipeline = create_feature_pipeline()
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].values
    
    X_transformed = feature_pipeline.fit_transform(X)
    print(f"✓ Features prepared: {X_transformed.shape}")
    
    # Get model configurations
    model_configs = get_model_configs(quick_mode=quick_mode)
    
    # Run experiments for each model
    print("\n" + "-"*60)
    print(f"Running {len(model_configs)} experiments...")
    print("-"*60)
    
    run_ids = {}
    results = {}
    
    for model_name, model_config in model_configs.items():
        run_id = run_experiment(
            model_name=model_name,
            model_config=model_config,
            X_train=X_transformed,
            y_train=y,
            feature_pipeline=feature_pipeline,
            cv_folds=cv_folds
        )
        run_ids[model_name] = run_id
        
        # Get metrics for comparison
        run = mlflow.get_run(run_id)
        results[model_name] = {
            'run_id': run_id,
            'cv_auc': run.data.metrics.get('cv_auc_mean', 0),
            'cv_precision_at_10': run.data.metrics.get('cv_precision_at_10', 0),
            'cv_recall_at_10': run.data.metrics.get('cv_recall_at_10', 0)
        }
    
    # Find best model
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<25} {'CV AUC':>10} {'P@10%':>10} {'R@10%':>10}")
    print("-"*60)
    
    best_model_name = None
    best_precision = 0
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['cv_auc']:>10.4f} {metrics['cv_precision_at_10']:>10.4f} {metrics['cv_recall_at_10']:>10.4f}")
        
        # Select best by precision@10% (primary business metric)
        if metrics['cv_precision_at_10'] > best_precision:
            best_precision = metrics['cv_precision_at_10']
            best_model_name = model_name
    
    print("-"*60)
    print(f"\n✓ Best model (by Precision@10%): {best_model_name}")
    print(f"  Run ID: {run_ids[best_model_name]}")
    
    # Save best run ID for later use
    with open("artifacts/best_run_id.txt", 'w') as f:
        f.write(run_ids[best_model_name])
    
    print(f"\n✓ Training complete! View experiments with: mlflow ui")
    
    return {
        'run_ids': run_ids,
        'results': results,
        'best_model': best_model_name,
        'best_run_id': run_ids[best_model_name]
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train churn prediction models with MLflow tracking'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/train.csv',
        help='Path to training data'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='churn_prediction',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode with smaller parameter grids'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    args = parser.parse_args()
    
    try:
        results = train(
            data_path=args.data,
            experiment_name=args.experiment_name,
            quick_mode=args.quick,
            cv_folds=args.cv_folds
        )
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
