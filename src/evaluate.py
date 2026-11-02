"""
Model evaluation module.

This module evaluates a trained model on the test set and produces:
- Aggregate metrics (AUC, Precision@10%, Recall@10%, F1)
- Subgroup performance by plan_type
- Calibration curve
- Confusion matrix

Usage:
    python -m src.evaluate --model <run_id> --data data/raw/test.csv
    python -m src.evaluate --model latest --data data/raw/test.csv
"""

import argparse
import sys
import os
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

from src.features import (
    create_feature_pipeline, FEATURE_COLUMNS, TARGET,
    get_feature_names, BLOCKED_FEATURES
)
from src.ingest import validate_data_simple
from src.train import precision_at_k, recall_at_k


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def calculate_metrics(y_true, y_proba, y_pred=None, threshold=0.5):
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        y_pred: Predicted labels (optional, will compute from threshold)
        threshold: Classification threshold
    
    Returns:
        Dict of metrics
    """
    if y_pred is None:
        y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_true, y_proba),
        'precision_at_10': precision_at_k(y_true, y_proba, k=0.1),
        'recall_at_10': recall_at_k(y_true, y_proba, k=0.1),
        'precision_at_5': precision_at_k(y_true, y_proba, k=0.05),
        'recall_at_5': recall_at_k(y_true, y_proba, k=0.05),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'average_precision': average_precision_score(y_true, y_proba),
        'n_samples': len(y_true),
        'n_positive': int(np.sum(y_true)),
        'base_rate': float(np.mean(y_true))
    }
    
    return metrics


def calculate_subgroup_metrics(df, y_proba, subgroup_col='plan_type'):
    """
    Calculate metrics for each subgroup.
    
    Args:
        df: DataFrame with features and target
        y_proba: Predicted probabilities
        subgroup_col: Column to group by
    
    Returns:
        Dict of subgroup metrics
    """
    subgroup_metrics = {}
    
    for group in df[subgroup_col].unique():
        mask = df[subgroup_col] == group
        y_true_group = df.loc[mask, TARGET].values
        y_proba_group = y_proba[mask]
        
        if len(y_true_group) > 0 and len(np.unique(y_true_group)) > 1:
            metrics = calculate_metrics(y_true_group, y_proba_group)
            subgroup_metrics[group] = metrics
        else:
            subgroup_metrics[group] = {'error': 'Insufficient data or single class'}
    
    return subgroup_metrics


def calculate_calibration_error(y_true, y_proba, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        ECE value
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_proba[mask])
            bin_size = np.sum(mask) / len(y_true)
            ece += bin_size * np.abs(bin_acc - bin_conf)
    
    return ece


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_calibration_curve(y_true, y_proba, output_path=None):
    """Plot calibration curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration curve
    ax1 = axes[0]
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax1.plot(prob_pred, prob_true, 's-', label='Model')
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Fraction of positives')
    ax1.set_title('Calibration Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predictions
    ax2 = axes[1]
    ax2.hist(y_proba[y_true == 0], bins=50, alpha=0.5, label='Negative (Renewed)', density=True)
    ax2.hist(y_proba[y_true == 1], bins=50, alpha=0.5, label='Positive (Churned)', density=True)
    ax2.set_xlabel('Predicted probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Prediction Distribution by Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration curve to {output_path}")
    
    plt.close()
    return fig


def plot_confusion_matrix(y_true, y_pred, output_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Renewed', 'Churned'],
           yticklabels=['Renewed', 'Churned'],
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")
    
    plt.close()
    return fig


def plot_roc_curve(y_true, y_proba, output_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to {output_path}")
    
    plt.close()
    return fig


def plot_precision_recall_curve(y_true, y_proba, output_path=None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'Model (AP = {ap:.3f})')
    ax.axhline(y=np.mean(y_true), color='k', linestyle='--', label=f'Baseline ({np.mean(y_true):.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved PR curve to {output_path}")
    
    plt.close()
    return fig


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate(
    run_id: str,
    data_path: str = "data/raw/test.csv",
    output_dir: str = "evaluation_results"
) -> dict:
    """
    Evaluate a model on the test set.
    
    Args:
        run_id: MLflow run ID or 'latest'
        data_path: Path to test data
        output_dir: Directory to save results
    
    Returns:
        Dict with all evaluation results
    """
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Get run ID
    if run_id == 'latest':
        # Read from saved file
        if os.path.exists("artifacts/best_run_id.txt"):
            with open("artifacts/best_run_id.txt", 'r') as f:
                run_id = f.read().strip()
            print(f"Using latest best run: {run_id}")
        else:
            raise ValueError("No best_run_id.txt found. Run training first.")
    
    print(f"\nRun ID: {run_id}")
    print(f"Test data: {data_path}")
    
    # Load model
    print("\n" + "-"*60)
    print("Loading model...")
    print("-"*60)
    
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✓ Model loaded: {type(model).__name__}")
    
    # Load feature pipeline
    pipeline_path = f"mlruns/1/{run_id}/artifacts/feature_pipeline.pkl"
    
    # Try alternative paths
    if not os.path.exists(pipeline_path):
        # Search for the pipeline
        import glob
        pipeline_files = glob.glob(f"mlruns/*/{run_id}/artifacts/feature_pipeline.pkl")
        if pipeline_files:
            pipeline_path = pipeline_files[0]
    
    if os.path.exists(pipeline_path):
        with open(pipeline_path, 'rb') as f:
            feature_pipeline = pickle.load(f)
        print(f"✓ Feature pipeline loaded")
    else:
        print("⚠ Feature pipeline not found, creating new one")
        feature_pipeline = create_feature_pipeline()
        # Need to fit on training data
        train_df = pd.read_csv("data/raw/train.csv")
        X_train = train_df[FEATURE_COLUMNS].copy()
        feature_pipeline.fit(X_train)
    
    # Load and validate test data
    print("\n" + "-"*60)
    print("Loading test data...")
    print("-"*60)
    
    df = validate_data_simple(data_path, verbose=False)
    print(f"✓ Loaded {len(df):,} rows")
    
    # Prepare features
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].values
    
    X_transformed = feature_pipeline.transform(X)
    print(f"✓ Features prepared: {X_transformed.shape}")
    
    # Make predictions
    print("\n" + "-"*60)
    print("Making predictions...")
    print("-"*60)
    
    y_proba = model.predict_proba(X_transformed)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    print(f"✓ Predictions generated")
    
    # Calculate metrics
    print("\n" + "-"*60)
    print("Calculating metrics...")
    print("-"*60)
    
    metrics = calculate_metrics(y, y_proba, y_pred)
    metrics['calibration_error'] = calculate_calibration_error(y, y_proba)
    
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-"*40)
    print(f"{'AUC':<25} {metrics['auc']:>10.4f}")
    print(f"{'Precision@10%':<25} {metrics['precision_at_10']:>10.4f}")
    print(f"{'Recall@10%':<25} {metrics['recall_at_10']:>10.4f}")
    print(f"{'Precision@5%':<25} {metrics['precision_at_5']:>10.4f}")
    print(f"{'Recall@5%':<25} {metrics['recall_at_5']:>10.4f}")
    print(f"{'F1 Score':<25} {metrics['f1']:>10.4f}")
    print(f"{'Average Precision':<25} {metrics['average_precision']:>10.4f}")
    print(f"{'Calibration Error (ECE)':<25} {metrics['calibration_error']:>10.4f}")
    print(f"{'Base Rate':<25} {metrics['base_rate']:>10.4f}")
    
    # Subgroup analysis
    print("\n" + "-"*60)
    print("Subgroup Analysis (by plan_type)")
    print("-"*60)
    
    subgroup_metrics = calculate_subgroup_metrics(df, y_proba, 'plan_type')
    
    print(f"\n{'Plan Type':<28} {'AUC':>8} {'P@10%':>8} {'R@10%':>8} {'Base Rate':>10} {'N':>8}")
    print("-"*75)
    
    for group, group_metrics in subgroup_metrics.items():
        if 'error' not in group_metrics:
            print(f"{group:<28} {group_metrics['auc']:>8.4f} {group_metrics['precision_at_10']:>8.4f} "
                  f"{group_metrics['recall_at_10']:>8.4f} {group_metrics['base_rate']:>10.4f} "
                  f"{group_metrics['n_samples']:>8,}")
    
    # Additional subgroup analysis by state
    print("\n" + "-"*60)
    print("Subgroup Analysis (by state)")
    print("-"*60)
    
    state_metrics = calculate_subgroup_metrics(df, y_proba, 'state')
    
    print(f"\n{'State':<10} {'AUC':>8} {'P@10%':>8} {'R@10%':>8} {'Base Rate':>10} {'N':>8}")
    print("-"*55)
    
    for group, group_metrics in state_metrics.items():
        if 'error' not in group_metrics:
            print(f"{group:<10} {group_metrics['auc']:>8.4f} {group_metrics['precision_at_10']:>8.4f} "
                  f"{group_metrics['recall_at_10']:>8.4f} {group_metrics['base_rate']:>10.4f} "
                  f"{group_metrics['n_samples']:>8,}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("\n" + "-"*60)
    print("Generating plots...")
    print("-"*60)
    
    plot_calibration_curve(y, y_proba, f"{output_dir}/calibration_curve.png")
    plot_confusion_matrix(y, y_pred, f"{output_dir}/confusion_matrix.png")
    plot_roc_curve(y, y_proba, f"{output_dir}/roc_curve.png")
    plot_precision_recall_curve(y, y_proba, f"{output_dir}/pr_curve.png")
    
    # Save metrics to JSON
    results = {
        'run_id': run_id,
        'data_path': data_path,
        'aggregate_metrics': metrics,
        'subgroup_metrics_plan_type': subgroup_metrics,
        'subgroup_metrics_state': state_metrics
    }
    
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved metrics to {output_dir}/metrics.json")
    
    # Print classification report
    print("\n" + "-"*60)
    print("Classification Report (threshold=0.5)")
    print("-"*60)
    print(classification_report(y, y_pred, target_names=['Renewed', 'Churned']))
    
    # Log to MLflow
    print("\n" + "-"*60)
    print("Logging to MLflow...")
    print("-"*60)
    
    with mlflow.start_run(run_id=run_id):
        # Log test metrics
        mlflow.log_metric("test_auc", metrics['auc'])
        mlflow.log_metric("test_precision_at_10", metrics['precision_at_10'])
        mlflow.log_metric("test_recall_at_10", metrics['recall_at_10'])
        mlflow.log_metric("test_f1", metrics['f1'])
        mlflow.log_metric("test_calibration_error", metrics['calibration_error'])
        
        # Log artifacts
        mlflow.log_artifacts(output_dir, artifact_path="evaluation")
    
    print("✓ Metrics logged to MLflow")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - metrics.json")
    print(f"  - calibration_curve.png")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - pr_curve.png")
    
    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate churn prediction model on test set'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='latest',
        help='MLflow run ID or "latest"'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/test.csv',
        help='Path to test data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    try:
        results = evaluate(
            run_id=args.model,
            data_path=args.data,
            output_dir=args.output
        )
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
