"""
Monitoring module for churn prediction model.

This module provides instrumentation for production monitoring:
- Prediction logging
- Data drift detection
- Prediction drift detection
- Alert threshold checking

The functions are designed to compare two DataFrames (baseline vs current),
which in production could be:
- Training data vs current month's data
- Last 30 days vs last 15 days
- Previous month vs current month

Usage:
    # Compare train vs test (simulation)
    python -m src.monitor --baseline data/raw/train.csv --current data/raw/test.csv
    
    # In production, you'd call these functions with your time windows
    from src.monitor import detect_drift, check_alerts
    drift_report = detect_drift(baseline_df, current_df)
    alerts = check_alerts(drift_report)
"""

import argparse
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.features import FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET


# =============================================================================
# CONFIGURATION - MONITORING THRESHOLDS
# =============================================================================

# These thresholds define when to alert. Adjust based on business needs.
MONITORING_CONFIG = {
    # Data drift thresholds
    "data_drift": {
        # For numeric features: alert if KS statistic > threshold
        "ks_statistic_threshold": 0.1,
        # For categorical features: alert if chi-square p-value < threshold
        "chi2_pvalue_threshold": 0.01,
        # Alert if more than this fraction of features have drifted
        "max_drifted_features_ratio": 0.3,
    },
    
    # Prediction drift thresholds
    "prediction_drift": {
        # Alert if mean predicted probability shifts by more than this
        "mean_shift_threshold": 0.05,
        # Alert if std of predictions changes by more than this ratio
        "std_change_ratio_threshold": 0.2,
        # Alert if KS statistic between prediction distributions > threshold
        "ks_statistic_threshold": 0.1,
    },
    
    # Performance thresholds (when outcomes are available)
    "performance": {
        # Alert if AUC drops below this
        "min_auc": 0.70,
        # Alert if precision@10% drops below this
        "min_precision_at_10": 0.50,
    },
    
    # Data quality thresholds
    "data_quality": {
        # Alert if null rate increases by more than this
        "null_rate_increase_threshold": 0.05,
        # Alert if row count drops below this ratio of expected
        "min_row_count_ratio": 0.8,
    }
}


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Set up logging for monitoring."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("churn_monitor")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(f"{log_dir}/monitor_{datetime.now().strftime('%Y%m%d')}.log")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# =============================================================================
# PREDICTION LOGGING
# =============================================================================

class PredictionLogger:
    """
    Logs predictions for monitoring.
    
    In production, this would write to a database or message queue.
    For this project, we write to a CSV file.
    """
    
    def __init__(self, log_path: str = "logs/predictions.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Initialize file with headers if it doesn't exist
        if not os.path.exists(log_path):
            headers = ["timestamp", "member_id", "churn_probability", "risk_tier"] + FEATURE_COLUMNS
            pd.DataFrame(columns=headers).to_csv(log_path, index=False)
    
    def log(self, member_id: str, probability: float, risk_tier: str, features: dict):
        """Log a single prediction."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "member_id": member_id,
            "churn_probability": probability,
            "risk_tier": risk_tier,
            **features
        }
        
        # Append to CSV
        pd.DataFrame([record]).to_csv(self.log_path, mode='a', header=False, index=False)
    
    def log_batch(self, df: pd.DataFrame, probabilities: np.ndarray, risk_tiers: List[str]):
        """Log a batch of predictions."""
        records = df[FEATURE_COLUMNS].copy()
        records["timestamp"] = datetime.now().isoformat()
        records["member_id"] = df["member_id"] if "member_id" in df.columns else range(len(df))
        records["churn_probability"] = probabilities
        records["risk_tier"] = risk_tiers
        
        # Append to CSV
        records.to_csv(self.log_path, mode='a', header=False, index=False)
    
    def get_recent(self, n_days: int = 30) -> pd.DataFrame:
        """Get recent predictions for analysis."""
        if not os.path.exists(self.log_path):
            return pd.DataFrame()
        
        df = pd.read_csv(self.log_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - pd.Timedelta(days=n_days)
        return df[df["timestamp"] >= cutoff]


# =============================================================================
# DRIFT DETECTION
# =============================================================================

def compute_numeric_drift(baseline: pd.Series, current: pd.Series) -> dict:
    """
    Compute drift metrics for a numeric feature using KS test.
    
    Args:
        baseline: Baseline distribution (e.g., training data)
        current: Current distribution (e.g., recent predictions)
    
    Returns:
        Dict with drift metrics
    """
    # Remove nulls for comparison
    baseline_clean = baseline.dropna()
    current_clean = current.dropna()
    
    if len(baseline_clean) == 0 or len(current_clean) == 0:
        return {"error": "Insufficient data"}
    
    # KS test
    ks_stat, ks_pvalue = stats.ks_2samp(baseline_clean, current_clean)
    
    # Basic statistics
    baseline_mean = baseline_clean.mean()
    current_mean = current_clean.mean()
    baseline_std = baseline_clean.std()
    current_std = current_clean.std()
    
    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "baseline_mean": float(baseline_mean),
        "current_mean": float(current_mean),
        "mean_shift": float(current_mean - baseline_mean),
        "mean_shift_pct": float((current_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0,
        "baseline_std": float(baseline_std),
        "current_std": float(current_std),
        "is_drifted": ks_stat > MONITORING_CONFIG["data_drift"]["ks_statistic_threshold"]
    }


def compute_categorical_drift(baseline: pd.Series, current: pd.Series) -> dict:
    """
    Compute drift metrics for a categorical feature using chi-square test.
    
    Args:
        baseline: Baseline distribution
        current: Current distribution
    
    Returns:
        Dict with drift metrics
    """
    # Get value counts
    baseline_counts = baseline.value_counts()
    current_counts = current.value_counts()
    
    # Align categories
    all_categories = set(baseline_counts.index) | set(current_counts.index)
    baseline_aligned = pd.Series({cat: baseline_counts.get(cat, 0) for cat in all_categories})
    current_aligned = pd.Series({cat: current_counts.get(cat, 0) for cat in all_categories})
    
    # Chi-square test
    # Need to handle case where expected frequencies are too low
    try:
        chi2, pvalue = stats.chisquare(current_aligned, baseline_aligned * len(current) / len(baseline))
    except Exception:
        chi2, pvalue = 0, 1
    
    # Distribution comparison
    baseline_dist = baseline.value_counts(normalize=True)
    current_dist = current.value_counts(normalize=True)
    
    return {
        "chi2_statistic": float(chi2),
        "chi2_pvalue": float(pvalue),
        "baseline_distribution": baseline_dist.to_dict(),
        "current_distribution": current_dist.to_dict(),
        "is_drifted": pvalue < MONITORING_CONFIG["data_drift"]["chi2_pvalue_threshold"]
    }


def detect_data_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Detect data drift between baseline and current DataFrames.
    
    Args:
        baseline_df: Baseline data (e.g., training data)
        current_df: Current data (e.g., recent data)
    
    Returns:
        Dict with drift report for all features
    """
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_size": len(baseline_df),
        "current_size": len(current_df),
        "features": {},
        "summary": {}
    }
    
    drifted_features = []
    
    # Check numeric features
    numeric_features = [f for f in NUMERIC_FEATURES if f in baseline_df.columns and f in current_df.columns]
    for feature in numeric_features:
        drift_metrics = compute_numeric_drift(baseline_df[feature], current_df[feature])
        drift_report["features"][feature] = drift_metrics
        if drift_metrics.get("is_drifted", False):
            drifted_features.append(feature)
    
    # Check categorical features
    categorical_features = [f for f in CATEGORICAL_FEATURES if f in baseline_df.columns and f in current_df.columns]
    for feature in categorical_features:
        drift_metrics = compute_categorical_drift(baseline_df[feature], current_df[feature])
        drift_report["features"][feature] = drift_metrics
        if drift_metrics.get("is_drifted", False):
            drifted_features.append(feature)
    
    # Summary
    total_features = len(numeric_features) + len(categorical_features)
    drift_report["summary"] = {
        "total_features_checked": total_features,
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_ratio": len(drifted_features) / total_features if total_features > 0 else 0
    }
    
    return drift_report


def detect_concept_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    baseline_proba: np.ndarray,
    current_proba: np.ndarray
) -> dict:
    """
    Detect concept drift by comparing model performance across time periods.
    
    Concept drift occurs when the relationship between features and target changes,
    even if the feature distributions remain the same.
    
    REQUIRES: Both DataFrames must have the target column (churned) with actual outcomes.
    
    Args:
        baseline_df: Baseline data with actual outcomes
        current_df: Current data with actual outcomes
        baseline_proba: Model predictions on baseline data
        current_proba: Model predictions on current data
    
    Returns:
        Dict with concept drift metrics
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from src.train import precision_at_k, recall_at_k
    
    # Check if we have labels
    if TARGET not in baseline_df.columns or TARGET not in current_df.columns:
        return {"error": "Target column not available - cannot measure concept drift"}
    
    baseline_y = baseline_df[TARGET].values
    current_y = current_df[TARGET].values
    
    # Calculate performance metrics for both periods
    baseline_metrics = {
        "auc": roc_auc_score(baseline_y, baseline_proba),
        "precision_at_10": precision_at_k(baseline_y, baseline_proba, k=0.1),
        "recall_at_10": recall_at_k(baseline_y, baseline_proba, k=0.1),
        "brier_score": brier_score_loss(baseline_y, baseline_proba),
        "base_rate": float(np.mean(baseline_y))
    }
    
    current_metrics = {
        "auc": roc_auc_score(current_y, current_proba),
        "precision_at_10": precision_at_k(current_y, current_proba, k=0.1),
        "recall_at_10": recall_at_k(current_y, current_proba, k=0.1),
        "brier_score": brier_score_loss(current_y, current_proba),
        "base_rate": float(np.mean(current_y))
    }
    
    # Calculate performance degradation
    auc_drop = baseline_metrics["auc"] - current_metrics["auc"]
    precision_drop = baseline_metrics["precision_at_10"] - current_metrics["precision_at_10"]
    base_rate_shift = current_metrics["base_rate"] - baseline_metrics["base_rate"]
    
    # Determine if concept drift occurred
    is_drifted = (
        auc_drop > 0.05 or  # AUC dropped by more than 5 points
        current_metrics["auc"] < MONITORING_CONFIG["performance"]["min_auc"] or
        current_metrics["precision_at_10"] < MONITORING_CONFIG["performance"]["min_precision_at_10"]
    )
    
    return {
        "timestamp": datetime.now().isoformat(),
        "baseline_metrics": baseline_metrics,
        "current_metrics": current_metrics,
        "auc_drop": float(auc_drop),
        "precision_at_10_drop": float(precision_drop),
        "base_rate_shift": float(base_rate_shift),
        "is_drifted": is_drifted,
        "interpretation": {
            "auc_change": f"{'Degraded' if auc_drop > 0.02 else 'Stable'} ({auc_drop:+.3f})",
            "base_rate_change": f"{'Increased' if base_rate_shift > 0 else 'Decreased'} churn ({base_rate_shift:+.1%})",
            "recommendation": "Retrain model" if is_drifted else "Continue monitoring"
        }
    }


def detect_prediction_drift(baseline_proba: np.ndarray, current_proba: np.ndarray) -> dict:
    """
    Detect drift in model predictions.
    
    Args:
        baseline_proba: Baseline predicted probabilities
        current_proba: Current predicted probabilities
    
    Returns:
        Dict with prediction drift metrics
    """
    # KS test on predictions
    ks_stat, ks_pvalue = stats.ks_2samp(baseline_proba, current_proba)
    
    # Statistics
    baseline_mean = np.mean(baseline_proba)
    current_mean = np.mean(current_proba)
    baseline_std = np.std(baseline_proba)
    current_std = np.std(current_proba)
    
    # Risk tier distribution
    def get_risk_distribution(proba):
        high = np.mean(proba >= 0.5)
        medium = np.mean((proba >= 0.3) & (proba < 0.5))
        low = np.mean(proba < 0.3)
        return {"high": high, "medium": medium, "low": low}
    
    baseline_risk_dist = get_risk_distribution(baseline_proba)
    current_risk_dist = get_risk_distribution(current_proba)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "baseline_mean": float(baseline_mean),
        "current_mean": float(current_mean),
        "mean_shift": float(current_mean - baseline_mean),
        "baseline_std": float(baseline_std),
        "current_std": float(current_std),
        "std_change_ratio": float((current_std - baseline_std) / baseline_std) if baseline_std > 0 else 0,
        "baseline_risk_distribution": baseline_risk_dist,
        "current_risk_distribution": current_risk_dist,
        "is_drifted": ks_stat > MONITORING_CONFIG["prediction_drift"]["ks_statistic_threshold"]
    }


# =============================================================================
# ALERT CHECKING
# =============================================================================

def check_alerts(drift_report: dict, prediction_drift: dict = None, concept_drift: dict = None) -> List[dict]:
    """
    Check drift reports against thresholds and generate alerts.
    
    Args:
        drift_report: Output from detect_data_drift()
        prediction_drift: Output from detect_prediction_drift() (optional)
        concept_drift: Output from detect_concept_drift() (optional)
    
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    # Check data drift
    drift_ratio = drift_report["summary"]["drift_ratio"]
    if drift_ratio > MONITORING_CONFIG["data_drift"]["max_drifted_features_ratio"]:
        alerts.append({
            "type": "DATA_DRIFT",
            "severity": "HIGH",
            "message": f"High data drift detected: {drift_ratio:.1%} of features have drifted",
            "details": drift_report["summary"]["drifted_features"],
            "threshold": MONITORING_CONFIG["data_drift"]["max_drifted_features_ratio"],
            "actual": drift_ratio
        })
    
    # Check individual feature drift
    for feature, metrics in drift_report["features"].items():
        if metrics.get("is_drifted", False):
            alerts.append({
                "type": "FEATURE_DRIFT",
                "severity": "MEDIUM",
                "message": f"Feature '{feature}' has drifted significantly",
                "details": metrics,
                "feature": feature
            })
    
    # Check prediction drift
    if prediction_drift:
        if prediction_drift.get("is_drifted", False):
            alerts.append({
                "type": "PREDICTION_DRIFT",
                "severity": "HIGH",
                "message": f"Prediction distribution has shifted (KS={prediction_drift['ks_statistic']:.3f})",
                "details": prediction_drift,
                "threshold": MONITORING_CONFIG["prediction_drift"]["ks_statistic_threshold"],
                "actual": prediction_drift["ks_statistic"]
            })
        
        mean_shift = abs(prediction_drift["mean_shift"])
        if mean_shift > MONITORING_CONFIG["prediction_drift"]["mean_shift_threshold"]:
            alerts.append({
                "type": "PREDICTION_MEAN_SHIFT",
                "severity": "MEDIUM",
                "message": f"Mean predicted probability shifted by {mean_shift:.3f}",
                "details": {
                    "baseline_mean": prediction_drift["baseline_mean"],
                    "current_mean": prediction_drift["current_mean"]
                },
                "threshold": MONITORING_CONFIG["prediction_drift"]["mean_shift_threshold"],
                "actual": mean_shift
            })
    
    # Check concept drift
    if concept_drift and "error" not in concept_drift:
        if concept_drift.get("is_drifted", False):
            alerts.append({
                "type": "CONCEPT_DRIFT",
                "severity": "HIGH",
                "message": f"Model performance degraded (AUC drop: {concept_drift['auc_drop']:.3f})",
                "details": {
                    "baseline_auc": concept_drift["baseline_metrics"]["auc"],
                    "current_auc": concept_drift["current_metrics"]["auc"],
                    "base_rate_shift": concept_drift["base_rate_shift"]
                },
                "recommendation": "Retrain model with recent data"
            })
        
        # Alert on base rate shift (even if performance hasn't degraded yet)
        if abs(concept_drift["base_rate_shift"]) > 0.03:  # 3 percentage point shift
            alerts.append({
                "type": "BASE_RATE_SHIFT",
                "severity": "MEDIUM",
                "message": f"Target base rate shifted by {concept_drift['base_rate_shift']:.1%}",
                "details": {
                    "baseline_base_rate": concept_drift["baseline_metrics"]["base_rate"],
                    "current_base_rate": concept_drift["current_metrics"]["base_rate"]
                }
            })
    
    return alerts


# =============================================================================
# MONITORING REPORT
# =============================================================================

def generate_monitoring_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    baseline_proba: np.ndarray = None,
    current_proba: np.ndarray = None,
    output_path: str = None
) -> dict:
    """
    Generate a comprehensive monitoring report.
    
    Args:
        baseline_df: Baseline data
        current_df: Current data
        baseline_proba: Baseline predictions (optional)
        current_proba: Current predictions (optional)
        output_path: Path to save report (optional)
    
    Returns:
        Complete monitoring report
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "baseline_period": "training_data",
        "current_period": "test_data",
        "data_drift": None,
        "prediction_drift": None,
        "concept_drift": None,
        "alerts": [],
        "recommendations": []
    }
    
    # Data drift
    report["data_drift"] = detect_data_drift(baseline_df, current_df)
    
    # Prediction drift (if predictions provided)
    if baseline_proba is not None and current_proba is not None:
        report["prediction_drift"] = detect_prediction_drift(baseline_proba, current_proba)
        
        # Concept drift (if labels available)
        if TARGET in baseline_df.columns and TARGET in current_df.columns:
            report["concept_drift"] = detect_concept_drift(
                baseline_df, current_df, baseline_proba, current_proba
            )
    
    # Check alerts
    report["alerts"] = check_alerts(
        report["data_drift"],
        report["prediction_drift"],
        report.get("concept_drift")
    )
    
    # Generate recommendations
    if len(report["alerts"]) > 0:
        high_severity = [a for a in report["alerts"] if a["severity"] == "HIGH"]
        if high_severity:
            report["recommendations"].append("URGENT: High severity alerts detected. Consider retraining the model.")
        
        drifted_features = report["data_drift"]["summary"]["drifted_features"]
        if drifted_features:
            report["recommendations"].append(f"Investigate drifted features: {', '.join(drifted_features)}")
        
        if report["prediction_drift"] and report["prediction_drift"].get("is_drifted"):
            report["recommendations"].append("Prediction distribution has shifted. Validate model performance on recent data.")
        
        if report["concept_drift"] and report["concept_drift"].get("is_drifted"):
            report["recommendations"].append("Concept drift detected. Model performance has degraded. Retrain recommended.")
    else:
        report["recommendations"].append("No significant drift detected. Continue monitoring.")
    
    # Save report
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {output_path}")
    
    return report


# =============================================================================
# MAIN - DEMONSTRATION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Monitor churn prediction model for drift'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='data/raw/train.csv',
        help='Path to baseline data (e.g., training data)'
    )
    parser.add_argument(
        '--current',
        type=str,
        default='data/raw/test.csv',
        help='Path to current data (e.g., recent data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='monitoring_report.json',
        help='Path to save monitoring report'
    )
    parser.add_argument(
        '--with-predictions',
        action='store_true',
        help='Also check prediction drift (requires model)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("MONITORING REPORT")
    print("="*60)
    
    # Load data
    print(f"\nLoading baseline data from {args.baseline}...")
    baseline_df = pd.read_csv(args.baseline)
    print(f"✓ Loaded {len(baseline_df):,} rows")
    
    print(f"\nLoading current data from {args.current}...")
    current_df = pd.read_csv(args.current)
    print(f"✓ Loaded {len(current_df):,} rows")
    
    # Get predictions if requested
    baseline_proba = None
    current_proba = None
    
    if args.with_predictions:
        print("\nLoading model for prediction drift analysis...")
        try:
            import mlflow
            import pickle
            from src.features import create_feature_pipeline
            
            mlflow.set_tracking_uri("file:./mlruns")
            
            with open("artifacts/best_run_id.txt", 'r') as f:
                run_id = f.read().strip()
            
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            
            # Load or create feature pipeline
            import glob
            pipeline_files = glob.glob(f"mlruns/*/{run_id}/artifacts/feature_pipeline.pkl")
            if pipeline_files:
                with open(pipeline_files[0], 'rb') as f:
                    feature_pipeline = pickle.load(f)
            else:
                feature_pipeline = create_feature_pipeline()
                feature_pipeline.fit(baseline_df[FEATURE_COLUMNS])
            
            # Get predictions
            baseline_proba = model.predict_proba(
                feature_pipeline.transform(baseline_df[FEATURE_COLUMNS])
            )[:, 1]
            current_proba = model.predict_proba(
                feature_pipeline.transform(current_df[FEATURE_COLUMNS])
            )[:, 1]
            
            print("✓ Predictions generated")
        except Exception as e:
            print(f"⚠ Could not load model: {e}")
            print("  Skipping prediction drift analysis")
    
    # Generate report
    print("\n" + "-"*60)
    print("Analyzing drift...")
    print("-"*60)
    
    report = generate_monitoring_report(
        baseline_df=baseline_df,
        current_df=current_df,
        baseline_proba=baseline_proba,
        current_proba=current_proba,
        output_path=args.output
    )
    
    # Print summary
    print("\n" + "="*60)
    print("DATA DRIFT SUMMARY")
    print("="*60)
    
    summary = report["data_drift"]["summary"]
    print(f"\nFeatures checked: {summary['total_features_checked']}")
    print(f"Features drifted: {summary['drifted_features_count']}")
    print(f"Drift ratio: {summary['drift_ratio']:.1%}")
    
    if summary["drifted_features"]:
        print(f"\nDrifted features:")
        for feat in summary["drifted_features"]:
            metrics = report["data_drift"]["features"][feat]
            if "ks_statistic" in metrics:
                print(f"  - {feat}: KS={metrics['ks_statistic']:.3f}, mean shift={metrics.get('mean_shift_pct', 0):.1f}%")
            else:
                print(f"  - {feat}: chi2 p-value={metrics.get('chi2_pvalue', 'N/A')}")
    
    if report["prediction_drift"]:
        print("\n" + "="*60)
        print("PREDICTION DRIFT SUMMARY")
        print("="*60)
        pd_report = report["prediction_drift"]
        print(f"\nKS statistic: {pd_report['ks_statistic']:.3f}")
        print(f"Mean shift: {pd_report['mean_shift']:.3f} ({pd_report['baseline_mean']:.3f} → {pd_report['current_mean']:.3f})")
        print(f"\nRisk distribution change:")
        print(f"  High risk:   {pd_report['baseline_risk_distribution']['high']:.1%} → {pd_report['current_risk_distribution']['high']:.1%}")
        print(f"  Medium risk: {pd_report['baseline_risk_distribution']['medium']:.1%} → {pd_report['current_risk_distribution']['medium']:.1%}")
        print(f"  Low risk:    {pd_report['baseline_risk_distribution']['low']:.1%} → {pd_report['current_risk_distribution']['low']:.1%}")
    
    if report.get("concept_drift") and "error" not in report["concept_drift"]:
        print("\n" + "="*60)
        print("CONCEPT DRIFT SUMMARY")
        print("="*60)
        cd_report = report["concept_drift"]
        print(f"\nPerformance comparison (baseline → current):")
        print(f"  AUC:           {cd_report['baseline_metrics']['auc']:.3f} → {cd_report['current_metrics']['auc']:.3f} ({cd_report['auc_drop']:+.3f})")
        print(f"  Precision@10%: {cd_report['baseline_metrics']['precision_at_10']:.3f} → {cd_report['current_metrics']['precision_at_10']:.3f} ({cd_report['precision_at_10_drop']:+.3f})")
        print(f"  Base rate:     {cd_report['baseline_metrics']['base_rate']:.1%} → {cd_report['current_metrics']['base_rate']:.1%} ({cd_report['base_rate_shift']:+.1%})")
        print(f"\nInterpretation:")
        print(f"  AUC: {cd_report['interpretation']['auc_change']}")
        print(f"  Base rate: {cd_report['interpretation']['base_rate_change']}")
        print(f"  Recommendation: {cd_report['interpretation']['recommendation']}")
    
    # Print alerts
    print("\n" + "="*60)
    print("ALERTS")
    print("="*60)
    
    if report["alerts"]:
        for alert in report["alerts"]:
            severity_icon = "🔴" if alert["severity"] == "HIGH" else "🟡"
            print(f"\n{severity_icon} [{alert['severity']}] {alert['type']}")
            print(f"   {alert['message']}")
    else:
        print("\n✓ No alerts triggered")
    
    # Print recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    print(f"\n✓ Full report saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
