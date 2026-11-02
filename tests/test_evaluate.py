"""
Tests for model evaluation module.
"""

import pytest
import pandas as pd
import numpy as np

from src.evaluate import (
    calculate_metrics,
    calculate_subgroup_metrics,
    calculate_calibration_error
)
from src.train import precision_at_k, recall_at_k
from src.features import TARGET


class TestCalculateMetrics:
    """Tests for metrics calculation."""
    
    def test_calculate_metrics_returns_dict(self):
        """Should return a dictionary of metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.2, 0.1, 0.2])
        
        metrics = calculate_metrics(y_true, y_proba)
        
        assert isinstance(metrics, dict)
    
    def test_calculate_metrics_has_required_keys(self):
        """Should include all required metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.2, 0.1, 0.2])
        
        metrics = calculate_metrics(y_true, y_proba)
        
        required_keys = ['auc', 'precision_at_10', 'recall_at_10', 'f1', 'n_samples', 'base_rate']
        for key in required_keys:
            assert key in metrics
    
    def test_auc_range(self):
        """AUC should be between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100)
        y_proba = np.random.random(100)
        
        metrics = calculate_metrics(y_true, y_proba)
        
        assert 0 <= metrics['auc'] <= 1
    
    def test_perfect_predictions_high_auc(self):
        """Perfect predictions should have high AUC."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9])
        
        metrics = calculate_metrics(y_true, y_proba)
        
        assert metrics['auc'] > 0.9
    
    def test_n_samples_correct(self):
        """n_samples should match input length."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        
        metrics = calculate_metrics(y_true, y_proba)
        
        assert metrics['n_samples'] == 5
    
    def test_base_rate_correct(self):
        """base_rate should be fraction of positives."""
        y_true = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0])  # 20% positive
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        metrics = calculate_metrics(y_true, y_proba)
        
        assert metrics['base_rate'] == 0.2


class TestSubgroupMetrics:
    """Tests for subgroup metrics calculation."""
    
    def test_subgroup_metrics_returns_dict(self, sample_valid_data):
        """Should return a dictionary of subgroup metrics."""
        np.random.seed(42)
        y_proba = np.random.random(len(sample_valid_data))
        
        subgroup_metrics = calculate_subgroup_metrics(sample_valid_data, y_proba, 'plan_type')
        
        assert isinstance(subgroup_metrics, dict)
    
    def test_subgroup_metrics_has_all_groups(self, sample_valid_data):
        """Should have metrics for each subgroup."""
        np.random.seed(42)
        y_proba = np.random.random(len(sample_valid_data))
        
        subgroup_metrics = calculate_subgroup_metrics(sample_valid_data, y_proba, 'plan_type')
        
        unique_plans = sample_valid_data['plan_type'].unique()
        for plan in unique_plans:
            assert plan in subgroup_metrics
    
    def test_subgroup_metrics_by_state(self, sample_valid_data):
        """Should work for state subgroups."""
        np.random.seed(42)
        y_proba = np.random.random(len(sample_valid_data))
        
        subgroup_metrics = calculate_subgroup_metrics(sample_valid_data, y_proba, 'state')
        
        unique_states = sample_valid_data['state'].unique()
        for state in unique_states:
            assert state in subgroup_metrics


class TestCalibrationError:
    """Tests for calibration error calculation."""
    
    def test_calibration_error_range(self):
        """Calibration error should be between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100)
        y_proba = np.random.random(100)
        
        ece = calculate_calibration_error(y_true, y_proba)
        
        assert 0 <= ece <= 1
    
    def test_perfect_calibration_low_error(self):
        """Perfectly calibrated predictions should have low error."""
        # Create perfectly calibrated predictions
        y_proba = np.array([0.1] * 10 + [0.5] * 10 + [0.9] * 10)
        y_true = np.array([0] * 9 + [1] * 1 +  # 10% positive at 0.1
                         [0] * 5 + [1] * 5 +   # 50% positive at 0.5
                         [0] * 1 + [1] * 9)    # 90% positive at 0.9
        
        ece = calculate_calibration_error(y_true, y_proba)
        
        assert ece < 0.1  # Should be low
    
    def test_poor_calibration_high_error(self):
        """Poorly calibrated predictions should have higher error."""
        # Predictions say 90% but actual is 10%
        y_proba = np.array([0.9] * 100)
        y_true = np.array([0] * 90 + [1] * 10)
        
        ece = calculate_calibration_error(y_true, y_proba)
        
        assert ece > 0.5  # Should be high
