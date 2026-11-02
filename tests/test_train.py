"""
Tests for model training module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.train import (
    precision_at_k,
    recall_at_k,
    get_model_configs
)
from src.features import create_feature_pipeline, FEATURE_COLUMNS, TARGET


class TestPrecisionAtK:
    """Tests for precision@k metric."""
    
    def test_precision_at_k_perfect(self):
        """Perfect predictions should give precision of 1.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        # Top 30% (3 samples) should all be positive
        precision = precision_at_k(y_true, y_proba, k=0.3)
        assert precision == 1.0
    
    def test_precision_at_k_worst(self):
        """Worst predictions should give precision of 0.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_proba = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        
        # Top 30% (3 samples) should all be negative
        precision = precision_at_k(y_true, y_proba, k=0.3)
        assert precision == 0.0
    
    def test_precision_at_k_range(self):
        """Precision should be between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100)
        y_proba = np.random.random(100)
        
        precision = precision_at_k(y_true, y_proba, k=0.1)
        assert 0 <= precision <= 1
    
    def test_precision_at_10_percent(self):
        """Test precision at 10%."""
        y_true = np.array([1] * 10 + [0] * 90)
        y_proba = np.array([0.9] * 10 + [0.1] * 90)
        
        precision = precision_at_k(y_true, y_proba, k=0.1)
        assert precision == 1.0


class TestRecallAtK:
    """Tests for recall@k metric."""
    
    def test_recall_at_k_perfect(self):
        """Perfect predictions should capture all positives in top k."""
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        # Top 20% (2 samples) should capture both positives
        recall = recall_at_k(y_true, y_proba, k=0.2)
        assert recall == 1.0
    
    def test_recall_at_k_range(self):
        """Recall should be between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100)
        y_proba = np.random.random(100)
        
        recall = recall_at_k(y_true, y_proba, k=0.1)
        assert 0 <= recall <= 1
    
    def test_recall_at_k_no_positives(self):
        """Recall should be 0 when no positives exist."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        
        recall = recall_at_k(y_true, y_proba, k=0.2)
        assert recall == 0.0


class TestModelConfigs:
    """Tests for model configurations."""
    
    def test_get_model_configs_quick(self):
        """Quick mode should return configs."""
        configs = get_model_configs(quick_mode=True)
        
        assert 'logistic_regression' in configs
        assert 'random_forest' in configs
        assert 'gradient_boosting' in configs
    
    def test_get_model_configs_full(self):
        """Full mode should return configs with more parameters."""
        configs = get_model_configs(quick_mode=False)
        
        assert 'logistic_regression' in configs
        # Full mode should have more parameter options
        assert len(configs['logistic_regression']['param_grid']['C']) > 1
    
    def test_configs_have_model_and_params(self):
        """Each config should have model and param_grid."""
        configs = get_model_configs(quick_mode=True)
        
        for name, config in configs.items():
            assert 'model' in config
            assert 'param_grid' in config


class TestModelTraining:
    """Tests for model training (using mocks to avoid full training)."""
    
    def test_model_can_fit(self, sample_valid_data):
        """Model should be able to fit on sample data."""
        from sklearn.ensemble import GradientBoostingClassifier
        
        pipeline = create_feature_pipeline()
        X = sample_valid_data[FEATURE_COLUMNS].copy()
        y = sample_valid_data[TARGET].values
        
        X_transformed = pipeline.fit_transform(X)
        
        model = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X_transformed, y)
        
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_model_predictions_valid_range(self, sample_valid_data):
        """Model predictions should be in [0, 1]."""
        from sklearn.ensemble import GradientBoostingClassifier
        
        pipeline = create_feature_pipeline()
        X = sample_valid_data[FEATURE_COLUMNS].copy()
        y = sample_valid_data[TARGET].values
        
        X_transformed = pipeline.fit_transform(X)
        
        model = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X_transformed, y)
        
        proba = model.predict_proba(X_transformed)[:, 1]
        
        assert proba.min() >= 0
        assert proba.max() <= 1
    
    def test_model_predictions_shape(self, sample_valid_data):
        """Model should predict for all samples."""
        from sklearn.ensemble import GradientBoostingClassifier
        
        pipeline = create_feature_pipeline()
        X = sample_valid_data[FEATURE_COLUMNS].copy()
        y = sample_valid_data[TARGET].values
        
        X_transformed = pipeline.fit_transform(X)
        
        model = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X_transformed, y)
        
        predictions = model.predict(X_transformed)
        
        assert len(predictions) == len(y)
