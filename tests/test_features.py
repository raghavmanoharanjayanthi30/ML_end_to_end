"""
Tests for feature engineering pipeline.
"""

import pytest
import pandas as pd
import numpy as np

from src.features import (
    create_feature_pipeline,
    prepare_features,
    validate_features,
    get_feature_names,
    FEATURE_COLUMNS,
    ALLOWED_FEATURES,
    BLOCKED_FEATURES,
    LEAKAGE_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
    TARGET
)


class TestFeatureDefinitions:
    """Tests for feature definitions."""
    
    def test_allowed_features_defined(self):
        """Allowed features should be defined."""
        assert len(ALLOWED_FEATURES) > 0
    
    def test_blocked_features_defined(self):
        """Blocked features should be defined."""
        assert len(BLOCKED_FEATURES) > 0
        assert 'gender' in BLOCKED_FEATURES
        assert 'race' in BLOCKED_FEATURES
    
    def test_leakage_features_defined(self):
        """Leakage features should be defined."""
        assert len(LEAKAGE_FEATURES) > 0
        assert 'days_to_renewal_decision' in LEAKAGE_FEATURES
        assert 'cancellation_form_started' in LEAKAGE_FEATURES
    
    def test_no_overlap_allowed_blocked(self):
        """Allowed and blocked features should not overlap."""
        overlap = set(ALLOWED_FEATURES) & set(BLOCKED_FEATURES)
        assert len(overlap) == 0, f"Overlap found: {overlap}"
    
    def test_no_overlap_allowed_leakage(self):
        """Allowed and leakage features should not overlap."""
        overlap = set(ALLOWED_FEATURES) & set(LEAKAGE_FEATURES)
        assert len(overlap) == 0, f"Overlap found: {overlap}"


class TestPipelineCreation:
    """Tests for pipeline creation."""
    
    def test_create_pipeline(self):
        """Pipeline should be created without errors."""
        pipeline = create_feature_pipeline()
        assert pipeline is not None
    
    def test_pipeline_has_steps(self):
        """Pipeline should have processing steps."""
        pipeline = create_feature_pipeline()
        assert len(pipeline.steps) > 0


class TestPipelineTransform:
    """Tests for pipeline transformation."""
    
    def test_pipeline_fit_transform(self, sample_features_only):
        """Pipeline should fit and transform data."""
        pipeline = create_feature_pipeline()
        X_transformed = pipeline.fit_transform(sample_features_only)
        
        assert X_transformed is not None
        assert len(X_transformed) == len(sample_features_only)
    
    def test_output_is_numeric(self, sample_features_only):
        """Transformed output should be numeric."""
        pipeline = create_feature_pipeline()
        X_transformed = pipeline.fit_transform(sample_features_only)
        
        assert np.issubdtype(X_transformed.dtype, np.number)
    
    def test_output_shape(self, sample_features_only):
        """Output should have expected number of features."""
        pipeline = create_feature_pipeline()
        X_transformed = pipeline.fit_transform(sample_features_only)
        
        # 7 numeric + 3 state dummies + 2 plan_type dummies + 2 boolean = 14
        expected_features = 14
        assert X_transformed.shape[1] == expected_features
    
    def test_no_nan_in_output(self, sample_features_only):
        """Transformed output should have no NaN values (imputation should work)."""
        pipeline = create_feature_pipeline()
        X_transformed = pipeline.fit_transform(sample_features_only)
        
        assert not np.isnan(X_transformed).any()


class TestMissingValueHandling:
    """Tests for missing value handling."""
    
    def test_handles_missing_satisfaction_score(self, sample_valid_data):
        """Pipeline should handle missing satisfaction scores."""
        # Ensure some missing values
        df = sample_valid_data.copy()
        df.loc[0:10, 'last_satisfaction_score'] = np.nan
        
        pipeline = create_feature_pipeline()
        X = df[FEATURE_COLUMNS].copy()
        X_transformed = pipeline.fit_transform(X)
        
        # Should not raise and should have no NaN
        assert not np.isnan(X_transformed).any()
    
    def test_all_missing_satisfaction_handled(self, sample_valid_data):
        """Pipeline should handle all missing satisfaction scores."""
        df = sample_valid_data.copy()
        df['last_satisfaction_score'] = np.nan  # All missing
        
        pipeline = create_feature_pipeline()
        X = df[FEATURE_COLUMNS].copy()
        X_transformed = pipeline.fit_transform(X)
        
        assert not np.isnan(X_transformed).any()


class TestFeatureValidation:
    """Tests for feature validation."""
    
    def test_validate_features_all_present(self, sample_valid_data):
        """Validation should pass when all features present."""
        missing = validate_features(sample_valid_data)
        assert len(missing) == 0
    
    def test_validate_features_missing_column(self, sample_valid_data):
        """Validation should detect missing columns."""
        df = sample_valid_data.drop(columns=['age'])
        missing = validate_features(df)
        assert 'age' in missing


class TestFeatureNames:
    """Tests for feature name retrieval."""
    
    def test_get_feature_names(self, sample_features_only):
        """Should return feature names."""
        pipeline = create_feature_pipeline()
        pipeline.fit(sample_features_only)
        
        names = get_feature_names(pipeline)
        assert len(names) > 0
        assert isinstance(names, list)
    
    def test_feature_names_match_output_shape(self, sample_features_only):
        """Feature names should match output shape."""
        pipeline = create_feature_pipeline()
        X_transformed = pipeline.fit_transform(sample_features_only)
        
        names = get_feature_names(pipeline)
        assert len(names) == X_transformed.shape[1]


class TestBlockedFeaturesExcluded:
    """Tests that blocked features are not used."""
    
    def test_blocked_features_not_in_feature_columns(self):
        """Blocked features should not be in FEATURE_COLUMNS."""
        for blocked in BLOCKED_FEATURES:
            assert blocked not in FEATURE_COLUMNS
    
    def test_leakage_features_not_in_feature_columns(self):
        """Leakage features should not be in FEATURE_COLUMNS."""
        for leakage in LEAKAGE_FEATURES:
            assert leakage not in FEATURE_COLUMNS
