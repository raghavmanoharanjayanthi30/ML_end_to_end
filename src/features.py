"""
Feature engineering pipeline for churn prediction.

This module creates a reproducible sklearn Pipeline that handles:
- Missing value imputation
- Categorical encoding
- Feature scaling
- Derived feature creation

The same pipeline is used for both training and inference to ensure consistency.

Usage:
    from src.features import create_feature_pipeline, FEATURE_COLUMNS, TARGET
    
    pipeline = create_feature_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Features we ARE allowed to use (per governance policy)
ALLOWED_FEATURES = [
    'age', 'state', 'is_rural', 'plan_type', 'tenure_years',
    'chronic_conditions', 'claims_last_12mo', 'denied_claims_last_12mo',
    'premium_increase_pct', 'service_calls_last_12mo', 'last_satisfaction_score'
]

# Features we are NOT allowed to use (blocked by governance)
BLOCKED_FEATURES = ['gender', 'race', 'dual_eligible', 'mental_health_diagnosis']

# Features that would cause data leakage
LEAKAGE_FEATURES = ['days_to_renewal_decision', 'cancellation_form_started']

# Target variable
TARGET = 'churned'

# ID column (not a feature)
ID_COLUMN = 'member_id'

# Columns for time-based splitting (not features)
DATE_COLUMN = 'renewal_decision_date'

# Numeric features (continuous)
NUMERIC_FEATURES = [
    'age', 'tenure_years', 'claims_last_12mo', 'denied_claims_last_12mo',
    'premium_increase_pct', 'service_calls_last_12mo', 'last_satisfaction_score'
]

# Categorical features
CATEGORICAL_FEATURES = ['state', 'plan_type']

# Boolean features (will be converted to int)
BOOLEAN_FEATURES = ['is_rural', 'chronic_conditions']

# All feature columns used in the model
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES


# =============================================================================
# CUSTOM TRANSFORMERS
# =============================================================================

class DerivedFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates derived features from raw features.
    
    Derived features:
    - denied_claims_ratio: denied_claims / (claims + 1)
    - satisfaction_missing: binary indicator for missing satisfaction score
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Denied claims ratio (add 1 to avoid division by zero)
        if 'claims_last_12mo' in X.columns and 'denied_claims_last_12mo' in X.columns:
            X['denied_claims_ratio'] = X['denied_claims_last_12mo'] / (X['claims_last_12mo'] + 1)
        
        # Satisfaction missing indicator
        if 'last_satisfaction_score' in X.columns:
            X['satisfaction_missing'] = X['last_satisfaction_score'].isnull().astype(int)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        return list(input_features) + ['denied_claims_ratio', 'satisfaction_missing']


class BooleanConverter(BaseEstimator, TransformerMixin):
    """Converts boolean columns to integers (0/1)."""
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].astype(int)
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects only the specified columns from a DataFrame."""
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Handle case where some columns might not exist
        available_cols = [c for c in self.columns if c in X.columns]
        return X[available_cols]


class DataFrameWrapper(BaseEstimator, TransformerMixin):
    """
    Wraps the output of a ColumnTransformer back into a DataFrame.
    Useful for debugging and interpretability.
    """
    
    def __init__(self, transformer, feature_names=None):
        self.transformer = transformer
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        if self.feature_names is None:
            try:
                self.feature_names = self.transformer.get_feature_names_out()
            except AttributeError:
                self.feature_names = None
        return self
    
    def transform(self, X):
        result = self.transformer.transform(X)
        if self.feature_names is not None:
            return pd.DataFrame(result, columns=self.feature_names, index=X.index)
        return result


# =============================================================================
# PIPELINE CREATION
# =============================================================================

def create_feature_pipeline(include_derived=True):
    """
    Create the feature engineering pipeline.
    
    Args:
        include_derived: Whether to include derived features (default True)
    
    Returns:
        sklearn Pipeline object
    """
    
    # Define transformers for different column types
    
    # Numeric features: impute missing with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features: one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Boolean features: just convert to int (no scaling needed)
    boolean_transformer = Pipeline(steps=[
        ('passthrough', 'passthrough')
    ])
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('bool', boolean_transformer, BOOLEAN_FEATURES)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Build the full pipeline
    if include_derived:
        pipeline = Pipeline(steps=[
            ('derived', DerivedFeatureCreator()),
            ('bool_convert', BooleanConverter(BOOLEAN_FEATURES)),
            ('preprocessor', preprocessor)
        ])
    else:
        pipeline = Pipeline(steps=[
            ('bool_convert', BooleanConverter(BOOLEAN_FEATURES)),
            ('preprocessor', preprocessor)
        ])
    
    return pipeline


def get_feature_names(pipeline, include_derived=True):
    """
    Get the feature names after transformation.
    
    Args:
        pipeline: Fitted sklearn pipeline
        include_derived: Whether derived features were included
    
    Returns:
        List of feature names
    """
    # Manually construct feature names (more reliable than get_feature_names_out)
    feature_names = []
    
    # Numeric features (scaled) - these keep their names
    feature_names.extend(NUMERIC_FEATURES)
    
    # Categorical features (one-hot encoded, first category dropped)
    # state: IN, KY, WV (OH dropped as first alphabetically)
    feature_names.extend(['state_IN', 'state_KY', 'state_WV'])
    # plan_type: individual_marketplace, medicare_advantage (commercial_group dropped)
    feature_names.extend(['plan_type_individual_marketplace', 'plan_type_medicare_advantage'])
    
    # Boolean features (converted to int)
    feature_names.extend(BOOLEAN_FEATURES)
    
    return feature_names


def prepare_features(df: pd.DataFrame, pipeline=None, fit=False):
    """
    Prepare features from a raw DataFrame.
    
    Args:
        df: Raw DataFrame with all columns
        pipeline: Optional pre-fitted pipeline (if None, creates new one)
        fit: Whether to fit the pipeline (True for training, False for inference)
    
    Returns:
        Tuple of (X_transformed, y, pipeline)
    """
    # Extract target if present
    y = None
    if TARGET in df.columns:
        y = df[TARGET].values
    
    # Select only allowed features
    X = df[FEATURE_COLUMNS].copy()
    
    # Create pipeline if not provided
    if pipeline is None:
        pipeline = create_feature_pipeline()
    
    # Fit and/or transform
    if fit:
        X_transformed = pipeline.fit_transform(X)
    else:
        X_transformed = pipeline.transform(X)
    
    return X_transformed, y, pipeline


def validate_features(df: pd.DataFrame) -> list:
    """
    Validate that a DataFrame has the required features.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        List of missing feature columns (empty if all present)
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    return missing


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test feature pipeline')
    parser.add_argument('--data', type=str, default='data/raw/train.csv',
                        help='Path to data file')
    args = parser.parse_args()
    
    print("Loading data...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} rows")
    
    print("\n" + "="*60)
    print("FEATURE CONFIGURATION")
    print("="*60)
    print(f"\nAllowed features ({len(ALLOWED_FEATURES)}):")
    for f in ALLOWED_FEATURES:
        print(f"  ✓ {f}")
    
    print(f"\nBlocked features ({len(BLOCKED_FEATURES)}):")
    for f in BLOCKED_FEATURES:
        print(f"  ✗ {f}")
    
    print(f"\nLeakage features ({len(LEAKAGE_FEATURES)}):")
    for f in LEAKAGE_FEATURES:
        print(f"  ⚠ {f}")
    
    print("\n" + "="*60)
    print("TESTING PIPELINE")
    print("="*60)
    
    # Validate features
    missing = validate_features(df)
    if missing:
        print(f"\n✗ Missing features: {missing}")
    else:
        print("\n✓ All required features present")
    
    # Create and fit pipeline
    print("\nCreating feature pipeline...")
    pipeline = create_feature_pipeline()
    
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].values
    
    print("Fitting pipeline...")
    X_transformed = pipeline.fit_transform(X)
    
    print(f"\n✓ Pipeline fitted successfully!")
    print(f"  Input shape:  {X.shape}")
    print(f"  Output shape: {X_transformed.shape}")
    
    # Show feature names
    print("\n" + "="*60)
    print("TRANSFORMED FEATURES")
    print("="*60)
    feature_names = get_feature_names(pipeline)
    for i, name in enumerate(feature_names):
        print(f"  {i+1:2d}. {name}")
    
    print(f"\n✓ Feature pipeline test complete!")
