"""
Shared test fixtures and configuration.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_valid_data():
    """Create a small valid dataset for testing."""
    np.random.seed(42)
    n = 150  # More than minimum required (100)
    
    # Generate claims first, then denied claims that don't exceed total
    claims = np.random.randint(5, 20, n)
    denied = np.array([np.random.randint(0, min(c, 5)) for c in claims])
    
    return pd.DataFrame({
        'member_id': [f'M{str(i).zfill(7)}' for i in range(1, n + 1)],
        'age': np.random.randint(19, 90, n),
        'gender': np.random.choice(['M', 'F', 'X'], n),
        'race': np.random.choice(['white', 'black', 'hispanic', 'asian', 'other'], n),
        'state': np.random.choice(['OH', 'IN', 'KY', 'WV'], n),
        'is_rural': np.random.choice([True, False], n),
        'plan_type': np.random.choice(['commercial_group', 'individual_marketplace', 'medicare_advantage'], n),
        'dual_eligible': np.random.choice([True, False], n),
        'mental_health_diagnosis': np.random.choice([True, False], n),
        'tenure_years': np.random.randint(0, 20, n),
        'chronic_conditions': np.random.choice([True, False], n),
        'claims_last_12mo': claims,
        'denied_claims_last_12mo': denied,
        'premium_increase_pct': np.random.uniform(-5, 20, n),
        'service_calls_last_12mo': np.random.randint(0, 10, n),
        'last_satisfaction_score': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], n),
        'renewal_decision_date': pd.date_range('2024-01-01', periods=n, freq='D').astype(str),
        'days_to_renewal_decision': np.random.randint(-30, 60, n),
        'cancellation_form_started': np.random.choice([0, 1], n),
        'churned': np.random.choice([0, 1], n, p=[0.75, 0.25])
    })


@pytest.fixture
def sample_invalid_data_duplicate_id(sample_valid_data):
    """Data with duplicate member_id."""
    df = sample_valid_data.copy()
    df.loc[1, 'member_id'] = df.loc[0, 'member_id']  # Create duplicate
    return df


@pytest.fixture
def sample_invalid_data_bad_age(sample_valid_data):
    """Data with out-of-range age."""
    df = sample_valid_data.copy()
    df.loc[0, 'age'] = 150  # Invalid age
    return df


@pytest.fixture
def sample_invalid_data_bad_state(sample_valid_data):
    """Data with invalid state."""
    df = sample_valid_data.copy()
    df.loc[0, 'state'] = 'XX'  # Invalid state
    return df


@pytest.fixture
def sample_invalid_data_bad_churned(sample_valid_data):
    """Data with invalid churned value."""
    df = sample_valid_data.copy()
    df.loc[0, 'churned'] = 2  # Invalid (should be 0 or 1)
    return df


@pytest.fixture
def sample_features_only(sample_valid_data):
    """Just the feature columns needed for prediction."""
    from src.features import FEATURE_COLUMNS
    return sample_valid_data[FEATURE_COLUMNS].copy()


@pytest.fixture
def temp_csv_path(tmp_path, sample_valid_data):
    """Save sample data to a temporary CSV file."""
    path = tmp_path / "test_data.csv"
    sample_valid_data.to_csv(path, index=False)
    return str(path)
