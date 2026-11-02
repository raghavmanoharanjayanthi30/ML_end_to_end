"""
Tests for data ingestion and validation module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.ingest import (
    validate_data_simple,
    EXPECTED_COLUMNS,
    VALID_GENDERS,
    VALID_STATES,
    VALID_PLAN_TYPES
)


class TestValidDataPasses:
    """Tests that valid data passes validation."""
    
    def test_valid_data_passes(self, temp_csv_path):
        """Valid data should pass validation without errors."""
        df = validate_data_simple(temp_csv_path, verbose=False)
        assert df is not None
        assert len(df) > 0
    
    def test_returns_dataframe(self, temp_csv_path):
        """Should return a pandas DataFrame."""
        df = validate_data_simple(temp_csv_path, verbose=False)
        assert isinstance(df, pd.DataFrame)
    
    def test_all_columns_present(self, temp_csv_path):
        """Returned DataFrame should have all expected columns."""
        df = validate_data_simple(temp_csv_path, verbose=False)
        for col in EXPECTED_COLUMNS:
            assert col in df.columns


class TestInvalidDataFails:
    """Tests that invalid data fails validation."""
    
    def test_duplicate_member_id_fails(self, tmp_path, sample_invalid_data_duplicate_id):
        """Duplicate member_id should fail validation."""
        path = tmp_path / "bad_data.csv"
        sample_invalid_data_duplicate_id.to_csv(path, index=False)
        
        with pytest.raises(ValueError):
            validate_data_simple(str(path), verbose=False)
    
    def test_invalid_age_fails(self, tmp_path, sample_invalid_data_bad_age):
        """Age outside valid range should fail validation."""
        path = tmp_path / "bad_data.csv"
        sample_invalid_data_bad_age.to_csv(path, index=False)
        
        with pytest.raises(ValueError):
            validate_data_simple(str(path), verbose=False)
    
    def test_invalid_state_fails(self, tmp_path, sample_invalid_data_bad_state):
        """Invalid state code should fail validation."""
        path = tmp_path / "bad_data.csv"
        sample_invalid_data_bad_state.to_csv(path, index=False)
        
        with pytest.raises(ValueError):
            validate_data_simple(str(path), verbose=False)
    
    def test_invalid_churned_fails(self, tmp_path, sample_invalid_data_bad_churned):
        """Invalid churned value should fail validation."""
        path = tmp_path / "bad_data.csv"
        sample_invalid_data_bad_churned.to_csv(path, index=False)
        
        with pytest.raises(ValueError):
            validate_data_simple(str(path), verbose=False)
    
    def test_missing_file_fails(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validate_data_simple("nonexistent_file.csv", verbose=False)


class TestMemberIdFormat:
    """Tests for member_id format validation."""
    
    def test_valid_member_id_format(self, sample_valid_data):
        """Valid member_id format should pass."""
        # All IDs in sample_valid_data are valid format M#######
        for mid in sample_valid_data['member_id']:
            assert mid.startswith('M')
            assert len(mid) == 8
    
    def test_invalid_member_id_format_fails(self, tmp_path, sample_valid_data):
        """Invalid member_id format should fail."""
        df = sample_valid_data.copy()
        df.loc[0, 'member_id'] = 'BADID123'  # Wrong format
        
        path = tmp_path / "bad_data.csv"
        df.to_csv(path, index=False)
        
        with pytest.raises(ValueError):
            validate_data_simple(str(path), verbose=False)


class TestBusinessLogic:
    """Tests for business logic validation."""
    
    def test_denied_claims_not_exceeding_total(self, tmp_path, sample_valid_data):
        """Denied claims should not exceed total claims."""
        df = sample_valid_data.copy()
        df.loc[0, 'claims_last_12mo'] = 5
        df.loc[0, 'denied_claims_last_12mo'] = 10  # More denied than total
        
        path = tmp_path / "bad_data.csv"
        df.to_csv(path, index=False)
        
        with pytest.raises(ValueError):
            validate_data_simple(str(path), verbose=False)
