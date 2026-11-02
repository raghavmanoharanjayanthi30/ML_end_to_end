"""
Tests for model serving module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.serve import (
    MemberInput,
    PredictionOutput,
    ModelService
)


class TestMemberInput:
    """Tests for MemberInput schema."""
    
    def test_valid_input(self):
        """Valid input should create MemberInput object."""
        input_data = {
            "age": 45,
            "state": "OH",
            "is_rural": False,
            "plan_type": "commercial_group",
            "tenure_years": 5,
            "chronic_conditions": True,
            "claims_last_12mo": 8,
            "denied_claims_last_12mo": 2,
            "premium_increase_pct": 5.5,
            "service_calls_last_12mo": 2,
            "last_satisfaction_score": 4.0,
            "member_id": "M0000001"
        }
        
        member = MemberInput(**input_data)
        
        assert member.age == 45
        assert member.state == "OH"
        assert member.plan_type == "commercial_group"
    
    def test_optional_satisfaction_score(self):
        """Satisfaction score should be optional (can be None)."""
        input_data = {
            "age": 45,
            "state": "OH",
            "is_rural": False,
            "plan_type": "commercial_group",
            "tenure_years": 5,
            "chronic_conditions": True,
            "claims_last_12mo": 8,
            "denied_claims_last_12mo": 2,
            "premium_increase_pct": 5.5,
            "service_calls_last_12mo": 2,
            "last_satisfaction_score": None
        }
        
        member = MemberInput(**input_data)
        
        assert member.last_satisfaction_score is None
    
    def test_age_validation_min(self):
        """Age below 18 should fail validation."""
        input_data = {
            "age": 15,  # Too young
            "state": "OH",
            "is_rural": False,
            "plan_type": "commercial_group",
            "tenure_years": 5,
            "chronic_conditions": True,
            "claims_last_12mo": 8,
            "denied_claims_last_12mo": 2,
            "premium_increase_pct": 5.5,
            "service_calls_last_12mo": 2
        }
        
        with pytest.raises(Exception):  # Pydantic validation error
            MemberInput(**input_data)
    
    def test_age_validation_max(self):
        """Age above 100 should fail validation."""
        input_data = {
            "age": 150,  # Too old
            "state": "OH",
            "is_rural": False,
            "plan_type": "commercial_group",
            "tenure_years": 5,
            "chronic_conditions": True,
            "claims_last_12mo": 8,
            "denied_claims_last_12mo": 2,
            "premium_increase_pct": 5.5,
            "service_calls_last_12mo": 2
        }
        
        with pytest.raises(Exception):  # Pydantic validation error
            MemberInput(**input_data)


class TestModelService:
    """Tests for ModelService class."""
    
    def test_model_service_init(self):
        """ModelService should initialize with None model."""
        service = ModelService()
        
        assert service.model is None
        assert service.feature_pipeline is None
    
    def test_get_risk_tier_high(self):
        """Probability >= 0.5 should be high risk."""
        service = ModelService()
        
        assert service.get_risk_tier(0.5) == "high"
        assert service.get_risk_tier(0.7) == "high"
        assert service.get_risk_tier(0.99) == "high"
    
    def test_get_risk_tier_medium(self):
        """Probability 0.3-0.5 should be medium risk."""
        service = ModelService()
        
        assert service.get_risk_tier(0.3) == "medium"
        assert service.get_risk_tier(0.4) == "medium"
        assert service.get_risk_tier(0.49) == "medium"
    
    def test_get_risk_tier_low(self):
        """Probability < 0.3 should be low risk."""
        service = ModelService()
        
        assert service.get_risk_tier(0.0) == "low"
        assert service.get_risk_tier(0.1) == "low"
        assert service.get_risk_tier(0.29) == "low"


class TestRiskTierDistribution:
    """Tests for risk tier assignment."""
    
    def test_risk_tiers_cover_all_probabilities(self):
        """All probabilities should map to a risk tier."""
        service = ModelService()
        
        probabilities = [0.0, 0.1, 0.2, 0.29, 0.3, 0.4, 0.49, 0.5, 0.7, 0.9, 1.0]
        
        for prob in probabilities:
            tier = service.get_risk_tier(prob)
            assert tier in ["low", "medium", "high"]
    
    def test_risk_tier_boundaries(self):
        """Test exact boundary values."""
        service = ModelService()
        
        # Boundary at 0.3
        assert service.get_risk_tier(0.299) == "low"
        assert service.get_risk_tier(0.3) == "medium"
        
        # Boundary at 0.5
        assert service.get_risk_tier(0.499) == "medium"
        assert service.get_risk_tier(0.5) == "high"


class TestPredictionOutput:
    """Tests for PredictionOutput schema."""
    
    def test_prediction_output_creation(self):
        """Should create valid PredictionOutput."""
        output = PredictionOutput(
            member_id="M0000001",
            churn_probability=0.75,
            risk_tier="high",
            top_features=[{"feature": "age", "importance": 0.3, "value": 65}],
            prediction_timestamp="2024-01-01T00:00:00"
        )
        
        assert output.member_id == "M0000001"
        assert output.churn_probability == 0.75
        assert output.risk_tier == "high"
    
    def test_prediction_output_optional_member_id(self):
        """member_id should be optional."""
        output = PredictionOutput(
            member_id=None,
            churn_probability=0.5,
            risk_tier="high",
            top_features=[],
            prediction_timestamp="2024-01-01T00:00:00"
        )
        
        assert output.member_id is None
