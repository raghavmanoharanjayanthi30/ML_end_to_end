"""
Model serving module with FastAPI.

This module provides a REST API for churn predictions:
- POST /predict - Predict churn for a single member or batch
- POST /predict/csv - Upload CSV and get predictions
- GET /health - Health check
- GET /model/info - Model metadata

Usage:
    # Start server
    python -m src.serve
    
    # Or with uvicorn directly
    uvicorn src.serve:app --reload --port 8000

    # Test with curl
    curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"age": 45, "state": "OH", ...}'
"""

import os
import sys
import pickle
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import io

from src.features import FEATURE_COLUMNS, ALLOWED_FEATURES, get_feature_names


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class MemberInput(BaseModel):
    """Input schema for a single member prediction."""
    age: int = Field(..., ge=18, le=100, description="Member age")
    state: str = Field(..., description="State code (OH, IN, KY, WV)")
    is_rural: bool = Field(..., description="Rural residence flag")
    plan_type: str = Field(..., description="Plan type")
    tenure_years: int = Field(..., ge=0, description="Years enrolled")
    chronic_conditions: bool = Field(..., description="Has chronic conditions")
    claims_last_12mo: int = Field(..., ge=0, description="Claims in last 12 months")
    denied_claims_last_12mo: int = Field(..., ge=0, description="Denied claims")
    premium_increase_pct: float = Field(..., description="Premium increase %")
    service_calls_last_12mo: int = Field(..., ge=0, description="Service calls")
    last_satisfaction_score: Optional[float] = Field(None, ge=1, le=5, description="Satisfaction score (1-5)")
    member_id: Optional[str] = Field(None, description="Member ID (optional)")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionOutput(BaseModel):
    """Output schema for a single prediction."""
    member_id: Optional[str]
    churn_probability: float
    risk_tier: str
    top_features: List[dict]
    prediction_timestamp: str


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionOutput]
    total_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class ModelInfo(BaseModel):
    """Model metadata."""
    model_type: str
    run_id: str
    features: List[str]
    loaded_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


# =============================================================================
# MODEL LOADING
# =============================================================================

class ModelService:
    """Service class for model loading and prediction."""
    
    def __init__(self):
        self.model = None
        self.feature_pipeline = None
        self.run_id = None
        self.loaded_at = None
        self.feature_names = None
    
    def load_model(self, run_id: str = "latest"):
        """Load model and feature pipeline from MLflow."""
        mlflow.set_tracking_uri("file:./mlruns")
        
        if run_id == "latest":
            if os.path.exists("artifacts/best_run_id.txt"):
                with open("artifacts/best_run_id.txt", 'r') as f:
                    run_id = f.read().strip()
            else:
                raise ValueError("No best_run_id.txt found. Run training first.")
        
        # Load model
        model_uri = f"runs:/{run_id}/model"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.run_id = run_id
        
        # Load feature pipeline
        import glob
        pipeline_files = glob.glob(f"mlruns/*/{run_id}/artifacts/feature_pipeline.pkl")
        
        if pipeline_files:
            with open(pipeline_files[0], 'rb') as f:
                self.feature_pipeline = pickle.load(f)
        else:
            # Create and fit new pipeline
            from src.features import create_feature_pipeline
            self.feature_pipeline = create_feature_pipeline()
            train_df = pd.read_csv("data/raw/train.csv")
            X_train = train_df[FEATURE_COLUMNS].copy()
            self.feature_pipeline.fit(X_train)
        
        self.feature_names = get_feature_names(self.feature_pipeline)
        self.loaded_at = datetime.now().isoformat()
        
        return self
    
    def predict(self, df: pd.DataFrame) -> tuple:
        """
        Make predictions for a DataFrame.
        
        Returns:
            Tuple of (probabilities, feature_contributions)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Ensure all required columns exist
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = None
        
        X = df[FEATURE_COLUMNS].copy()
        X_transformed = self.feature_pipeline.transform(X)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_transformed)[:, 1]
        
        # Get feature contributions (for tree-based models)
        feature_contributions = self._get_feature_contributions(X_transformed)
        
        return probabilities, feature_contributions
    
    def _get_feature_contributions(self, X_transformed: np.ndarray) -> List[List[dict]]:
        """Get top contributing features for each prediction."""
        contributions = []
        
        # Check if model has feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            for i in range(len(X_transformed)):
                # Weight importances by feature values
                weighted = importances * np.abs(X_transformed[i])
                top_indices = np.argsort(weighted)[-3:][::-1]
                
                top_features = []
                for idx in top_indices:
                    if idx < len(self.feature_names):
                        top_features.append({
                            'feature': self.feature_names[idx],
                            'importance': float(importances[idx]),
                            'value': float(X_transformed[i, idx])
                        })
                contributions.append(top_features)
        else:
            # For models without feature_importances_, return empty
            contributions = [[] for _ in range(len(X_transformed))]
        
        return contributions
    
    def get_risk_tier(self, probability: float) -> str:
        """Classify probability into risk tier."""
        if probability >= 0.5:
            return "high"
        elif probability >= 0.3:
            return "medium"
        else:
            return "low"


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting member churn at Meridian Health",
    version="1.0.0"
)

# Global model service
model_service = ModelService()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        model_service.load_model("latest")
        print(f"✓ Model loaded: {type(model_service.model).__name__}")
        print(f"  Run ID: {model_service.run_id}")
    except Exception as e:
        print(f"⚠ Warning: Could not load model on startup: {e}")
        print("  Model will need to be loaded manually or via /model/load endpoint")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model metadata."""
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type=type(model_service.model).__name__,
        run_id=model_service.run_id,
        features=model_service.feature_names,
        loaded_at=model_service.loaded_at
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_single(member: MemberInput):
    """Predict churn for a single member."""
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to DataFrame
    df = pd.DataFrame([member.model_dump()])
    
    # Make prediction
    probabilities, contributions = model_service.predict(df)
    
    return PredictionOutput(
        member_id=member.member_id,
        churn_probability=float(probabilities[0]),
        risk_tier=model_service.get_risk_tier(probabilities[0]),
        top_features=contributions[0] if contributions else [],
        prediction_timestamp=datetime.now().isoformat()
    )


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(members: List[MemberInput]):
    """Predict churn for multiple members."""
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to DataFrame
    df = pd.DataFrame([m.model_dump() for m in members])
    
    # Make predictions
    probabilities, contributions = model_service.predict(df)
    
    # Build response
    predictions = []
    for i, (prob, contrib) in enumerate(zip(probabilities, contributions)):
        predictions.append(PredictionOutput(
            member_id=members[i].member_id,
            churn_probability=float(prob),
            risk_tier=model_service.get_risk_tier(prob),
            top_features=contrib if contrib else [],
            prediction_timestamp=datetime.now().isoformat()
        ))
    
    # Count risk tiers
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    for p in predictions:
        risk_counts[p.risk_tier] += 1
    
    return BatchPredictionOutput(
        predictions=predictions,
        total_count=len(predictions),
        high_risk_count=risk_counts["high"],
        medium_risk_count=risk_counts["medium"],
        low_risk_count=risk_counts["low"]
    )


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and get predictions.
    
    Returns a CSV with predictions added.
    """
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")
    
    # Make predictions
    try:
        probabilities, contributions = model_service.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making predictions: {e}")
    
    # Add predictions to DataFrame
    df['churn_probability'] = probabilities
    df['risk_tier'] = [model_service.get_risk_tier(p) for p in probabilities]
    
    # Add top features as JSON string
    df['top_features'] = [json.dumps(c) for c in contributions]
    
    # Return as CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )


# =============================================================================
# CLI INTERFACE (for batch processing)
# =============================================================================

def predict_from_cli(input_path: str, output_path: str):
    """
    CLI interface for batch predictions.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to save predictions
    """
    print("="*60)
    print("BATCH PREDICTION")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model_service.load_model("latest")
    print(f"✓ Model loaded: {type(model_service.model).__name__}")
    
    # Load data
    print(f"\nLoading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df):,} rows")
    
    # Make predictions
    print("\nMaking predictions...")
    probabilities, contributions = model_service.predict(df)
    
    # Build output DataFrame
    output_df = pd.DataFrame({
        'member_id': df['member_id'] if 'member_id' in df.columns else range(len(df)),
        'churn_probability': probabilities,
        'risk_tier': [model_service.get_risk_tier(p) for p in probabilities],
        'top_feature_1': [c[0]['feature'] if c else '' for c in contributions],
        'top_feature_2': [c[1]['feature'] if len(c) > 1 else '' for c in contributions],
        'top_feature_3': [c[2]['feature'] if len(c) > 2 else '' for c in contributions]
    })
    
    # Sort by probability (highest risk first)
    output_df = output_df.sort_values('churn_probability', ascending=False)
    
    # Save
    output_df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to {output_path}")
    
    # Summary
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    print(f"Total members: {len(output_df):,}")
    print(f"High risk: {(output_df['risk_tier'] == 'high').sum():,}")
    print(f"Medium risk: {(output_df['risk_tier'] == 'medium').sum():,}")
    print(f"Low risk: {(output_df['risk_tier'] == 'low').sum():,}")
    print(f"\nTop 10 highest risk members:")
    print(output_df.head(10).to_string(index=False))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Churn prediction serving (API or batch)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['api', 'batch'],
        default='api',
        help='Serving mode: api (FastAPI server) or batch (CLI)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input CSV path (batch mode only)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output CSV path (batch mode only)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='API host (api mode only)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API port (api mode only)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'batch':
        if not args.input:
            print("Error: --input required for batch mode")
            return 1
        predict_from_cli(args.input, args.output)
    else:
        print(f"Starting API server on {args.host}:{args.port}")
        print(f"API docs available at http://{args.host}:{args.port}/docs")
        uvicorn.run(app, host=args.host, port=args.port)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
