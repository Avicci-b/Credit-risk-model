"""
FastAPI Application for Credit Risk Model Deployment
Task 6: Model Deployment API
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# FastAPI imports
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
import uvicorn

# Import Pydantic models
from src.api.pydantic_models import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Probability Model API",
    description="API for predicting credit risk using alternative data (e-commerce transactions)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing artifacts
MODEL = None
LABEL_ENCODERS = None
SCALER = None
FEATURE_NAMES = None
MODEL_VERSION = "1.0.0"
START_TIME = datetime.now()


class ModelLoader:
    """
    Load and manage model artifacts
    """
    
    @staticmethod
    def load_model_artifacts():
        """Load model and preprocessing artifacts"""
        global MODEL, LABEL_ENCODERS, SCALER, FEATURE_NAMES
        
        try:
            # Load from MLflow registry (if available) or local file
            model_path = os.getenv("MODEL_PATH", "models/best_model.joblib")
            encoders_path = os.getenv("ENCODERS_PATH", "models/label_encoders.joblib")
            scaler_path = os.getenv("SCALER_PATH", "models/scaler.joblib")
            features_path = os.getenv("FEATURES_PATH", "models/feature_names.json")
            
            logger.info(f"Loading model from: {model_path}")
            
            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load artifacts
            MODEL = joblib.load(model_path)
            logger.info(f"✅ Model loaded: {type(MODEL).__name__}")
            
            if os.path.exists(encoders_path):
                LABEL_ENCODERS = joblib.load(encoders_path)
                logger.info(f"✅ Label encoders loaded: {len(LABEL_ENCODERS)} encoders")
            
            if os.path.exists(scaler_path):
                SCALER = joblib.load(scaler_path)
                logger.info("✅ Scaler loaded")
            
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    FEATURE_NAMES = json.load(f)
                logger.info(f"✅ Feature names loaded: {len(FEATURE_NAMES)} features")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model artifacts: {str(e)}")
            raise
    
    @staticmethod
    def get_model():
        """Get the loaded model"""
        if MODEL is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please check service health."
            )
        return MODEL
    
    @staticmethod
    def get_preprocessors():
        """Get preprocessing artifacts"""
        return LABEL_ENCODERS, SCALER, FEATURE_NAMES


class FeatureProcessor:
    """
    Process features for prediction
    """
    
    @staticmethod
    def prepare_features(request: PredictionRequest) -> pd.DataFrame:
        """
        Prepare features from request for model prediction
        """
        # Convert request to dictionary
        features = request.dict()
        
        # Remove CustomerId if present (not used in prediction)
        if 'CustomerId' in features:
            customer_id = features.pop('CustomerId')
        else:
            customer_id = None
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Apply label encoding if encoders available
        if LABEL_ENCODERS:
            for col, encoder in LABEL_ENCODERS.items():
                if col in df.columns:
                    # Handle unseen labels
                    try:
                        df[col] = encoder.transform(df[col].astype(str))
                    except ValueError:
                        # If unseen label, use most common class
                        df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Ensure correct feature order
        if FEATURE_NAMES:
            # Add missing features with default values
            for feature in FEATURE_NAMES:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # Reorder columns
            df = df[FEATURE_NAMES]
        
        # Apply scaling if scaler available
        if SCALER and hasattr(MODEL, 'predict_proba'):  # Only scale for models that need it
            try:
                df_scaled = SCALER.transform(df)
                df = pd.DataFrame(df_scaled, columns=df.columns)
            except Exception as e:
                logger.warning(f"Scaling failed: {str(e)}")
        
        logger.debug(f"Prepared features shape: {df.shape}")
        return df, customer_id
    
    @staticmethod
    def determine_risk_category(probability: float) -> str:
        """
        Convert probability to risk category
        """
        if probability < 0.3:
            return "LOW"
        elif probability < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"


# Startup event - load model
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Credit Risk Model API...")
    try:
        ModelLoader.load_model_artifacts()
        logger.info("✅ API startup completed successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint
    """
    uptime = (datetime.now() - START_TIME).total_seconds()
    
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_version=MODEL_VERSION,
        uptime=uptime
    )


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict credit risk for a single customer
    """
    try:
        # Get model and preprocessors
        model = ModelLoader.get_model()
        le, scaler, feature_names = ModelLoader.get_preprocessors()
        
        # Prepare features
        features_df, customer_id = FeatureProcessor.prepare_features(request)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features_df)[0, 1]
        else:
            # For models without probability, use decision function or default
            probability = float(model.predict(features_df)[0])
        
        # Determine risk category
        risk_category = FeatureProcessor.determine_risk_category(probability)
        
        # Binary prediction (threshold at 0.5)
        binary_prediction = 1 if probability >= 0.5 else 0
        
        # Create response
        return PredictionResponse(
            customer_id=customer_id,
            risk_probability=round(probability, 4),
            risk_category=risk_category,
            prediction=binary_prediction,
            model_version=MODEL_VERSION,
            features_used=len(features_df.columns)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict credit risk for multiple customers (batch processing)
    """
    try:
        model = ModelLoader.get_model()
        predictions = []
        total_probability = 0.0
        
        for customer_request in request.customers:
            # Prepare features for each customer
            features_df, customer_id = FeatureProcessor.prepare_features(customer_request)
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features_df)[0, 1]
            else:
                probability = float(model.predict(features_df)[0])
            
            risk_category = FeatureProcessor.determine_risk_category(probability)
            binary_prediction = 1 if probability >= 0.5 else 0
            
            predictions.append(
                PredictionResponse(
                    customer_id=customer_id,
                    risk_probability=round(probability, 4),
                    risk_category=risk_category,
                    prediction=binary_prediction,
                    model_version=MODEL_VERSION,
                    features_used=len(features_df.columns)
                )
            )
            
            total_probability += probability
        
        # Calculate average probability
        avg_probability = total_probability / len(predictions) if predictions else 0.0
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            avg_risk_probability=round(avg_probability, 4)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Feature info endpoint
@app.get("/features", tags=["Information"])
async def get_features():
    """
    Get information about the features used by the model
    """
    if FEATURE_NAMES is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature information not available"
        )
    
    return {
        "total_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "model_type": type(MODEL).__name__ if MODEL else "Not loaded",
        "version": MODEL_VERSION
    }


# Model info endpoint
@app.get("/model", tags=["Information"])
async def get_model_info():
    """
    Get information about the deployed model
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    model_info = {
        "model_type": type(MODEL).__name__,
        "version": MODEL_VERSION,
        "parameters": MODEL.get_params() if hasattr(MODEL, 'get_params') else {},
        "features_count": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        "supports_probability": hasattr(MODEL, 'predict_proba')
    }
    
    return model_info


# Root endpoint
@app.get("/", tags=["Information"])
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Credit Risk Probability Model API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict": "/predict (POST)",
            "batch_predict": "/predict/batch (POST)",
            "health": "/health (GET)",
            "features": "/features (GET)",
            "model_info": "/model (GET)"
        }
    }


# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": request.url.path
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )