"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np


class PredictionRequest(BaseModel):
    """
    Request model for credit risk prediction
    Matches the features used in model training
    """
    # Basic customer info
    CustomerId: Optional[str] = Field(None, description="Customer identifier")
    
    # Transaction features
    CurrencyCode: str = Field("UGX", description="Currency code")
    CountryCode: int = Field(256, description="Country code")
    ProductCategory: str = Field("airtime", description="Product category")
    Amount: float = Field(1000.0, description="Transaction amount")
    Value: float = Field(1000.0, description="Transaction value")
    PricingStrategy: int = Field(2, description="Pricing strategy")
    FraudResult: int = Field(0, description="Fraud indicator")
    
    # RFM features
    recency_days: float = Field(15.0, description="Days since last transaction")
    transaction_count: float = Field(10.0, description="Number of transactions")
    total_amount: float = Field(5000.0, description="Total transaction amount")
    avg_transaction_amount: float = Field(500.0, description="Average transaction amount")
    std_transaction_amount: float = Field(100.0, description="Std of transaction amounts")
    log_total_amount: float = Field(8.5, description="Log of total amount")
    
    # Temporal features
    transaction_hour: int = Field(14, description="Hour of transaction (0-23)")
    transaction_day: int = Field(15, description="Day of month (1-31)")
    transaction_month: int = Field(11, description="Month (1-12)")
    transaction_year: int = Field(2024, description="Year")
    transaction_dayofweek: int = Field(2, description="Day of week (0=Monday)")
    transaction_weekofyear: int = Field(46, description="Week of year (1-53)")
    hour_sin: float = Field(0.5, description="Sine of hour (cyclical)")
    hour_cos: float = Field(0.866, description="Cosine of hour (cyclical)")
    dayofweek_sin: float = Field(0.434, description="Sine of day of week")
    dayofweek_cos: float = Field(0.901, description="Cosine of day of week")
    is_business_hours: int = Field(1, description="1 if business hours, 0 otherwise")
    is_weekend: int = Field(0, description="1 if weekend, 0 otherwise")
    
    class Config:
        schema_extra = {
            "example": {
                "CustomerId": "CustomerId_1234",
                "CurrencyCode": "UGX",
                "CountryCode": 256,
                "ProductCategory": "airtime",
                "Amount": 1000.0,
                "Value": 1000.0,
                "PricingStrategy": 2,
                "FraudResult": 0,
                "recency_days": 15.0,
                "transaction_count": 10.0,
                "total_amount": 5000.0,
                "avg_transaction_amount": 500.0,
                "std_transaction_amount": 100.0,
                "log_total_amount": 8.5,
                "transaction_hour": 14,
                "transaction_day": 15,
                "transaction_month": 11,
                "transaction_year": 2024,
                "transaction_dayofweek": 2,
                "transaction_weekofyear": 46,
                "hour_sin": 0.5,
                "hour_cos": 0.866,
                "dayofweek_sin": 0.434,
                "dayofweek_cos": 0.901,
                "is_business_hours": 1,
                "is_weekend": 0
            }
        }
    
    @validator('transaction_hour')
    def validate_hour(cls, v):
        if v < 0 or v > 23:
            raise ValueError('transaction_hour must be between 0 and 23')
        return v
    
    @validator('transaction_day')
    def validate_day(cls, v):
        if v < 1 or v > 31:
            raise ValueError('transaction_day must be between 1 and 31')
        return v
    
    @validator('transaction_month')
    def validate_month(cls, v):
        if v < 1 or v > 12:
            raise ValueError('transaction_month must be between 1 and 12')
        return v


class PredictionResponse(BaseModel):
    """
    Response model for credit risk prediction
    """
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of being high-risk")
    risk_category: str = Field(..., description="Risk category (LOW/MEDIUM/HIGH)")
    prediction: int = Field(..., description="Binary prediction (0=low-risk, 1=high-risk)")
    model_version: str = Field(..., description="Model version used for prediction")
    features_used: int = Field(..., description="Number of features used in prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CustomerId_1234",
                "risk_probability": 0.15,
                "risk_category": "LOW",
                "prediction": 0,
                "model_version": "1.0.0",
                "features_used": 25
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Model version")
    uptime: float = Field(..., description="Service uptime in seconds")


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch predictions
    """
    customers: List[PredictionRequest] = Field(..., description="List of customer data")
    
    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
                        "CustomerId": "CustomerId_1234",
                        "CurrencyCode": "UGX",
                        "CountryCode": 256,
                        "ProductCategory": "airtime",
                        "Amount": 1000.0,
                        "Value": 1000.0,
                        "PricingStrategy": 2,
                        "FraudResult": 0,
                        "recency_days": 15.0,
                        "transaction_count": 10.0,
                        "total_amount": 5000.0,
                        "avg_transaction_amount": 500.0,
                        "std_transaction_amount": 100.0,
                        "log_total_amount": 8.5,
                        "transaction_hour": 14,
                        "transaction_day": 15,
                        "transaction_month": 11,
                        "transaction_year": 2024,
                        "transaction_dayofweek": 2,
                        "transaction_weekofyear": 46,
                        "hour_sin": 0.5,
                        "hour_cos": 0.866,
                        "dayofweek_sin": 0.434,
                        "dayofweek_cos": 0.901,
                        "is_business_hours": 1,
                        "is_weekend": 0
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch predictions
    """
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_customers: int = Field(..., description="Total number of customers processed")
    avg_risk_probability: float = Field(..., description="Average risk probability")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "customer_id": "CustomerId_1234",
                        "risk_probability": 0.15,
                        "risk_category": "LOW",
                        "prediction": 0,
                        "model_version": "1.0.0",
                        "features_used": 25
                    }
                ],
                "total_customers": 1,
                "avg_risk_probability": 0.15
            }
        }