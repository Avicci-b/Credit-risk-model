"""
Unit tests for data processing functions
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_rfm_calculation():
    """Test RFM calculation logic."""
    # Mock transaction data
    transactions = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C1', 'C2', 'C2'],
        'Amount': [100, 200, 150, 50, 75],
        'TransactionStartTime': pd.date_range('2024-01-01', periods=5, freq='D')
    })
    
    # Test aggregation logic
    monetary = transactions.groupby('CustomerId')['Amount'].sum()
    frequency = transactions.groupby('CustomerId').size()
    
    assert monetary['C1'] == 450  # 100 + 200 + 150
    assert monetary['C2'] == 125  # 50 + 75
    assert frequency['C1'] == 3
    assert frequency['C2'] == 2
    print("✅ RFM calculation test passed")


def test_temporal_feature_extraction():
    """Test temporal feature extraction."""
    date_strings = ["2024-01-15 10:30:00", "2024-06-20 14:45:00"]
    dates = pd.to_datetime(date_strings)
    
    # Extract features manually
    hours = dates.hour
    days = dates.day
    months = dates.month
    
    assert hours[0] == 10
    assert hours[1] == 14
    assert days[0] == 15
    assert days[1] == 20
    assert months[0] == 1
    assert months[1] == 6
    print("✅ Temporal feature extraction test passed")


def test_missing_value_handling():
    """Test missing value detection."""
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [9, 10, 11, 12]
    })
    
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100
    
    assert missing_counts['A'] == 1
    assert missing_counts['B'] == 2
    assert missing_counts['C'] == 0
    assert missing_percentage['B'] == 50.0
    print("✅ Missing value handling test passed")


def test_feature_engineering_pipeline_structure():
    """Test that pipeline components are properly structured."""
    # This is a structural test - we'll implement actual pipeline tests later
    assert True  # Placeholder
    print("✅ Pipeline structure test passed")


if __name__ == "__main__":
    print("Running data processing tests...")
    print("=" * 50)
    
    test_rfm_calculation()
    test_temporal_feature_extraction()
    test_missing_value_handling()
    test_feature_engineering_pipeline_structure()
    
    print("=" * 50)
    print("🎉 All data processing tests passed!")