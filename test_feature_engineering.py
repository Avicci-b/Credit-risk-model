"""
Test script for feature engineering pipeline - FIXED VERSION
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data_processing import (
    DataLoader, RFMCalculator, TemporalFeatureExtractor,
    create_feature_pipeline, process_data_for_training
)


def test_data_loader():
    """Test data loading functionality"""
    print("Testing DataLoader...")
    
    # Create sample data WITH ALL REQUIRED COLUMNS
    sample_data = pd.DataFrame({
        'TransactionId': [f'T{i}' for i in range(5)],
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
        'Amount': [100, 200, 50, 150, 300],
        'TransactionStartTime': pd.date_range('2024-01-01', periods=5),
        'ProductCategory': ['A', 'B', 'A', 'C', 'B']
    })
    
    # Save sample data
    sample_path = 'data/raw/test_sample.csv'
    os.makedirs('data/raw', exist_ok=True)
    sample_data.to_csv(sample_path, index=False)
    
    # Test loading
    loader = DataLoader()
    df_loaded = loader.load_data(sample_path)
    
    assert not df_loaded.empty, "Data loading failed"
    assert 'CustomerId' in df_loaded.columns, "CustomerId column missing"
    assert len(df_loaded) == 5, "Incorrect number of rows loaded"
    
    print("✅ DataLoader test passed")
    
    # Clean up
    if os.path.exists(sample_path):
        os.remove(sample_path)
    return True


def test_rfm_calculator():
    """Test RFM calculation - FIXED"""
    print("\nTesting RFMCalculator...")
    
    # Create sample data WITH TransactionId
    df = pd.DataFrame({
        'TransactionId': [f'T{i}' for i in range(5)],
        'CustomerId': ['C1', 'C1', 'C1', 'C2', 'C2'],
        'Amount': [100, 200, 150, 50, 75],
        'TransactionStartTime': pd.date_range('2024-01-01', periods=5)
    })
    
    calculator = RFMCalculator(snapshot_date='2024-01-10')
    calculator.fit(df)
    df_with_rfm = calculator.transform(df)
    
    # Check RFM features were added
    rfm_features = ['recency_days', 'transaction_count', 'total_amount', 
                    'avg_transaction_amount', 'std_transaction_amount']
    
    for feature in rfm_features:
        assert feature in df_with_rfm.columns, f"Missing RFM feature: {feature}"
    
    # Verify calculations
    customer1_data = df_with_rfm[df_with_rfm['CustomerId'] == 'C1']
    if not customer1_data.empty:
        customer1 = customer1_data.iloc[0]
        assert customer1['transaction_count'] == 3, "Incorrect frequency calculation"
        assert customer1['total_amount'] == 450, "Incorrect monetary calculation"
    
    print("✅ RFMCalculator test passed")
    return True


def test_temporal_extractor():
    """Test temporal feature extraction"""
    print("\nTesting TemporalFeatureExtractor...")
    
    df = pd.DataFrame({
        'TransactionStartTime': pd.to_datetime(['2024-01-15 10:30:00', 
                                                '2024-01-16 14:45:00',
                                                '2024-01-17 20:15:00'])
    })
    
    extractor = TemporalFeatureExtractor()
    df_with_time = extractor.transform(df)
    
    # Check temporal features
    time_features = ['transaction_hour', 'transaction_day', 'transaction_month',
                     'transaction_year', 'transaction_dayofweek']
    
    for feature in time_features:
        assert feature in df_with_time.columns, f"Missing temporal feature: {feature}"
    
    # Check specific values
    assert df_with_time.iloc[0]['transaction_hour'] == 10
    assert df_with_time.iloc[0]['transaction_day'] == 15
    assert df_with_time.iloc[0]['transaction_month'] == 1
    
    print("✅ TemporalFeatureExtractor test passed")
    return True


def test_feature_pipeline():
    """Test complete feature pipeline - FIXED"""
    print("\nTesting complete feature pipeline...")
    
    # Create more comprehensive sample data WITH REQUIRED COLUMNS
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'TransactionId': [f'T{i}' for i in range(n_samples)],
        'CustomerId': [f'C{i%10}' for i in range(n_samples)],
        'Amount': np.random.normal(100, 50, n_samples),
        'TransactionStartTime': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'ProductCategory': np.random.choice(['Airtime', 'Financial', 'Utility', 'TV'], n_samples),
        'ChannelId': np.random.choice(['Web', 'Mobile', 'App'], n_samples),
        'CountryCode': [256] * n_samples,  # Uganda
        'ProviderId': [f'P{np.random.randint(1, 6)}' for _ in range(n_samples)]
    })
    
    # Save sample data
    sample_path = 'data/raw/pipeline_test.csv'
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv(sample_path, index=False)
    
    # Test pipeline
    try:
        config = {
            'snapshot_date': '2024-01-05',
            'encoding_strategy': 'label',  # Use label for testing (no target needed)
            'scale_numerical': False
        }
        
        X_processed, y, pipeline = process_data_for_training(
            data_path=sample_path,
            config=config
        )
        
        assert X_processed is not None, "Pipeline returned None"
        assert not X_processed.empty, "Processed data is empty"
        
        print(f"✅ Pipeline test passed")
        print(f"   Input shape: {df.shape}")
        print(f"   Output shape: {X_processed.shape}")
        print(f"   Output columns: {len(X_processed.columns)}")
        print(f"   Sample columns: {list(X_processed.columns)[:10]}...")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        raise
    
    finally:
        # Clean up
        if os.path.exists(sample_path):
            os.remove(sample_path)
    
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("FEATURE ENGINEERING TESTS - FIXED VERSION")
    print("=" * 50)
    
    try:
        test_data_loader()
        test_rfm_calculator()
        test_temporal_extractor()
        test_feature_pipeline()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()