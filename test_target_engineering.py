"""
Test script for target variable creation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.target_engineering import TargetVariableCreator


def test_target_creator_initialization():
    """Test TargetVariableCreator initialization"""
    print("Testing TargetVariableCreator initialization...")
    
    creator = TargetVariableCreator(n_clusters=3, method='kmeans', random_state=42)
    
    assert creator.n_clusters == 3
    assert creator.method == 'kmeans'
    assert creator.random_state == 42
    assert creator.high_risk_cluster_ is None
    
    print("✅ Initialization test passed")
    return True


def test_feature_extraction():
    """Test feature extraction logic"""
    print("\nTesting feature extraction...")
    
    # Create test data
    test_data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
        'recency_days': [10, 10, 5, 5, 20],
        'transaction_count': [5, 5, 3, 3, 1],
        'total_amount': [1000, 1000, 500, 500, 100],
        'avg_transaction_amount': [200, 200, 166.67, 166.67, 100],
        'other_feature': [1, 2, 3, 4, 5]
    })
    
    creator = TargetVariableCreator()
    features = creator._extract_customer_features(
        test_data, 'CustomerId', 
        ['recency_days', 'transaction_count', 'total_amount']
    )
    
    assert len(features) == 3  # 3 unique customers
    assert 'recency_days' in features.columns
    assert 'transaction_count' in features.columns
    assert 'total_amount' in features.columns
    assert 'other_feature' not in features.columns
    
    print("✅ Feature extraction test passed")
    return True


def test_clustering_logic():
    """Test clustering logic - FIXED VERSION"""
    print("\nTesting clustering logic...")
    
    # Create simple test data
    np.random.seed(42)
    n_customers = 100
    test_features = pd.DataFrame({
        'recency_days': np.random.randint(1, 100, n_customers),
        'transaction_count': np.random.randint(1, 50, n_customers),
        'total_amount': np.random.normal(1000, 500, n_customers)
    }, index=[f'C{i}' for i in range(n_customers)])
    
    creator = TargetVariableCreator(n_clusters=3, random_state=42)
    
    # Test clustering - use actual scaler
    clustered = creator._perform_clustering(test_features)
    
    assert 'cluster' in clustered.columns
    assert len(clustered['cluster'].unique()) == 3
    assert creator.cluster_stats_ is not None
    
    print("✅ Clustering logic test passed")
    return True


def test_high_risk_identification():
    """Test high-risk cluster identification"""
    print("\nTesting high-risk identification...")
    
    # Create test clusters with known patterns
    test_data = pd.DataFrame({
        'recency_days': [30, 5, 10, 25, 3, 20],  # Higher = more risky
        'transaction_count': [1, 10, 5, 2, 15, 3],  # Lower = more risky
        'total_amount': [100, 1000, 500, 200, 1500, 300],  # Lower = more risky
        'cluster': [0, 1, 2, 0, 1, 2]
    }).set_index(pd.Index([f'C{i}' for i in range(6)]))
    
    creator = TargetVariableCreator(n_clusters=3)
    creator.cluster_stats_ = pd.DataFrame({
        'cluster': [0, 1, 2],
        'size': [2, 2, 2]
    })
    
    # Cluster 0 should be high-risk (high recency, low frequency, low monetary)
    result = creator._identify_high_risk_cluster(test_data)
    
    assert 'is_high_risk' in result.columns
    assert result['is_high_risk'].sum() > 0  # Some customers should be high-risk
    
    print("✅ High-risk identification test passed")
    return True


def test_target_variable_creation():
    """Test target variable creation"""
    print("\nTesting target variable creation...")
    
    # Create test clustered data
    test_data = pd.DataFrame({
        'recency_days': [10, 20, 30],
        'cluster': [0, 1, 2],
        'is_high_risk': [0, 1, 0]
    }, index=['C1', 'C2', 'C3'])
    
    creator = TargetVariableCreator()
    creator.high_risk_cluster_ = 1
    
    target_df = creator._create_target_variable(test_data, 'CustomerId')
    
    assert 'CustomerId' in target_df.columns
    assert 'is_high_risk' in target_df.columns
    assert len(target_df) == 3
    assert target_df['is_high_risk'].sum() == 1
    
    print("✅ Target variable creation test passed")
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("TARGET ENGINEERING TESTS")
    print("=" * 50)
    
    try:
        test_target_creator_initialization()
        test_feature_extraction()
        test_clustering_logic()
        test_high_risk_identification()
        test_target_variable_creation()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TARGET ENGINEERING TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()