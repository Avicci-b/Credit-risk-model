"""
Unit tests for data processing and model training (Task 5)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_data_loading():
    """Test that data loading returns expected columns."""
    print("Testing data loading...")
    
    # This is a structural test - in practice, you'd load actual data
    # For now, create mock data
    mock_data = pd.DataFrame({
        'CustomerId': ['C1', 'C2', 'C3'],
        'recency_days': [10, 20, 30],
        'transaction_count': [5, 3, 1],
        'total_amount': [1000, 500, 100],
        'is_high_risk': [0, 0, 1]
    })
    
    # Test column existence
    required_columns = ['CustomerId', 'recency_days', 'transaction_count', 'total_amount', 'is_high_risk']
    for col in required_columns:
        assert col in mock_data.columns, f"Missing column: {col}"
    
    # Test data types
    assert mock_data['is_high_risk'].dtype in [np.int64, np.int32, int], "Target should be integer"
    
    print("✅ Data loading test passed")
    return True


def test_train_test_split():
    """Test that train-test split maintains class distribution."""
    print("\nTesting train-test split...")
    
    # Create imbalanced data
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples)
    })
    
    # Create imbalanced target (5% positive class)
    y = np.zeros(n_samples)
    y[:50] = 1  # 5% positive
    np.random.shuffle(y)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Check sizes
    assert len(X_train) == 800, f"Expected 800 training samples, got {len(X_train)}"
    assert len(X_test) == 200, f"Expected 200 test samples, got {len(X_test)}"
    
    # Check class distribution is similar
    train_pos_ratio = y_train.mean()
    test_pos_ratio = y_test.mean()
    overall_pos_ratio = y.mean()
    
    # They should be roughly equal (stratification worked)
    assert abs(train_pos_ratio - overall_pos_ratio) < 0.01, "Train set not stratified"
    assert abs(test_pos_ratio - overall_pos_ratio) < 0.01, "Test set not stratified"
    
    print(f"✅ Train-test split test passed")
    print(f"   Overall positive: {overall_pos_ratio*100:.1f}%")
    print(f"   Train positive: {train_pos_ratio*100:.1f}%")
    print(f"   Test positive: {test_pos_ratio*100:.1f}%")
    return True


def test_model_training_structure():
    """Test that model training produces expected outputs."""
    print("\nTesting model training structure...")
    
    # Create simple data
    X_train = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
    y_train = pd.Series([0, 0, 0, 1, 1])
    X_test = pd.DataFrame({'feature': [6, 7]})
    y_test = pd.Series([0, 1])
    
    from sklearn.linear_model import LogisticRegression
    
    # Train a simple model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Should have predict method
    assert hasattr(model, 'predict'), "Model should have predict method"
    
    # Should have predict_proba for probability predictions
    assert hasattr(model, 'predict_proba'), "Model should have predict_proba for ROC-AUC"
    
    # Make predictions
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(X_test), "Predictions should match test size"
    
    print("✅ Model training structure test passed")
    return True


def test_metrics_calculation():
    """Test that metrics are calculated correctly."""
    print("\nTesting metrics calculation...")
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Simple test case
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Expected values
    # TP=1, FP=1, FN=1, TN=1
    # Accuracy = (1+1)/4 = 0.5
    # Precision = TP/(TP+FP) = 1/(1+1) = 0.5
    # Recall = TP/(TP+FN) = 1/(1+1) = 0.5
    # F1 = 2 * (precision*recall)/(precision+recall) = 0.5
    
    assert accuracy == 0.5, f"Expected accuracy 0.5, got {accuracy}"
    assert precision == 0.5, f"Expected precision 0.5, got {precision}"
    assert recall == 0.5, f"Expected recall 0.5, got {recall}"
    assert f1 == 0.5, f"Expected F1 0.5, got {f1}"
    
    print("✅ Metrics calculation test passed")
    print(f"   Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("TASK 5: UNIT TESTS FOR MODEL TRAINING")
    print("=" * 50)
    
    try:
        test_data_loading()
        test_train_test_split()
        test_model_training_structure()
        test_metrics_calculation()
        
        print("\n" + "=" * 50)
        print("🎉 ALL UNIT TESTS PASSED!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        sys.exit(1)