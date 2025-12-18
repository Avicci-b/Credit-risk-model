"""
Run script for model training (Task 5)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import main


if __name__ == "__main__":
    print("Starting Model Training Pipeline...")
    print("=" * 60)
    
    # Check if data exists
    data_files = [
        'data/processed/dataset_with_target.parquet',
        'data/processed/dataset_with_target.parquet'
    ]
    
    data_found = False
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
            data_found = True
            break
    
    if not data_found:
        print("❌ ERROR: No training data found!")
        print("Please complete Task 4 first and ensure datasets exist:")
        print("  - data/processed/dataset_with_target_FIXED.parquet")
        print("  - data/processed/customer_level_dataset.parquet")
        sys.exit(1)
    
    # Start MLflow server (in background)
    print("\n📊 Starting MLflow tracking...")
    print("Note: MLflow UI will be available at http://127.0.0.1:5000")
    
    # Run training
    result = main()
    
    if result == 0:
        print("\n🎉 Training completed! You can now:")
        print("1. View results in MLflow UI: mlflow ui")
        print("2. Check reports/ directory for visualizations")
        print("3. Run tests: python -m pytest tests/ -v")
    else:
        print("\n❌ Training failed. Check logs above.")
        sys.exit(1)