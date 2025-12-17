"""
Runner script for feature engineering pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import process_data_for_training, save_processed_data


def main():
    """Run feature engineering on actual data"""
    
    # Configuration
    config = {
        'snapshot_date': '2019-02-28',  # One month after last transaction
        'encoding_strategy': 'label',  # Will change to 'woe' in Task 4 with target
        'scale_numerical': False  # Will add scaling in Task 5 with modeling
    }
    
    print("Starting Feature Engineering Pipeline...")
    print("=" * 50)
    
    # File paths
    raw_data_path = 'data/raw/data.csv'
    processed_data_path = 'data/processed/features.parquet'
    
    # Ensure directories exist
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        # Process data
        print("1. Processing raw data...")
        X_processed, y, pipeline = process_data_for_training(
            data_path=raw_data_path,
            config=config
        )
        
        print(f"   ✓ Processed {X_processed.shape[0]} samples")
        print(f"   ✓ Generated {X_processed.shape[1]} features")
        
        # Save processed data
        print("\n2. Saving processed data...")
        save_processed_data(
            X=X_processed,
            y=y,
            output_path=processed_data_path
        )
        
        print(f"   ✓ Saved to {processed_data_path}")
        
        # Display feature summary
        print("\n3. Feature Summary:")
        print("=" * 30)
        
        # Group features by type
        rfm_features = [f for f in X_processed.columns if 'recency' in f or 'transaction_count' in f 
                       or 'amount' in f and 'avg' in f or 'std' in f or 'total' in f]
        temporal_features = [f for f in X_processed.columns if 'transaction_' in f or 'hour' in f 
                           or 'day' in f or 'month' in f or 'week' in f]
        categorical_features = [f for f in X_processed.columns if f not in rfm_features + temporal_features 
                              and not f.startswith('Customer')]
        
        print(f"   RFM Features: {len(rfm_features)}")
        print(f"   Temporal Features: {len(temporal_features)}")
        print(f"   Categorical Features: {len(categorical_features)}")
        print(f"   Total Features: {X_processed.shape[1]}")
        
        print("\n4. Sample Features:")
        print("=" * 30)
        print(X_processed.head().T)
        
        print("\n" + "=" * 50)
        print("✅ FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {str(e)}")
        print("Please ensure data/raw/data.csv exists")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()