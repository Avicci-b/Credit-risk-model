# fix_customer_id.py
import pandas as pd
import numpy as np

print("Fixing CustomerId in processed features...")

# Load the original data to get CustomerId
original_df = pd.read_csv('data/raw/data.csv')
print(f"Original data shape: {original_df.shape}")
print(f"Original columns: {list(original_df.columns)}")

# Load processed features
features_df = pd.read_parquet('data/processed/features.parquet')
print(f"\nProcessed features shape: {features_df.shape}")
print(f"Processed columns: {list(features_df.columns)}")

# Add CustomerId back from original data
# Assuming the order is the same (should be since we didn't shuffle)
if 'CustomerId' not in features_df.columns and len(features_df) == len(original_df):
    features_df['CustomerId'] = original_df['CustomerId']
    print(f"\n✅ Added CustomerId column")
    
    # Save fixed features
    features_df.to_parquet('data/processed/features_with_customer.parquet', index=False)
    print(f"Saved to data/processed/features_with_customer.parquet")
    
    # Update the file path for Task 4
    import json
    try:
        with open('config/target_config.yaml', 'r') as f:
            config_content = f.read()
        config_content = config_content.replace(
            'features.parquet', 
            'features_with_customer.parquet'
        )
        with open('config/target_config.yaml', 'w') as f:
            f.write(config_content)
        print("Updated target config")
    except:
        pass
    
    print("\nFixed dataset info:")
    print(f"Shape: {features_df.shape}")
    print(f"Has CustomerId: {'CustomerId' in features_df.columns}")
    
else:
    print("\n❌ Could not add CustomerId - mismatch in data lengths")
    print(f"Original: {len(original_df)}, Processed: {len(features_df)}")