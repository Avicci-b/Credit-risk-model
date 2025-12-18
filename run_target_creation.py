"""
Run script for proxy target creation (Task 4)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.target_engineering import create_and_merge_target


def main():
    """Run target variable creation"""
    
    print("=" * 60)
    print("TASK 4: PROXY TARGET VARIABLE CREATION")
    print("=" * 60)
    
    # Configuration
    config = {
        'n_clusters': 3,
        'clustering_method': 'kmeans',
        'random_state': 42,
        'rfm_features': [
            'recency_days',
            'transaction_count', 
            'total_amount',
            'avg_transaction_amount',
            'std_transaction_amount'
        ],
        'visualize': True,
        'save_report': True
    }
    
    # Create directories
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    print("\n1. Configuration:")
    print("-" * 30)
    print(f"   Clustering method: {config['clustering_method']}")
    print(f"   Number of clusters: {config['n_clusters']}")
    print(f"   RFM features: {config['rfm_features']}")
    
    print("\n2. Creating target variable...")
    print("-" * 30)
    
    try:
        # Run target creation
        final_df = create_and_merge_target(
            features_path='data/processed/features_with_customer.parquet',
            output_path='data/processed/dataset_with_target.parquet',
            config=config
        )
        
        print("\n3. Results Summary:")
        print("-" * 30)
        print(f"   ✓ Total customers: {final_df['CustomerId'].nunique():,}")
        print(f"   ✓ High-risk customers: {final_df['is_high_risk'].sum():,}")
        print(f"   ✓ High-risk percentage: {final_df['is_high_risk'].mean()*100:.2f}%")
        print(f"   ✓ Final dataset shape: {final_df.shape}")
        
        print("\n4. Files Created:")
        print("-" * 30)
        print("   ✓ data/processed/dataset_with_target.parquet")
        print("   ✓ reports/figures/rfm_clusters.png (visualization)")
        print("   ✓ reports/cluster_analysis_report.json (analysis)")
        
        print("\n" + "=" * 60)
        print("✅ TASK 4 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext: Task 5 - Model Training")
        print("The dataset is now ready with features and target variable.")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease ensure Task 3 is completed and features are generated.")
        print("Run: python run_feature_engineering.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()