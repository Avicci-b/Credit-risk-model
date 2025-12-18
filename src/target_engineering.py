"""
Proxy Target Variable Engineering
Create credit risk target variable via RFM clustering
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Clustering and preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TargetVariableCreator:
    """
    Create proxy target variable using RFM clustering
    """
    
    def __init__(self, n_clusters: int = 3, method: str = 'kmeans', 
                 random_state: int = 42):
        """
        Initialize target creator
        
        Args:
            n_clusters: Number of clusters for segmentation
            method: Clustering method ('kmeans' or 'gmm')
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.cluster_labels_ = None
        self.cluster_stats_ = None
        self.high_risk_cluster_ = None
        
    def create_target_from_features(self, features_df: pd.DataFrame, 
                                   customer_id_col: str = 'CustomerId',
                                   rfm_features: list = None) -> pd.DataFrame:
        """
        Create target variable from pre-computed RFM features
        
        Args:
            features_df: DataFrame with RFM features (can be the processed data)
            customer_id_col: Name of customer ID column
            rfm_features: List of RFM feature names to use for clustering
            
        Returns:
            DataFrame with customer_id and is_high_risk column
        """
        logger.info("Creating proxy target variable...")
        
        # Default RFM features if not specified
        if rfm_features is None:
            rfm_features = [
                'recency_days', 
                'transaction_count', 
                'total_amount',
                'avg_transaction_amount',
                'std_transaction_amount'
            ]
        
        # Extract unique customers with their RFM features
        customer_features = self._extract_customer_features(
            features_df, customer_id_col, rfm_features
        )
        
        # Perform clustering
        clustered_customers = self._perform_clustering(customer_features)
        
        # Identify high-risk cluster
        clustered_customers = self._identify_high_risk_cluster(clustered_customers)
        
        # Create target variable
        target_df = self._create_target_variable(clustered_customers, customer_id_col)
        
        logger.info(f"Target variable created for {len(target_df)} customers")
        logger.info(f"High-risk customers: {target_df['is_high_risk'].sum()} "
                   f"({target_df['is_high_risk'].mean()*100:.1f}%)")
        
        return target_df
    
    def _extract_customer_features(self, features_df: pd.DataFrame, 
                                  customer_id_col: str, 
                                  rfm_features: list) -> pd.DataFrame:
        """
        Extract unique customer RFM features
        """
        # Get unique customers (take first occurrence of each customer)
        customer_features = features_df.drop_duplicates(subset=[customer_id_col], 
                                                       keep='first').copy()
        
        # Select only RFM features that exist
        available_features = [f for f in rfm_features if f in customer_features.columns]
        missing_features = [f for f in rfm_features if f not in customer_features.columns]
        
        if missing_features:
            logger.warning(f"Missing RFM features: {missing_features}")
        
        # Keep only needed columns
        columns_to_keep = [customer_id_col] + available_features
        customer_features = customer_features[columns_to_keep].set_index(customer_id_col)
        
        # Handle any missing values
        customer_features = customer_features.fillna(customer_features.median())
        
        logger.info(f"Extracted features for {len(customer_features)} customers")
        logger.info(f"Using features: {available_features}")
        
        return customer_features
    
    def _perform_clustering(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        """
        Perform clustering on RFM features
        """
        logger.info(f"Performing {self.method} clustering with {self.n_clusters} clusters...")
        
        # Scale features
        scaled_features = self.scaler.fit_transform(customer_features)
        
        # Perform clustering
        if self.method == 'kmeans':
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
        elif self.method == 'gmm':
            self.cluster_model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Fit and predict
        cluster_labels = self.cluster_model.fit_predict(scaled_features)
        self.cluster_labels_ = cluster_labels
        
        # Add cluster labels to dataframe
        clustered_df = customer_features.copy()
        clustered_df['cluster'] = cluster_labels
        clustered_df['cluster'] = clustered_df['cluster'].astype('category')
        
        # Calculate cluster statistics
        self._calculate_cluster_statistics(clustered_df)
        
        return clustered_df
    
    def _calculate_cluster_statistics(self, clustered_df: pd.DataFrame):
        """
        Calculate statistics for each cluster
        """
        cluster_stats = []
        
        for cluster_id in range(self.n_clusters):
            cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
            
            stats = {
                'cluster': cluster_id,
                'size': len(cluster_data),
                'size_percentage': len(cluster_data) / len(clustered_df) * 100
            }
            
            # Calculate mean for each feature
            for feature in clustered_df.columns:
                if feature != 'cluster':
                    stats[f'{feature}_mean'] = cluster_data[feature].mean()
                    stats[f'{feature}_median'] = cluster_data[feature].median()
            
            cluster_stats.append(stats)
        
        self.cluster_stats_ = pd.DataFrame(cluster_stats)
        
        logger.info("Cluster statistics calculated:")
        for _, row in self.cluster_stats_.iterrows():
            logger.info(f"  Cluster {row['cluster']}: {row['size']} customers "
                       f"({row['size_percentage']:.1f}%)")
    
    def _identify_high_risk_cluster(self, clustered_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify which cluster represents high-risk customers
        High-risk = High recency (inactive), low frequency, low monetary
        """
        logger.info("Identifying high-risk cluster...")
        
        # Get cluster means
        cluster_means = clustered_df.groupby('cluster').mean()
        
        # Score each cluster (higher score = higher risk)
        risk_scores = {}
        
        for cluster_id in cluster_means.index:
            # High recency (inactive) = higher risk
            recency_score = cluster_means.loc[cluster_id, 'recency_days'] if 'recency_days' in cluster_means.columns else 0
            
            # Low frequency = higher risk
            if 'transaction_count' in cluster_means.columns:
                freq_score = -cluster_means.loc[cluster_id, 'transaction_count']
            else:
                freq_score = 0
            
            # Low monetary = higher risk
            if 'total_amount' in cluster_means.columns:
                monetary_score = -cluster_means.loc[cluster_id, 'total_amount']
            else:
                monetary_score = 0
            
            # Combined risk score
            risk_score = recency_score + freq_score + monetary_score
            risk_scores[cluster_id] = risk_score
        
        # Cluster with highest risk score is high-risk
        self.high_risk_cluster_ = max(risk_scores, key=risk_scores.get)
        
        logger.info(f"Risk scores: {risk_scores}")
        logger.info(f"Identified cluster {self.high_risk_cluster_} as high-risk")
        
        # Add risk labels
        clustered_df['is_high_risk'] = (clustered_df['cluster'] == self.high_risk_cluster_).astype(int)
        
        return clustered_df
    
    def _create_target_variable(self, clustered_df: pd.DataFrame, 
                               customer_id_col: str) -> pd.DataFrame:
        """
        Create target variable DataFrame
        """
        target_df = clustered_df[['is_high_risk']].copy()
        target_df[customer_id_col] = target_df.index
        target_df = target_df.reset_index(drop=True)
        target_df = target_df[[customer_id_col, 'is_high_risk']]
        
        return target_df
    
    def visualize_clusters(self, clustered_df: pd.DataFrame, 
                          save_path: str = None):
        """
        Visualize RFM clusters
        """
        logger.info("Creating cluster visualizations...")
        
        # Select features for visualization
        vis_features = []
        for feature in ['recency_days', 'transaction_count', 'total_amount']:
            if feature in clustered_df.columns:
                vis_features.append(feature)
        
        if len(vis_features) < 2:
            logger.warning("Not enough features for visualization")
            return
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # 2D scatter plots
        if len(vis_features) >= 2:
            ax1 = fig.add_subplot(2, 3, 1)
            scatter1 = ax1.scatter(clustered_df[vis_features[0]], 
                                 clustered_df[vis_features[1]],
                                 c=clustered_df['cluster'], 
                                 cmap='viridis', alpha=0.6)
            ax1.set_xlabel(vis_features[0])
            ax1.set_ylabel(vis_features[1])
            ax1.set_title('Cluster Segmentation')
            plt.colorbar(scatter1, ax=ax1)
        
        if len(vis_features) >= 3:
            # 3D scatter plot
            ax2 = fig.add_subplot(2, 3, 2, projection='3d')
            scatter2 = ax2.scatter(clustered_df[vis_features[0]],
                                 clustered_df[vis_features[1]],
                                 clustered_df[vis_features[2]],
                                 c=clustered_df['cluster'],
                                 cmap='viridis', alpha=0.6)
            ax2.set_xlabel(vis_features[0])
            ax2.set_ylabel(vis_features[1])
            ax2.set_zlabel(vis_features[2])
            ax2.set_title('3D Cluster View')
        
        # Box plots for each cluster
        for i, feature in enumerate(vis_features[:3]):
            ax = fig.add_subplot(2, 3, i+4)
            clustered_df.boxplot(column=feature, by='cluster', ax=ax)
            ax.set_title(f'{feature} by Cluster')
            ax.set_xlabel('Cluster')
            ax.set_ylabel(feature)
        
        # Risk distribution
        ax_risk = fig.add_subplot(2, 3, 6)
        risk_counts = clustered_df['is_high_risk'].value_counts()
        colors = ['green', 'red'] if len(risk_counts) == 2 else ['blue']
        ax_risk.bar(risk_counts.index.astype(str), risk_counts.values, color=colors)
        ax_risk.set_title('High-Risk Distribution')
        ax_risk.set_xlabel('Is High Risk')
        ax_risk.set_ylabel('Number of Customers')
        
        plt.suptitle('RFM Cluster Analysis for Credit Risk Proxy', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Cluster visualization saved to {save_path}")
        
        plt.show()
    
    def generate_cluster_report(self, clustered_df: pd.DataFrame, 
                               output_path: str = None) -> dict:
        """
        Generate detailed cluster analysis report
        """
        report = {
            'clustering_method': self.method,
            'n_clusters': self.n_clusters,
            'total_customers': len(clustered_df),
            'high_risk_cluster': int(self.high_risk_cluster_),
            'high_risk_customers': int(clustered_df['is_high_risk'].sum()),
            'high_risk_percentage': float(clustered_df['is_high_risk'].mean() * 100),
            'cluster_statistics': self.cluster_stats_.to_dict('records'),
            'feature_importance': self._calculate_feature_importance(clustered_df)
        }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Cluster report saved to {output_path}")
        
        return report
    
    def _calculate_feature_importance(self, clustered_df: pd.DataFrame) -> dict:
        """
        Calculate which features are most important for clustering
        """
        # Use variance between clusters as importance measure
        features = [col for col in clustered_df.columns if col not in ['cluster', 'is_high_risk']]
        
        importance = {}
        for feature in features:
            # Calculate between-cluster variance
            overall_mean = clustered_df[feature].mean()
            between_variance = 0
            
            for cluster_id in range(self.n_clusters):
                cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
                cluster_mean = cluster_data[feature].mean()
                cluster_size = len(cluster_data)
                between_variance += cluster_size * (cluster_mean - overall_mean) ** 2
            
            importance[feature] = float(between_variance / (len(clustered_df) * self.n_clusters))
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance


def create_and_merge_target(features_path: str, 
                           output_path: str = None,
                           config: dict = None) -> pd.DataFrame:
    """
    Main function to create and merge target variable
    
    Args:
        features_path: Path to processed features (from Task 3)
        output_path: Path to save final dataset with target
        config: Configuration dictionary
    
    Returns:
        DataFrame with features and target
    """
    if config is None:
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
    
    logger.info("=" * 50)
    logger.info("PROXY TARGET VARIABLE ENGINEERING")
    logger.info("=" * 50)
    
    try:
        # 1. Load processed features
        logger.info("1. Loading processed features...")
        if features_path.endswith('.parquet'):
            features_df = pd.read_parquet(features_path)
        else:
            features_df = pd.read_csv(features_path)
        
        logger.info(f"   Loaded {len(features_df):,} samples with {len(features_df.columns)} features")
        
        # 2. Create target variable
        logger.info("\n2. Creating target variable via RFM clustering...")
        target_creator = TargetVariableCreator(
            n_clusters=config['n_clusters'],
            method=config['clustering_method'],
            random_state=config['random_state']
        )
        
        target_df = target_creator.create_target_from_features(
            features_df=features_df,
            customer_id_col='CustomerId',
            rfm_features=config['rfm_features']
        )
        
        # 3. Visualize clusters
        if config.get('visualize', True):
            logger.info("\n3. Generating cluster visualizations...")
            # Get customer-level data for visualization
            customer_features = target_creator._extract_customer_features(
                features_df, 'CustomerId', config['rfm_features']
            )
            clustered_customers = pd.DataFrame({
                'cluster': target_creator.cluster_labels_,
                'is_high_risk': target_df.set_index('CustomerId')['is_high_risk']
            }, index=customer_features.index)
            clustered_customers = pd.concat([customer_features, clustered_customers], axis=1)
            
            viz_path = 'reports/figures/rfm_clusters.png' if config.get('save_report', True) else None
            target_creator.visualize_clusters(clustered_customers, save_path=viz_path)
        
        # 4. Generate report
        if config.get('save_report', True):
            logger.info("\n4. Generating cluster analysis report...")
            report_path = 'reports/cluster_analysis_report.json'
            target_creator.generate_cluster_report(clustered_customers, output_path=report_path)
        
        # 5. Merge target with features
        logger.info("\n5. Merging target variable with features...")
        final_df = features_df.merge(target_df, on='CustomerId', how='left')
        
        # Fill any missing targets (shouldn't happen, but just in case)
        final_df['is_high_risk'] = final_df['is_high_risk'].fillna(0).astype(int)
        
        logger.info(f"   Final dataset: {len(final_df):,} samples, {len(final_df.columns)} columns")
        logger.info(f"   Target distribution: {final_df['is_high_risk'].sum():,} high-risk "
                   f"({final_df['is_high_risk'].mean()*100:.1f}%)")
        
        # 6. Save final dataset
        if output_path:
            logger.info(f"\n6. Saving final dataset to {output_path}")
            if output_path.endswith('.parquet'):
                final_df.to_parquet(output_path, index=False)
            else:
                final_df.to_csv(output_path, index=False)
            
            logger.info(f"   Dataset saved successfully")
        
        logger.info("\n" + "=" * 50)
        logger.info("✅ TARGET VARIABLE CREATION COMPLETED")
        logger.info("=" * 50)
        
        return final_df
        
    except Exception as e:
        logger.error(f"Error in target variable creation: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Example usage
    """
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
    
    # Create directories if needed
    import os
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        # Run target creation
        final_df = create_and_merge_target(
            features_path='data/processed/features.parquet',
            output_path='data/processed/dataset_with_target.parquet',
            config=config
        )
        
        print("\n🎯 Target Variable Summary:")
        print("=" * 30)
        print(f"Total samples: {len(final_df):,}")
        print(f"High-risk customers: {final_df['is_high_risk'].sum():,}")
        print(f"High-risk percentage: {final_df['is_high_risk'].mean()*100:.2f}%")
        print(f"Features with target: {len(final_df.columns)}")
        print("\n✅ Ready for model training in Task 5!")
        
    except FileNotFoundError:
        print("❌ Error: Processed features not found. Please run Task 3 first.")
        print("   Expected file: data/processed/features.parquet")
    except Exception as e:
        print(f"❌ Error: {str(e)}")