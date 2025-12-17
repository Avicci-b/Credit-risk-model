"""
Feature Engineering Pipeline for Credit Risk Model
Transforms raw transaction data into model-ready features
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn components
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Feature engineering libraries
from category_encoders import WOEEncoder, TargetEncoder
from xverse.transformer import MonotonicBinning

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate raw data"""
    
    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """
        Load transaction data from CSV file
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Convert date column
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            logger.info(f"Date range: {df['TransactionStartTime'].min()} to {df['TransactionStartTime'].max()}")
        
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """
        Basic data validation
        """
        required_columns = ['CustomerId', 'Amount', 'TransactionStartTime']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        if df.empty:
            logger.error("DataFrame is empty")
            return False
        
        logger.info("Data validation passed")
        return True


class RFMCalculator(BaseEstimator, TransformerMixin):
    """
    Calculate RFM (Recency, Frequency, Monetary) features per customer
    """
    
    def __init__(self, snapshot_date: Optional[str] = None):
        """
        Initialize RFM calculator
        
        Args:
            snapshot_date: Reference date for recency calculation (YYYY-MM-DD)
                          If None, uses max transaction date in data
        """
        self.snapshot_date = snapshot_date
        self.rfm_features_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Calculate RFM metrics (fit method for pipeline compatibility)
        """
        logger.info("Calculating RFM features...")
        self.rfm_features_ = self._calculate_rfm(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Merge RFM features back to transaction data
        """
        if self.rfm_features_ is None:
            self.rfm_features_ = self._calculate_rfm(X)
        
        # Merge RFM features
        X_transformed = X.copy()
        X_transformed = X_transformed.merge(
            self.rfm_features_, 
            on='CustomerId', 
            how='left'
        )
        
        logger.info(f"Added RFM features: {list(self.rfm_features_.columns[1:])}")
        return X_transformed
    
    def _calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer - FIXED for timezone issue
        """
        # Set snapshot date

        if self.snapshot_date:
            snapshot_date = pd.to_datetime(self.snapshot_date)
        else:
            snapshot_date = df['TransactionStartTime'].max()
    
        # Ensure both datetimes are timezone-naive or both are timezone-aware
        # Convert snapshot_date to same timezone as TransactionStartTime if needed
        if hasattr(df['TransactionStartTime'].dtype, 'tz') and df['TransactionStartTime'].dt.tz is not None:
        # Data has timezone info 
            if snapshot_date.tz is None:
            # snapshot_date is timezone-naive, make it aware (UTC)
                snapshot_date = snapshot_date.tz_localize('UTC')
        else:
        # Data is timezone-naive
            if snapshot_date.tz is not None:
            # snapshot_date is timezone-aware, make it naive
                snapshot_date = snapshot_date.tz_localize(None)
    
        # Make a copy to avoid modifying original
        df_copy = df.copy()
    
        # Calculate recency - handle timezone properly
        recency = df_copy.groupby('CustomerId')['TransactionStartTime'].max()
    
        # Ensure both are same type for subtraction
        if recency.dt.tz is not None and snapshot_date.tz is None:
            snapshot_date = snapshot_date.tz_localize(recency.dt.tz)
        elif recency.dt.tz is None and snapshot_date.tz is not None:
            snapshot_date = snapshot_date.tz_localize(None)
    
        recency_days = (snapshot_date - recency).dt.days
    
        # Calculate frequency (count of transactions)
        frequency = df_copy.groupby('CustomerId').size()
    
        # Calculate monetary values
        monetary = df_copy.groupby('CustomerId')['Amount'].agg(['sum', 'mean', 'std'])
    
        # Combine into RFM dataframe
        rfm_df = pd.DataFrame({
            'recency_days': recency_days,
            'transaction_count': frequency,
            'total_amount': monetary['sum'],
            'avg_transaction_amount': monetary['mean'],
            'std_transaction_amount': monetary['std']
        }).reset_index()
    
        # Handle missing std (when only one transaction)
        rfm_df['std_transaction_amount'] = rfm_df['std_transaction_amount'].fillna(0)
    
        # Add monetary score (log transform for normalization)
        rfm_df['log_total_amount'] = np.log1p(rfm_df['total_amount'].abs())
    
        logger.info(f"RFM calculated for {len(rfm_df)} customers")
    
        return rfm_df
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from transaction timestamps
    """
    
    def __init__(self, extract_cyclical: bool = True):
        self.extract_cyclical = extract_cyclical
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features
        """
        X_transformed = X.copy()
        
        if 'TransactionStartTime' not in X_transformed.columns:
            logger.warning("TransactionStartTime not found, skipping temporal features")
            return X_transformed
        
        # Basic temporal features
        X_transformed['transaction_hour'] = X_transformed['TransactionStartTime'].dt.hour
        X_transformed['transaction_day'] = X_transformed['TransactionStartTime'].dt.day
        X_transformed['transaction_month'] = X_transformed['TransactionStartTime'].dt.month
        X_transformed['transaction_year'] = X_transformed['TransactionStartTime'].dt.year
        X_transformed['transaction_dayofweek'] = X_transformed['TransactionStartTime'].dt.dayofweek
        X_transformed['transaction_weekofyear'] = X_transformed['TransactionStartTime'].dt.isocalendar().week
        
        # Cyclical features (for hour, dayofweek)
        if self.extract_cyclical:
            X_transformed['hour_sin'] = np.sin(2 * np.pi * X_transformed['transaction_hour']/24)
            X_transformed['hour_cos'] = np.cos(2 * np.pi * X_transformed['transaction_hour']/24)
            X_transformed['dayofweek_sin'] = np.sin(2 * np.pi * X_transformed['transaction_dayofweek']/7)
            X_transformed['dayofweek_cos'] = np.cos(2 * np.pi * X_transformed['transaction_dayofweek']/7)
        
        # Business hours indicator
        X_transformed['is_business_hours'] = X_transformed['transaction_hour'].between(9, 17).astype(int)
        
        # Weekend indicator
        X_transformed['is_weekend'] = X_transformed['transaction_dayofweek'].isin([5, 6]).astype(int)
        
        logger.info(f"Extracted {len([c for c in X_transformed.columns if 'transaction' in c or 'hour' in c or 'day' in c])} temporal features")
        return X_transformed


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Handle categorical variable encoding
    """
    
    def __init__(self, encoding_strategy: str = 'woe', 
                 target_col: Optional[str] = None,
                 top_n_categories: int = 10):
        """
        Args:
            encoding_strategy: 'onehot', 'label', 'woe', 'target'
            target_col: Required for WOE and target encoding
            top_n_categories: Keep top N categories, group others as 'Other'
        """
        self.encoding_strategy = encoding_strategy
        self.target_col = target_col
        self.top_n_categories = top_n_categories
        self.encoders_ = {}
        self.category_mappings_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        # Identify categorical columns
        self.categorical_cols_ = self._identify_categorical_columns(X)
        
        if not self.categorical_cols_:
            logger.info("No categorical columns found")
            return self
        
        logger.info(f"Categorical columns: {self.categorical_cols_}")
        
        # Apply encoding strategy
        for col in self.categorical_cols_:
            self._fit_column_encoder(X, col, y)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        
        if not self.categorical_cols_:
            return X_transformed
        
        for col in self.categorical_cols_:
            if col in self.category_mappings_:
                X_transformed = self._transform_column(X_transformed, col)
        
        logger.info(f"Encoded {len(self.categorical_cols_)} categorical columns")
        return X_transformed
    
    def _identify_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """Identify categorical columns"""
        categorical_cols = []
        
        # Object type columns
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Low cardinality numeric columns (could be categorical)
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if X[col].nunique() < 20 and col not in ['Amount', 'Value']:
                categorical_cols.append(col)
        
        categorical_cols.extend(object_cols)
        
        # Remove non-categorical columns
        non_categorical = ['TransactionId', 'TransactionStartTime', 'CustomerId', 
                          'AccountId', 'SubscriptionId', 'BatchId']
        categorical_cols = [c for c in categorical_cols if c not in non_categorical]
        
        return categorical_cols
    
    def _fit_column_encoder(self, X: pd.DataFrame, col: str, y=None):
        """Fit encoder for a specific column"""
        if self.encoding_strategy == 'label':
            # Label encoding
            encoder = LabelEncoder()
            encoder.fit(X[col].astype(str))
            self.encoders_[col] = encoder
            
        elif self.encoding_strategy in ['woe', 'target'] and y is not None:
            # WOE or Target encoding
            if self.encoding_strategy == 'woe':
                encoder = WOEEncoder()
            else:
                encoder = TargetEncoder()
            
            encoder.fit(X[col], y)
            self.encoders_[col] = encoder
            
        # For one-hot, we just track unique values
        elif self.encoding_strategy == 'onehot':
            unique_vals = X[col].value_counts().head(self.top_n_categories).index.tolist()
            self.category_mappings_[col] = unique_vals
    
    def _transform_column(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        """Transform a specific column"""
        if self.encoding_strategy == 'label':
            X[col] = self.encoders_[col].transform(X[col].astype(str))
            
        elif self.encoding_strategy in ['woe', 'target']:
            # These create new columns, drop original
            encoded = self.encoders_[col].transform(X[[col]])
            for encoded_col in encoded.columns:
                if encoded_col != col:
                    X[encoded_col] = encoded[encoded_col]
            X = X.drop(columns=[col])
            
        elif self.encoding_strategy == 'onehot':
            # Limit to top N categories
            top_categories = self.category_mappings_[col]
            X[col] = X[col].apply(lambda x: x if x in top_categories else 'Other')
            
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select final features for modeling
    """
    
    def __init__(self, feature_list: Optional[List[str]] = None):
        self.feature_list = feature_list
        
    def fit(self, X, y=None):
        if self.feature_list is None:
            # Default feature selection if none provided
            self.feature_list = self._select_features_automatically(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select only specified features"""
        if self.feature_list:
            available_features = [f for f in self.feature_list if f in X.columns]
            missing_features = [f for f in self.feature_list if f not in X.columns]
            
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
            
            return X[available_features]
        return X
    
    def _select_features_automatically(self, X: pd.DataFrame) -> List[str]:
        """Automatically select important features"""
        # Exclude ID and date columns
        exclude_patterns = ['Id', 'Time', 'Date', 'Customer', 'Account', 'Subscription']
        
        features = []
        for col in X.columns:
            if not any(pattern in col for pattern in exclude_patterns):
                features.append(col)
        
        return features


def create_feature_pipeline(config: Dict) -> Pipeline:
    """
    Create complete feature engineering pipeline
    
    Args:
        config: Dictionary with pipeline configuration
    
    Returns:
        sklearn Pipeline object
    """
    # Extract config parameters
    snapshot_date = config.get('snapshot_date', None)
    encoding_strategy = config.get('encoding_strategy', 'woe')
    scale_numerical = config.get('scale_numerical', True)
    
    # Define pipeline steps
    steps = [
        ('rfm_calculator', RFMCalculator(snapshot_date=snapshot_date)),
        ('temporal_extractor', TemporalFeatureExtractor()),
    ]
    
    # Add categorical encoding step (will be fit with target later)
    steps.append(('categorical_encoder', CategoricalEncoder(
        encoding_strategy=encoding_strategy,
        target_col='is_high_risk'  # Will be added in Task 4
    )))
    
    # Add numerical scaling if requested
    if scale_numerical:
        # We'll handle this with ColumnTransformer for specific columns
        pass
    
    # Add feature selector
    steps.append(('feature_selector', FeatureSelector()))
    
    # Create pipeline
    pipeline = Pipeline(steps)
    
    logger.info(f"Created feature pipeline with {len(steps)} steps")
    return pipeline


def process_data_for_training(data_path: str, config: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to process raw data for training
    
    Args:
        data_path: Path to raw data CSV
        config: Pipeline configuration
    
    Returns:
        X_processed: Feature matrix
        y: Target variable (will be None for now, added in Task 4)
    """
    if config is None:
        config = {
            'snapshot_date': '2019-02-28',  # One month after last transaction
            'encoding_strategy': 'woe',
            'scale_numerical': True
        }
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_data(data_path)
    
    # Validate data
    if not data_loader.validate_data(df):
        raise ValueError("Data validation failed")
    
    # For now, return features only (target will be added in Task 4)
    X = df.copy()
    
    # Create and apply pipeline
    pipeline = create_feature_pipeline(config)
    X_processed = pipeline.fit_transform(X)
    
    logger.info(f"Processed data shape: {X_processed.shape}")
    logger.info(f"Processed columns: {list(X_processed.columns)}")
    
    # y will be None for now (created in Task 4)
    y = None
    
    return X_processed, y, pipeline


def save_processed_data(X: pd.DataFrame, y: pd.DataFrame, output_path: str):
    """
    Save processed data to disk
    """
    # Combine features and target
    if y is not None:
        data_to_save = X.copy()
        data_to_save['target'] = y
    else:
        data_to_save = X
    
    # Save to parquet for efficiency
    if output_path.endswith('.parquet'):
        data_to_save.to_parquet(output_path, index=False)
    else:
        data_to_save.to_csv(output_path, index=False)
    
    logger.info(f"Saved processed data to {output_path}")
    logger.info(f"Saved data shape: {data_to_save.shape}")


if __name__ == "__main__":
    """
    Example usage
    """
    # Example configuration
    config = {
        'snapshot_date': '2019-02-28',
        'encoding_strategy': 'woe',
        'scale_numerical': True
    }
    
    try:
        # Process data
        X_processed, y, pipeline = process_data_for_training(
            data_path='../data/raw/data.csv',
            config=config
        )
        
        # Save processed data
        save_processed_data(
            X=X_processed,
            y=y,
            output_path='../data/processed/training_data.parquet'
        )
        
        print("✅ Feature engineering completed successfully!")
        print(f"   Features shape: {X_processed.shape}")
        print(f"   Features: {list(X_processed.columns)}")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise