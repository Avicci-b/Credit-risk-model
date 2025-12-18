"""
Model Training and Tracking Script for Credit Risk Model
Task 5: Complete implementation with MLflow tracking
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Model imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import mlflow.xgboost

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import logging
import joblib
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT_NAME = "Credit_Risk_Modeling"


class ModelTrainer:
    """
    Complete model training pipeline with MLflow tracking
    """
    
    def __init__(self, data_path: str, use_customer_level: bool = True):
        """
        Initialize model trainer
        
        Args:
            data_path: Path to dataset
            use_customer_level: Use customer-level (True) or transaction-level (False) data
        """
        self.data_path = data_path
        self.use_customer_level = use_customer_level
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
        # Setup MLflow
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        # Set tracking URI (local by default)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
        
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        logger.info(f"MLflow experiment set to: {MLFLOW_EXPERIMENT_NAME}")
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    def load_and_prepare_data(self):
        """
        Load data and prepare for training
        """
        logger.info("Loading and preparing data...")
        
        # Load the appropriate dataset
        if self.use_customer_level and os.path.exists('data/processed/dataset_with_target.parquet'):
            df = pd.read_parquet('data/processed/dataset_with_target.parquet')
            logger.info("Using customer-level dataset")
        else:
            df = pd.read_parquet(self.data_path)
            logger.info("Using transaction-level dataset")
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)[:10]}...")
        
        # Separate features and target
        # Drop non-feature columns
        columns_to_drop = ['is_high_risk', 'CustomerId']
        # Only drop columns that exist
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        X = df.drop(columns=columns_to_drop)
        y = df['is_high_risk']
        
        # Check class distribution
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        logger.info(f"High-risk percentage: {y.mean()*100:.4f}%")
        
        # Handle extremely imbalanced data
        if y.mean() < 0.01:  # Less than 1% positive class
            logger.warning("Extremely imbalanced dataset detected!")
            # We'll handle this during training with class weights or sampling
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=y  # Maintain class distribution
        )
        
        logger.info(f"Training set: {self.X_train.shape}, {self.y_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}, {self.y_test.shape}")
        
        return X, y
    
    def train_logistic_regression(self):
        """Train and evaluate Logistic Regression"""
        logger.info("\n" + "="*50)
        logger.info("Training Logistic Regression")
        logger.info("="*50)
        
        with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
            # Log parameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("random_state", RANDOM_STATE)
            
            # Create model with class weights for imbalanced data
            model = LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight='balanced',  # Handle class imbalance
                solver='liblinear'
            )
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, "Logistic Regression")
            
            # Log model
            signature = infer_signature(self.X_train, model.predict(self.X_train))
            mlflow.sklearn.log_model(
                model, 
                "logistic_regression_model",
                signature=signature,
                input_example=self.X_train.iloc[:5]
            )
            
            self.models['logistic_regression'] = model
            self.results['logistic_regression'] = metrics
            
            return model, metrics
    
    def train_decision_tree(self):
        """Train and evaluate Decision Tree"""
        logger.info("\n" + "="*50)
        logger.info("Training Decision Tree")
        logger.info("="*50)
        
        with mlflow.start_run(run_name="Decision_Tree"):
            # Log parameters
            mlflow.log_param("model_type", "DecisionTree")
            mlflow.log_param("random_state", RANDOM_STATE)
            
            model = DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight='balanced',
                max_depth=10  # Prevent overfitting
            )
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, "Decision Tree")
            
            # Log model
            mlflow.sklearn.log_model(model, "decision_tree_model")
            
            self.models['decision_tree'] = model
            self.results['decision_tree'] = metrics
            
            return model, metrics
    
    def train_random_forest(self):
        """Train and evaluate Random Forest"""
        logger.info("\n" + "="*50)
        logger.info("Training Random Forest")
        logger.info("="*50)
        
        with mlflow.start_run(run_name="Random_Forest"):
            # Log parameters
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("random_state", RANDOM_STATE)
            
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                class_weight='balanced_subsample',
                n_jobs=-1,
                max_depth=15,
                min_samples_split=5
            )
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, "Random Forest")
            
            # Log model
            mlflow.sklearn.log_model(model, "random_forest_model")
            
            self.models['random_forest'] = model
            self.results['random_forest'] = metrics
            
            return model, metrics
    
    def train_xgboost(self):
        """Train and evaluate XGBoost"""
        logger.info("\n" + "="*50)
        logger.info("Training XGBoost")
        logger.info("="*50)
        
        with mlflow.start_run(run_name="XGBoost"):
            # Log parameters
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param("random_state", RANDOM_STATE)
            
            # Handle imbalanced data with scale_pos_weight
            # Calculate ratio for scale_pos_weight
            neg_count = np.sum(self.y_train == 0)
            pos_count = np.sum(self.y_train == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
            
            model = XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            )
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, "XGBoost")
            
            # Log model
            mlflow.xgboost.log_model(model, "xgboost_model")
            
            self.models['xgboost'] = model
            self.results['xgboost'] = metrics
            
            return model, metrics
    
    def _evaluate_model(self, model, model_name: str) -> dict:
        """
        Evaluate model and return metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
        }
        
        # ROC-AUC if probability predictions available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        # Log metrics to MLflow
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Print results
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        logger.info(f"  Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
        
        # Create and save visualization
        self._save_evaluation_plots(model, model_name, y_pred, y_pred_proba, cm)
        
        return metrics
    
    def _save_evaluation_plots(self, model, model_name: str, y_pred, y_pred_proba, cm):
        """Save evaluation plots"""
        os.makedirs('reports/figures', exist_ok=True)
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = f'reports/figures/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(cm_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
        plt.close()
        
        # 2. ROC Curve (if probabilities available)
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            roc_path = f'reports/figures/roc_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(roc_path, dpi=100, bbox_inches='tight')
            mlflow.log_artifact(roc_path)
            plt.close()
    
    def hyperparameter_tuning(self, model_type: str = 'random_forest'):
        """
        Perform hyperparameter tuning using GridSearchCV
        """
        logger.info("\n" + "="*50)
        logger.info(f"Hyperparameter Tuning for {model_type}")
        logger.info("="*50)
        
        with mlflow.start_run(run_name=f"{model_type}_tuned"):
            # Define parameter grids for different models
            param_grids = {
                'logistic_regression': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
            }
            
            if model_type not in param_grids:
                logger.error(f"No parameter grid defined for {model_type}")
                return None
            
            # Create base model
            if model_type == 'logistic_regression':
                base_model = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000)
            elif model_type == 'random_forest':
                base_model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample')
            elif model_type == 'xgboost':
                base_model = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False)
            else:
                logger.error(f"Model type {model_type} not supported for tuning")
                return None
            
            # Perform Grid Search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grids[model_type],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            logger.info(f"Starting GridSearchCV with {len(param_grids[model_type])} parameter combinations...")
            
            # Fit the grid search
            grid_search.fit(self.X_train, self.y_train)
            
            # Log tuning results
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Evaluate best model
            best_model = grid_search.best_estimator_
            metrics = self._evaluate_model(best_model, f"{model_type}_tuned")
            
            # Log best model
            if model_type == 'xgboost':
                mlflow.xgboost.log_model(best_model, f"{model_type}_tuned_model")
            else:
                mlflow.sklearn.log_model(best_model, f"{model_type}_tuned_model")
            
            # Save tuning results
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': {
                    'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                }
            }
            
            results_path = f'reports/{model_type}_tuning_results.json'
            with open(results_path, 'w') as f:
                json.dump(tuning_results, f, indent=2)
            
            mlflow.log_artifact(results_path)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
            
            self.models[f'{model_type}_tuned'] = best_model
            self.results[f'{model_type}_tuned'] = metrics
            
            return best_model, metrics
    
    def compare_models(self):
        """Compare all trained models and identify the best"""
        logger.info("\n" + "="*50)
        logger.info("Model Comparison")
        logger.info("="*50)
        
        if not self.results:
            logger.error("No models to compare. Train models first.")
            return None
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            row = {'model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by ROC-AUC if available, otherwise by F1-Score
        if 'roc_auc' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
            best_metric = 'roc_auc'
        else:
            comparison_df = comparison_df.sort_values('f1_score', ascending=False)
            best_metric = 'f1_score'
        
        # Identify best model
        best_model_name = comparison_df.iloc[0]['model']
        best_model = self.models.get(best_model_name)
        
        logger.info("\nModel Performance Summary:")
        logger.info(comparison_df.to_string())
        
        logger.info(f"\n✅ Best Model: {best_model_name}")
        logger.info(f"   Best {best_metric}: {comparison_df.iloc[0][best_metric]:.4f}")
        
        # Save comparison
        comparison_path = 'reports/model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        mlflow.log_artifact(comparison_path)
        
        return best_model_name, best_model, comparison_df
    
    def register_best_model(self, model_name: str, model):
        """Register the best model in MLflow Model Registry"""
        logger.info("\n" + "="*50)
        logger.info(f"Registering Best Model: {model_name}")
        logger.info("="*50)
        
        # Register model
        if 'xgboost' in model_name.lower():
            mlflow.xgboost.log_model(model, "registered_model")
        else:
            mlflow.sklearn.log_model(model, "registered_model")
        
        # Get the run ID
        run_id = mlflow.active_run().info.run_id
        
        # Register the model (this creates version 1)
        model_uri = f"runs:/{run_id}/registered_model"
        registered_model = mlflow.register_model(model_uri, "CreditRiskModel")
        
        logger.info(f"✅ Model registered: {registered_model.name}")
        logger.info(f"   Version: {registered_model.version}")
        logger.info(f"   Run ID: {run_id}")
        
        # Transition to Production stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="CreditRiskModel",
            version=registered_model.version,
            stage="Production"
        )
        
        logger.info("✅ Model transitioned to Production stage")
        
        # Save model locally as well
        model_path = 'models/best_model.joblib'
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"✅ Model saved locally: {model_path}")
        
        return registered_model
    
    def save_training_summary(self):
        """Save training summary report"""
        summary = {
            'training_date': datetime.now().isoformat(),
            'dataset': self.data_path,
            'use_customer_level': self.use_customer_level,
            'train_shape': self.X_train.shape if self.X_train is not None else None,
            'test_shape': self.X_test.shape if self.X_test is not None else None,
            'models_trained': list(self.models.keys()),
            'results': self.results,
            'random_state': RANDOM_STATE,
            'test_size': TEST_SIZE
        }
        
        summary_path = 'reports/training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Training summary saved: {summary_path}")
        
        return summary


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("TASK 5: MODEL TRAINING AND TRACKING")
    print("=" * 70)
    
    # Configuration
    DATA_PATH = "data/processed/dataset_with_target_FIXED.parquet"
    USE_CUSTOMER_LEVEL = True  # Set to False for transaction-level
    
    # Create directories
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(DATA_PATH, USE_CUSTOMER_LEVEL)
        
        # 1. Load and prepare data
        trainer.load_and_prepare_data()
        
        # 2. Train models (at least 2 as required)
        print("\n1. Training Baseline Models...")
        trainer.train_logistic_regression()  # Simple, interpretable
        trainer.train_random_forest()        # Ensemble method
        # trainer.train_xgboost()            # Uncomment for third model
        # trainer.train_decision_tree()      # Uncomment for fourth model
        
        # 3. Hyperparameter tuning (optional but recommended)
        print("\n2. Performing Hyperparameter Tuning...")
        trainer.hyperparameter_tuning('random_forest')
        
        # 4. Compare models
        print("\n3. Comparing Models...")
        best_model_name, best_model, comparison_df = trainer.compare_models()
        
        # 5. Register best model
        if best_model:
            print("\n4. Registering Best Model...")
            trainer.register_best_model(best_model_name, best_model)
        
        # 6. Save summary
        print("\n5. Saving Training Summary...")
        trainer.save_training_summary()
        
        print("\n" + "=" * 70)
        print("✅ TASK 5 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext Steps:")
        print("1. Run MLflow UI: mlflow ui")
        print("2. Check reports/ for visualizations and summary")
        print("3. Proceed to Task 6: Model Deployment")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()