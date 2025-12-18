# final_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import os
import json
import joblib
from datetime import datetime

print("=" * 70)
print("TASK 5: FINAL TRAINING WITH DATA LEAKAGE CHECK")
print("=" * 70)

# Create directories
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load and prepare data
print("\n1. Loading and preparing data...")
df = pd.read_parquet('data/processed/dataset_with_target.parquet')
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# 2. Check for data leakage
print("\n2. Checking for data leakage...")

# Check if target column name appears in feature names
target_col = 'is_high_risk'
feature_cols = [col for col in df.columns if col != target_col]

print(f"   Target column: {target_col}")
print(f"   Feature columns: {len(feature_cols)}")

# Look for columns that might contain target information
suspicious_cols = []
for col in feature_cols:
    if target_col.lower() in col.lower() or 'risk' in col.lower() or 'target' in col.lower():
        suspicious_cols.append(col)

if suspicious_cols:
    print(f"   ⚠️  Suspicious columns found: {suspicious_cols}")
    print("   Removing these columns to prevent data leakage...")
    df = df.drop(columns=suspicious_cols)
    feature_cols = [col for col in df.columns if col != target_col]

# 3. Encode categorical variables
print("\n3. Encoding categorical variables...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != 'CustomerId']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"   Encoded {col}: {len(le.classes_)} categories")

# 4. Prepare features and target
print("\n4. Preparing features and target...")
X = df.drop(columns=[target_col, 'CustomerId'] if 'CustomerId' in df.columns else [target_col])
y = df[target_col]

print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")
print(f"   Target distribution: {y.value_counts().to_dict()}")
print(f"   High-risk percentage: {y.mean()*100:.4f}%")

# 5. Split data
print("\n5. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape}")
print(f"   Test: {X_test.shape}")

# 6. Scale features (for Logistic Regression)
print("\n6. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Setup MLflow
print("\n7. Setting up MLflow...")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Credit_Risk_Final")

# 8. Train Logistic Regression
print("\n" + "="*50)
print("8. Training Logistic Regression")
print("="*50)

with mlflow.start_run(run_name="Logistic_Regression_Final"):
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear'
    )
    
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
    
    lr_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_params(lr.get_params())
    for name, value in lr_metrics.items():
        mlflow.log_metric(name, value)
    mlflow.sklearn.log_model(lr, "logistic_regression_model")
    
    print(f"✅ Logistic Regression trained!")
    for name, value in lr_metrics.items():
        print(f"   {name}: {value:.4f}")

# End the run explicitly
mlflow.end_run()

# 9. Train Random Forest
print("\n" + "="*50)
print("9. Training Random Forest")
print("="*50)

with mlflow.start_run(run_name="Random_Forest_Final"):
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1,
        max_depth=15
    )
    
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    rf_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    mlflow.log_param("model", "RandomForest")
    mlflow.log_params(rf.get_params())
    for name, value in rf_metrics.items():
        mlflow.log_metric(name, value)
    mlflow.sklearn.log_model(rf, "random_forest_model")
    
    print(f"✅ Random Forest trained!")
    for name, value in rf_metrics.items():
        print(f"   {name}: {value:.4f}")

# End the run
mlflow.end_run()

# 10. Hyperparameter Tuning
print("\n" + "="*50)
print("10. Hyperparameter Tuning (Grid Search)")
print("="*50)

with mlflow.start_run(run_name="Random_Forest_Tuned"):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced_subsample'),
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    print("Running grid search...")
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    best_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_cv_score': grid_search.best_score_
    }
    
    mlflow.log_param("model", "RandomForest_Tuned")
    mlflow.log_params(grid_search.best_params_)
    for name, value in best_metrics.items():
        mlflow.log_metric(name, value)
    mlflow.sklearn.log_model(best_rf, "random_forest_tuned_model")
    
    print(f"✅ Tuning completed!")
    print(f"   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"   Test ROC-AUC: {best_metrics['roc_auc']:.4f}")

# 11. Save models locally
print("\n" + "="*50)
print("11. Saving Models Locally")
print("="*50)

# Save label encoders
joblib.dump(label_encoders, 'models/label_encoders.joblib')
print(f"✅ Label encoders saved: models/label_encoders.joblib")

# Save scaler
joblib.dump(scaler, 'models/scaler.joblib')
print(f"✅ Scaler saved: models/scaler.joblib")

# Save best model
joblib.dump(best_rf, 'models/best_model.joblib')
print(f"✅ Best model saved: models/best_model.joblib")

# Save feature names
feature_names = list(X.columns)
with open('models/feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print(f"✅ Feature names saved: models/feature_names.json")

# 12. Create final report
print("\n" + "="*50)
print("12. Creating Final Report")
print("="*50)

results = {
    'training_date': datetime.now().isoformat(),
    'dataset_info': {
        'original_shape': (95662, 27),
        'features_shape': X.shape,
        'target_distribution': dict(y.value_counts()),
        'high_risk_percentage': float(y.mean() * 100)
    },
    'model_performance': {
        'logistic_regression': lr_metrics,
        'random_forest': rf_metrics,
        'random_forest_tuned': best_metrics
    },
    'best_model': {
        'name': 'random_forest_tuned',
        'parameters': grid_search.best_params_,
        'roc_auc': best_metrics['roc_auc']
    },
    'files_created': [
        'models/best_model.joblib',
        'models/label_encoders.joblib',
        'models/scaler.joblib',
        'models/feature_names.json'
    ]
}

with open('reports/task5_complete_report.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Complete report saved: reports/task5_complete_report.json")

# 13. Display results
print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)

print(f"\n{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
print("-" * 75)

models = [
    ('Logistic Regression', lr_metrics),
    ('Random Forest', rf_metrics),
    ('Random Forest Tuned', best_metrics)
]

for name, metrics in models:
    print(f"{name:<25} "
          f"{metrics['accuracy']:<10.4f} "
          f"{metrics['precision']:<10.4f} "
          f"{metrics['recall']:<10.4f} "
          f"{metrics['f1_score']:<10.4f} "
          f"{metrics['roc_auc']:<10.4f}")

print("-" * 75)

# Check if scores are realistic (not 1.0000)
if any(metrics['accuracy'] > 0.99 for _, metrics in models):
    print("\n⚠️  WARNING: Models showing near-perfect accuracy.")
    print("   This could indicate data leakage or a very easy classification problem.")
    print("   Check if target information is accidentally included in features.")

print("\n" + "="*70)
print("✅ TASK 5 COMPLETED SUCCESSFULLY!")
print("="*70)

print("\n📁 MODELS CREATED IN 'models/' FOLDER:")
print("   1. best_model.joblib          - Trained Random Forest model")
print("   2. label_encoders.joblib      - Encoders for categorical features")
print("   3. scaler.joblib              - Feature scaler for Logistic Regression")
print("   4. feature_names.json         - List of feature names")

print("\n📊 MLflow Experiments:")
print("   View at: http://127.0.0.1:5000")
print("   Experiment: 'Credit_Risk_Final'")

print("\n🎯 NEXT STEP: Task 6 - Model Deployment")
print("   Your models are ready for API deployment with FastAPI!")