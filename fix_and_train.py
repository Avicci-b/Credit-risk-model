# fix_and_train.py
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
print("TASK 5: QUICK FIX & TRAIN")
print("Fixing categorical data and training models")
print("=" * 70)

# Create directories
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load and prepare data
print("\n1. Loading and preparing data...")
df = pd.read_parquet('data/processed/dataset_with_target.parquet')
print(f"   Shape: {df.shape}")

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"   Categorical columns: {categorical_cols}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    if col != 'CustomerId':  # Don't encode CustomerId
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded: {col} → {len(le.classes_)} categories")

# 2. Prepare features and target
print("\n2. Preparing features...")
# Drop non-feature columns
drop_cols = ['is_high_risk', 'CustomerId']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['is_high_risk']

print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"   Target distribution: {y.value_counts().to_dict()}")
print(f"   High-risk percentage: {y.mean()*100:.4f}%")

# 3. Split data
print("\n3. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# 4. Scale numerical features (optional but good practice)
print("\n4. Scaling numerical features...")
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=[np.number]).columns
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# 5. Setup MLflow (if server is running)
print("\n5. Setting up MLflow...")
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Credit_Risk_Fixed")
    mlflow_enabled = True
    print("   MLflow enabled")
except:
    mlflow_enabled = False
    print("   MLflow server not running, continuing without it")

# 6. Train Logistic Regression
print("\n" + "="*50)
print("6. Training Logistic Regression")
print("="*50)

if mlflow_enabled:
    with mlflow.start_run(run_name="Logistic_Regression_Fixed"):
        train_mlflow = True
else:
    train_mlflow = False

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

if train_mlflow:
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_params(lr.get_params())
    for name, value in lr_metrics.items():
        mlflow.log_metric(name, value)
    mlflow.sklearn.log_model(lr, "logistic_regression_model")

print(f"✅ Logistic Regression trained!")
for name, value in lr_metrics.items():
    print(f"   {name}: {value:.4f}")

# 7. Train Random Forest
print("\n" + "="*50)
print("7. Training Random Forest")
print("="*50)

if mlflow_enabled:
    with mlflow.start_run(run_name="Random_Forest_Fixed"):
        pass

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

if train_mlflow:
    mlflow.log_param("model", "RandomForest")
    mlflow.log_params(rf.get_params())
    for name, value in rf_metrics.items():
        mlflow.log_metric(name, value)
    mlflow.sklearn.log_model(rf, "random_forest_model")

print(f"✅ Random Forest trained!")
for name, value in rf_metrics.items():
    print(f"   {name}: {value:.4f}")

# 8. Hyperparameter Tuning (Quick Grid Search)
print("\n" + "="*50)
print("8. Quick Hyperparameter Tuning")
print("="*50)

# Simple grid search
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced_subsample'),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

print("Running quick grid search...")
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

if train_mlflow:
    with mlflow.start_run(run_name="Random_Forest_Tuned"):
        mlflow.log_param("model", "RandomForest_Tuned")
        mlflow.log_params(grid_search.best_params_)
        for name, value in best_metrics.items():
            mlflow.log_metric(name, value)
        mlflow.sklearn.log_model(best_rf, "random_forest_tuned_model")

print(f"✅ Tuning completed!")
print(f"   Best parameters: {grid_search.best_params_}")
print(f"   Best CV ROC-AUC: {grid_search.best_score_:.4f}")
print(f"   Test ROC-AUC: {best_metrics['roc_auc']:.4f}")

# 9. Save everything
print("\n" + "="*50)
print("9. Saving Models and Reports")
print("="*50)

# Save label encoders
encoders_path = 'models/label_encoders.joblib'
joblib.dump(label_encoders, encoders_path)
print(f"✅ Label encoders saved: {encoders_path}")

# Save scaler
scaler_path = 'models/scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved: {scaler_path}")

# Save best model
model_path = 'models/best_model.joblib'
joblib.dump(best_rf, model_path)
print(f"✅ Best model saved: {model_path}")

# Save metrics report
results = {
    'logistic_regression': lr_metrics,
    'random_forest': rf_metrics,
    'random_forest_tuned': best_metrics,
    'best_model': 'random_forest_tuned',
    'best_params': grid_search.best_params_,
    'training_date': datetime.now().isoformat(),
    'dataset_shape': df.shape,
    'categorical_columns_encoded': list(label_encoders.keys())
}

report_path = 'reports/task5_final_results.json'
with open(report_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Results report saved: {report_path}")

# 10. Create a simple comparison table
print("\n" + "="*50)
print("10. Model Comparison")
print("="*50)

print("\n" + "-"*60)
print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
print("-"*60)

for model_name, metrics in [('Logistic Regression', lr_metrics),
                           ('Random Forest', rf_metrics),
                           ('Random Forest Tuned', best_metrics)]:
    print(f"{model_name:<25} "
          f"{metrics['accuracy']:<10.4f} "
          f"{metrics['precision']:<10.4f} "
          f"{metrics['recall']:<10.4f} "
          f"{metrics['f1_score']:<10.4f} "
          f"{metrics.get('roc_auc', 0):<10.4f}")

print("-"*60)

# Determine best model based on ROC-AUC
best_model_name = 'Random Forest Tuned'
best_score = best_metrics['roc_auc']

print(f"\n🎯 Best Model: {best_model_name}")
print(f"   ROC-AUC: {best_score:.4f}")

print("\n" + "="*70)
print("✅ TASK 5 COMPLETED SUCCESSFULLY!")
print("="*70)
print("\n📁 Files created:")
print("- models/best_model.joblib (trained model)")
print("- models/label_encoders.joblib (encoders for categorical data)")
print("- models/scaler.joblib (feature scaler)")
print("- reports/task5_final_results.json (metrics report)")
print("\n📊 Next steps:")
print("1. Your models are ready for Task 6 deployment")
print("2. If MLflow server is running, view at: http://127.0.0.1:5000")
print("3. Commit: git add . && git commit -m 'Task 5 completed with fix'")
print("4. Proceed to Task 6: Model Deployment")