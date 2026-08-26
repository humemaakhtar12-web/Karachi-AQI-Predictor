import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

print("--- Starting MLflow Model Training Pipeline ---")

# 1. Connect to MLflow SQLite Database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Karachi_AQI_Model_Training")

# 2. Fetch Data
df = pd.read_csv("data/processed/karachi_features.csv")

# Auto-detect target column (AQI or aqi or pm2_5)
target_col = None
for col in ["aqi", "AQI", "pm2_5", "pm25"]:
    if col in df.columns:
        target_col = col
        break

if not target_col:
    # Pick last numerical column if target name differs
    target_col = df.select_dtypes(include=[np.number]).columns[-1]

print(f"Target Column Detected: {target_col}")

# Select numeric feature columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

X = df[numeric_cols]
y = df[target_col]

# Simple Train-Test Split (Last 20% for testing)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 3. Train Model & Register in MLflow
with mlflow.start_run(run_name="RandomForest_AQI_Model"):
    n_estimators = 100
    max_depth = 10
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions & Metrics
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Log Hyperparameters & Metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    
    # Save Model Artifact Locally and to MLflow Registry
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="Karachi_AQI_Model")
    
    print(f"Metrics -> RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")
    print("SUCCESS: AQI Model Trained and Registered in MLflow Model Registry!")
