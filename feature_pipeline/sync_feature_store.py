import pandas as pd
import mlflow
import os

print("--- Registering Features in MLflow Feature Store ---")

# SQLite backend setup (Fixes MLflow FileStore Exception)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Karachi_AQI_Feature_Store")

df = pd.read_csv("data/processed/karachi_features.csv")

with mlflow.start_run(run_name="hourly_feature_sync"):
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/latest_features.parquet", index=False)
    mlflow.log_artifact("data/processed/latest_features.parquet", artifact_path="feature_store")
    
    mlflow.log_param("num_rows", len(df))
    mlflow.log_param("num_features", len(df.columns))
    
    print("SUCCESS: Features synced to MLflow SQLite Feature Store!")
