import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("models/metrics", exist_ok=True)

df = pd.read_csv("data/processed/karachi_features.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

drop_cols = ['timestamp', 'city', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
feature_cols = [c for c in df.columns if c not in drop_cols]

with open("models/feature_info.json", "w") as f:
    json.dump({"feature_names": feature_cols}, f, indent=4)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

metrics_summary = {}
comparison_summary = {}

horizons = [24, 48, 72]

print("--- Starting Model Training & Experimentation (Random Forest vs XGBoost) ---")

for h in horizons:
    target_col = f'target_aqi_{h}h'
    
    train_mask = train_df[target_col].notna()
    test_mask = test_df[target_col].notna()
    
    X_tr, y_tr = X_train[train_mask], train_df.loc[train_mask, target_col]
    X_te, y_te = X_test[test_mask], test_df.loc[test_mask, target_col]
    
    # 1. Random Forest Training
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_tr, y_tr)
    rf_preds = rf_model.predict(X_te)
    
    rf_rmse = float(np.sqrt(mean_squared_error(y_te, rf_preds)))
    rf_mae = float(mean_absolute_error(y_te, rf_preds))
    rf_r2 = float(r2_score(y_te, rf_preds))
    
    joblib.dump(rf_model, f"models/aqi_model_{h}h.joblib")
    
    # 2. XGBoost Training (Comparison Model)
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_model.fit(X_tr, y_tr)
    xgb_preds = xgb_model.predict(X_te)
    
    xgb_rmse = float(np.sqrt(mean_squared_error(y_te, xgb_preds)))
    xgb_mae = float(mean_absolute_error(y_te, xgb_preds))
    xgb_r2 = float(r2_score(y_te, xgb_preds))
    
    # Primary Metrics
    metrics_summary[f"{h}h"] = {
        "rmse": round(rf_rmse, 3),
        "mae": round(rf_mae, 3),
        "r2": round(rf_r2, 3)
    }
    
    # Model Comparison
    comparison_summary[f"{h}h"] = {
        "RandomForest": {"rmse": round(rf_rmse, 3), "mae": round(rf_mae, 3), "r2": round(rf_r2, 3)},
        "XGBoost": {"rmse": round(xgb_rmse, 3), "mae": round(xgb_mae, 3), "r2": round(xgb_r2, 3)}
    }

with open("models/metrics/model_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=4)

with open("models/metrics/model_comparison.json", "w") as f:
    json.dump(comparison_summary, f, indent=4)

print("SUCCESS: Trained Random Forest & XGBoost models and generated comparison metrics!")
