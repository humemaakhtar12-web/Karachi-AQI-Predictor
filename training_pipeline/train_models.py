import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def load_data():
    file_path = "data/processed/karachi_features.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def get_feature_and_targets(df):
    target_cols = ['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
    ignore_cols = ['timestamp'] + target_cols
    feature_cols = [col for col in df.columns if col not in ignore_cols]
    return feature_cols, target_cols

def chronological_split(df):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    
    # Classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Regression metrics required by PDF
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4)
    }

def train_pipeline():
    print("Loading data...")
    df = load_data()
    feature_cols, target_cols = get_feature_and_targets(df)
    
    train_df, val_df, test_df = chronological_split(df)
    
    os.makedirs("models/metrics", exist_ok=True)
    all_metrics = {}
    
    # Map horizon targets to model filenames
    horizons = {
        "24h": "target_aqi_24h",
        "48h": "target_aqi_48h",
        "72h": "target_aqi_72h"
    }
    
    for label, target_col in horizons.items():
        print(f"\n--- Training {label} AQI Model ---")
        
        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_val, y_val = val_df[feature_cols], val_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]
        
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        
        clf.fit(X_train, y_train)
        
        train_eval = evaluate_model(clf, X_train, y_train)
        val_eval = evaluate_model(clf, X_val, y_val)
        test_eval = evaluate_model(clf, X_test, y_test)
        
        model_path = f"models/aqi_model_{label}.joblib"
        joblib.dump(clf, model_path)
        print(f"Saved model to {model_path}")
        
        all_metrics[label] = {
            "train": train_eval,
            "validation": val_eval,
            "test": test_eval
        }
        
        print(f"Test Accuracy ({label}): {test_eval['accuracy']} | Test RMSE: {test_eval['rmse']} | Test MAE: {test_eval['mae']} | Test R2: {test_eval['r2']}")

    # Save detailed metrics JSON
    with open("models/metrics/model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    # Save feature information metadata
    feature_info = {
        "features": feature_cols,
        "number_of_features": len(feature_cols),
        "targets": target_cols
    }
    with open("models/feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=4)
        
    print("\nTraining completed successfully! Metrics and Feature metadata updated.")

if __name__ == "__main__":
    train_pipeline()