import os
import json
import joblib
import pandas as pd
import numpy as np
import shap

def run_prediction_pipeline():
    print("Loading latest features and feature metadata...")
    features_path = "data/processed/karachi_features.csv"
    feature_info_path = "models/feature_info.json"
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature dataset missing: {features_path}")
    if not os.path.exists(feature_info_path):
        raise FileNotFoundError(f"Feature info metadata missing: {feature_info_path}")
        
    df = pd.read_csv(features_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    with open(feature_info_path, 'r') as f:
        feature_info = json.load(f)
    
    feature_cols = feature_info['features']
    
    # Latest single observation for prediction
    latest_row = df.iloc[-1]
    latest_timestamp = str(latest_row['timestamp'])
    X_latest = pd.DataFrame([latest_row[feature_cols]])
    
    # Load models
    models = {
        "24h": joblib.load("models/aqi_model_24h.joblib"),
        "48h": joblib.load("models/aqi_model_48h.joblib"),
        "72h": joblib.load("models/aqi_model_72h.joblib")
    }
    
    predictions = {}
    shap_summaries = {}
    
    # Background sample for TreeExplainer (using last 100 rows for speed and stability)
    background_data = df[feature_cols].tail(100)
    
    for label, model in models.items():
        # Predict AQI
        pred_val = int(model.predict(X_latest)[0])
        predictions[f"aqi_{label}"] = pred_val
        
        # Calculate SHAP values
        print(f"Calculating SHAP explainability for {label} model...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_latest)
        
        # Handle binary vs multi-class output format in SHAP
        if isinstance(shap_values, list):
            # For multi-class, pick the array corresponding to the predicted class
            class_idx = list(model.classes_).index(pred_val) if pred_val in model.classes_ else 0
            vals = shap_values[class_idx][0]
        elif len(np.array(shap_values).shape) == 3:
            class_idx = list(model.classes_).index(pred_val) if pred_val in model.classes_ else 0
            vals = shap_values[0, :, class_idx]
        else:
            vals = shap_values[0]
            
        # Top 5 most influential features for this prediction
        feature_impact = dict(zip(feature_cols, vals))
        sorted_impact = sorted(feature_impact.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        shap_summaries[label] = {
            "top_features": [
                {"feature": feat, "shap_value": round(float(val), 4)}
                for feat, val in sorted_impact
            ]
        }

    # AQI Category Definition helper
    def get_aqi_category(val):
        categories = {
            1: "Good",
            2: "Fair",
            3: "Moderate",
            4: "Poor",
            5: "Very Poor"
        }
        return categories.get(val, "Unknown")

    output_data = {
        "latest_timestamp": latest_timestamp,
        "current_aqi": int(latest_row['openweather_aqi']),
        "predictions": {
            "24h": {
                "aqi": predictions["aqi_24h"],
                "category": get_aqi_category(predictions["aqi_24h"])
            },
            "48h": {
                "aqi": predictions["aqi_48h"],
                "category": get_aqi_category(predictions["aqi_48h"])
            },
            "72h": {
                "aqi": predictions["aqi_72h"],
                "category": get_aqi_category(predictions["aqi_72h"])
            }
        },
        "explainability": shap_summaries
    }
    
    output_file = "models/latest_predictions.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nPrediction pipeline complete! Results saved to {output_file}")
    print(f"Latest Predictions: 24h -> AQI {predictions['aqi_24h']}, 48h -> AQI {predictions['aqi_48h']}, 72h -> AQI {predictions['aqi_72h']}")

if __name__ == "__main__":
    run_prediction_pipeline()