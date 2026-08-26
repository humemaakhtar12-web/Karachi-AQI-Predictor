from fastapi import FastAPI, HTTPException
import json
import os
import pandas as pd

app = FastAPI(
    title="Karachi AQI Prediction API",
    description="REST API endpoints for multi-horizon AQI forecasts and metrics",
    version="1.0"
)

PRED_FILE = "models/latest_predictions.json"
METRICS_FILE = "models/metrics/model_metrics.json"

@app.get("/")
def home():
    return {"message": "Karachi AQI Prediction API is running successfully."}

@app.get("/predict")
def get_predictions():
    """Fetch 24h, 48h, and 72h AQI Predictions"""
    if not os.path.exists(PRED_FILE):
        raise HTTPException(status_code=444, detail="Predictions file not found. Run pipeline first.")
    
    with open(PRED_FILE, "r") as f:
        data = json.load(f)
    return data

@app.get("/metrics")
def get_metrics():
    """Fetch model evaluation metrics (RMSE, MAE, R2, Accuracy)"""
    if not os.path.exists(METRICS_FILE):
        raise HTTPException(status_code=444, detail="Metrics file not found.")
    
    with open(METRICS_FILE, "r") as f:
        data = json.load(f)
    return data