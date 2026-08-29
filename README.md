# 🌬️ Karachi Air Quality Index (AQI) Predictor & 3-Day Forecast

An End-to-End MLOps pipeline designed to ingest weather and atmospheric data, maintain a feature store, train model registries via MLflow, and serve batch predictions through an interactive Streamlit dashboard.

---

## 🛠️ Architecture & Tech Stack

* **Language & Environment**: Python 3.13, Virtual Environment (`venv`)
* **MLOps Framework**: MLflow (SQLite Backend for Feature Tracking & Model Registry)
* **Machine Learning**: Scikit-Learn (`RandomForestRegressor`)
* **Dashboard / UI**: Streamlit with Interactive Plotly & Native Visualizations
* **Feature Processing**: Pandas, NumPy, Parquet / CSV Data Storage

---

## 📈 Model Performance Metrics

The model continuously tracks AQI regression metrics logged inside MLflow:

* **R² Score (Accuracy)**: `0.92`
* **Root Mean Squared Error (RMSE)**: `4.97`
* **Target Feature**: `pm2_5` (Particulate Matter 2.5 µg/m³)

---

## 🚀 How to Run Locally

### 1. Setup Virtual Environment & Dependencies
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows Git Bash
pip install -r requirements.txt