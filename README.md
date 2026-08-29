# 🌬️ Karachi Air Quality Index (AQI) Predictor & 3-Day Forecast System

An end-to-end MLOps pipeline designed to ingest environmental weather data, maintain a local feature store, track machine learning experiments via **MLflow**, and serve predictions through an interactive **Streamlit** dashboard and **API**.

---

## 🏗️ Architecture & MLOps Workflow

1. **Feature Pipeline (`feature_pipeline/`)**: Ingests weather and air quality parameters, processes feature transformations, and updates local feature stores (`data/`).
2. **Training Pipeline (`training_pipeline/`)**: Trains regression models on historical datasets and logs hyperparameter metrics into MLflow (`mlflow.db` & `mlruns/`).
3. **Inference & Prediction (`prediction.py` & `api.py`)**: Generates 24/48/72-hour air quality predictions.
4. **Interactive UI (`app.py`)**: Displays real-time PM2.5 metrics, 3-day forecast trends, and feature importance charts.
5. **Continuous Integration (`.github/workflows/`)**: Automated GitHub Actions workflow verifying pipeline code syntax on pushes to `main`.

---

## 📂 Actual Directory Structure

```text
Karachi-AQI-Predictor/
│
├── .github/workflows/          # CI/CD GitHub Actions workflow
│   └── main.yml
├── dashboard/                  # Dashboard assets and components
├── data/                       # Historical and processed feature data
├── feature_pipeline/           # Automated data ingestion scripts
├── mlruns/                     # MLflow experiment artifacts
├── models/                     # Saved trained model artifacts
├── training_pipeline/          # Model training & MLflow logging pipeline
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── api.py                      # API backend interface
├── app.py                      # Main Streamlit web application
├── mlflow.db                   # SQLite database for MLflow tracking
├── prediction.py               # Batch & real-time prediction engine
├── prediction_backup.py        # Backup prediction logic
└── requirements.txt            # Python dependencies
```

---

## 📈 Model Performance & Evaluation Metrics

The model tracks AQI regression metrics logged inside MLflow:

* **Algorithm**: `RandomForestRegressor`
* **Target Feature**: `pm2_5` (Particulate Matter 2.5 µg/m³)
* **R² Score**: `0.92`
* **RMSE**: `4.97`

---

## 🚀 How to Run Locally

### 1. Environment Setup
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows Git Bash
pip install -r requirements.txt
```

### 2. Run Pipeline & Serving
```bash
# Execute model training
python training_pipeline/train.py

# Launch Streamlit app
streamlit run app.py
```
