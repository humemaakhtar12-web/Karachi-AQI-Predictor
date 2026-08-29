# 🌬️ Karachi Air Quality Index (AQI) Predictor & 3-Day Forecast System

An production-grade, end-to-end MLOps pipeline designed to ingest environmental data, manage an operational feature store, track multi-stage machine learning experiments via **MLflow**, register model artifacts, and serve real-time batch predictions through an interactive **Streamlit** web application.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Architecture & MLOps Workflow](#-architecture--mlops-workflow)
3. [Project Directory Structure](#-project-directory-structure)
4. [Tech Stack & Dependencies](#-tech-stack--dependencies)
5. [Model Performance & Evaluation Metrics](#-model-performance--evaluation-metrics)
6. [Step-by-Step Local Setup & Execution Guide](#-step-by-step-local-setup--execution-guide)
7. [Dashboard & UI Capabilities](#-dashboard--ui-capabilities)

---

## 🌟 Project Overview

Air pollution is a major environmental concern, particularly in metropolitan areas like Karachi. This project establishes an automated, robust machine learning pipeline to forecast particulate matter ($PM_{2.5}$) levels. By adhering to strict MLOps principles, this system ensures reproducibility, version control for data and models, and seamless deployment.

---

## 🏗️ Architecture & MLOps Workflow

The pipeline is structured into distinct, decoupled stages:
1. **Feature Pipeline (`feature_pipeline/`)**: Ingests raw meteorological and air quality parameters, processes feature engineering, and syncs clean datasets into the local feature store (`parquet` / CSV format).
2. **Training Pipeline (`training_pipeline/`)**: Automatically queries features, handles target column detection (`pm2_5`), trains regression models, and tracks experiments using **MLflow** with a local SQLite metadata store (`mlflow.db`).
3. **Model Registry & Artifacts**: Best-performing models are serialized and version-controlled inside the MLflow Model Registry (`models/` and `mlruns/`).
4. **Inference & Serving (`app.py`)**: Loads registered models from the registry/storage to compute real-time estimations and plot continuous 72-hour forecasting curves via Streamlit.

---

## 📂 Project Directory Structure

```text
Karachi-AQI-Predictor/
│
├── .github/workflows/          # CI/CD automation configuration files
├── data/                       # Processed feature stores and historical datasets
│   └── processed/              # Latest sync parquet features
├── feature_pipeline/           # Automated data ingestion & feature store sync
│   └── sync_feature_store.py
├── training_pipeline/          # Model training, hyperparameter & experiment tracking
│   └── train.py
├── models/                     # Serialized model joblib/skops binaries & comparison metrics
│   └── metrics/                # JSON performance logs
├── mlruns/ & mlflow.db         # MLflow tracking server backend (SQLite + Artifact store)
├── app.py                      # Interactive Streamlit Web Application (Frontend)
├── api.py                      # Backend prediction handling interface
├── requirements.txt            # Python dependencies specification
└── README.md                   # Comprehensive project documentation