import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; }
    .stAlert { margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🌫️ Karachi 3-Day Air Quality Index (AQI) Forecasting")
st.caption("Serverless Machine Learning System for Atmospheric Prediction")

# Load Prediction Data
pred_file = "models/latest_predictions.json"
metrics_file = "models/metrics/model_metrics.json"

if not os.path.exists(pred_file):
    st.error("Prediction file missing. Please run `prediction.py` first.")
    st.stop()

with open(pred_file, "r") as f:
    pred_data = json.load(f)

# Sidebar Info
st.sidebar.header("📍 Location & Status")
st.sidebar.write("**City:** Karachi, Pakistan")
st.sidebar.write(f"**Last Updated:** {pred_data.get('latest_timestamp', 'N/A')}")

curr_aqi = pred_data.get("current_aqi", 3)

st.sidebar.markdown("---")
st.sidebar.header("📊 AQI Scale Guide")
st.sidebar.write("1: Good 🟢")
st.sidebar.write("2: Fair 🟡")
st.sidebar.write("3: Moderate 🟠")
st.sidebar.write("4: Poor 🔴")
st.sidebar.write("5: Very Poor 🟣")

# Metrics Display
col_curr, col24, col48, col72 = st.columns(4)

col_curr.metric("Current AQI", f"Level {curr_aqi}")

p24 = pred_data['predictions']['24h']
p48 = pred_data['predictions']['48h']
p72 = pred_data['predictions']['72h']

col24.metric("24-Hour Forecast", f"Level {p24['aqi']}", delta=p24['category'])
col48.metric("48-Hour Forecast", f"Level {p48['aqi']}", delta=p48['category'])
col72.metric("72-Hour Forecast", f"Level {p72['aqi']}", delta=p72['category'])

st.markdown("---")

# Hazardous AQI Alert System
max_pred = max(p24['aqi'], p48['aqi'], p72['aqi'])
if max_pred >= 4:
    st.error("🚨 **HAZARDOUS AIR QUALITY ALERT:** Unhealthy or Poor Air Quality forecasted over the next 72 hours. Sensitive groups should wear masks and avoid outdoor activity.")
else:
    st.success("✅ **Air Quality Notice:** Forecasted atmospheric conditions are within moderate/safe operational limits.")

# Tabs Layout
tab1, tab2, tab3 = st.tabs(["🔍 SHAP Explainability", "📈 Model Performance Metrics", "📜 Raw Forecast Data"])

with tab1:
    st.subheader("Model Feature Importance (SHAP)")
    st.write("Top environmental & lag factors driving the forecast:")
    
    horizon_choice = st.selectbox("Select Forecast Horizon", ["24h", "48h", "72h"])
    shap_info = pred_data['explainability'][horizon_choice]['top_features']
    
    df_shap = pd.DataFrame(shap_info)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_shap, x="shap_value", y="feature", palette="Blues_r", ax=ax)
    ax.set_title(f"Top Feature Impacts for {horizon_choice} Model")
    ax.set_xlabel("SHAP Value (Impact on Model Decision)")
    ax.set_ylabel("Feature Name")
    st.pyplot(fig)

with tab2:
    st.subheader("Model Evaluation Metrics (PDF Compliant)")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            m_data = json.load(f)
            
        metrics_list = []
        for h in ["24h", "48h", "72h"]:
            test_m = m_data[h]['test']
            metrics_list.append({
                "Horizon": h,
                "Accuracy": test_m['accuracy'],
                "RMSE": test_m['rmse'],
                "MAE": test_m['mae'],
                "R2 Score": test_m['r2']
            })
        st.table(pd.DataFrame(metrics_list))
    else:
        st.write("Metrics JSON not found.")

with tab3:
    st.subheader("Predictions JSON Payload")
    st.json(pred_data)