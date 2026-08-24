import os
import json
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "karachi_features.csv"
)

PREDICTION_FILE = os.path.join(
    BASE_DIR,
    "models",
    "latest_predictions.json"
)


st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="AQI",
    layout="wide"
)


@st.cache_data
def load_features():

    if not os.path.exists(FEATURE_FILE):
        st.error("Feature dataset not found.")
        st.stop()

    df = pd.read_csv(FEATURE_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    return df


@st.cache_data
def load_predictions():

    if not os.path.exists(PREDICTION_FILE):
        st.error("Prediction file not found.")
        st.stop()

    with open(PREDICTION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def aqi_description(aqi):

    descriptions = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }

    return descriptions.get(int(aqi), "Unknown")


df = load_features()
prediction_data = load_predictions()

latest = df.iloc[-1]

latest_timestamp = latest["timestamp"]

current_aqi = int(round(latest["openweather_aqi"]))


# ============================================================
# HEADER
# ============================================================

st.title("Karachi Air Quality Index Predictor")

st.markdown(
    """
    ## AI-Based AQI Forecasting Dashboard

    This system uses historical weather and air pollution
    data to forecast Karachi's AQI for the next 24, 48,
    and 72 hours.
    """
)

st.info(
    "Latest available observation: "
    + latest_timestamp.strftime("%Y-%m-%d %H:%M UTC")
)


# ============================================================
# CURRENT AQI
# ============================================================

st.subheader("Current Available AQI")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("AQI Category", current_aqi)

with col2:
    st.metric("AQI Status", aqi_description(current_aqi))

with col3:
    st.metric("Location", "Karachi")


# ============================================================
# FORECAST
# ============================================================

st.subheader("AQI Forecast")

predictions = prediction_data["predictions"]

col1, col2, col3 = st.columns(3)

with col1:
    aqi_24 = int(predictions["24h"]["aqi_category"])
    st.metric("Next 24 Hours", aqi_24)
    st.write(aqi_description(aqi_24))

with col2:
    aqi_48 = int(predictions["48h"]["aqi_category"])
    st.metric("Next 48 Hours", aqi_48)
    st.write(aqi_description(aqi_48))

with col3:
    aqi_72 = int(predictions["72h"]["aqi_category"])
    st.metric("Next 72 Hours", aqi_72)
    st.write(aqi_description(aqi_72))


# ============================================================
# WEATHER
# ============================================================

st.subheader("Latest Weather Conditions")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Temperature", f"{latest['temperature']:.1f} C")

with col2:
    st.metric("Humidity", f"{latest['humidity']:.0f}%")

with col3:
    st.metric("Pressure", f"{latest['pressure']:.1f} hPa")

with col4:
    st.metric("Wind Speed", f"{latest['wind_speed']:.1f}")

with col5:
    st.metric("Cloud Cover", f"{latest['cloud_cover']:.0f}%")


# ============================================================
# POLLUTANTS
# ============================================================

st.subheader("Latest Pollutant Measurements")

pollutants = {
    "PM2.5": latest["pm2_5"],
    "PM10": latest["pm10"],
    "CO": latest["co"],
    "NO": latest["no"],
    "NO2": latest["no2"],
    "O3": latest["o3"],
    "SO2": latest["so2"],
    "NH3": latest["nh3"]
}

pollutant_df = pd.DataFrame(
    {
        "Pollutant": list(pollutants.keys()),
        "Value": [
            round(float(value), 2)
            for value in pollutants.values()
        ]
    }
)

st.dataframe(
    pollutant_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# AQI TREND - DOWNSAMPLED
# ============================================================

st.subheader("Historical AQI Trend")

# Use daily average instead of all 25,000+ hourly records.
daily_aqi = (
    df.set_index("timestamp")["openweather_aqi"]
    .resample("D")
    .mean()
    .dropna()
)

st.line_chart(
    daily_aqi,
    height=350
)


# ============================================================
# RECENT AQI
# ============================================================

st.subheader("Recent AQI Trend")

recent_aqi = df[
    ["timestamp", "openweather_aqi"]
].tail(168).copy()

recent_aqi = recent_aqi.set_index("timestamp")

st.line_chart(
    recent_aqi["openweather_aqi"],
    height=300
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.subheader("System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Model: Random Forest")

with col2:
    st.write("Forecast Horizons: 24h / 48h / 72h")

with col3:
    st.write("Input Features: 68")


st.caption(
    "AQI categories: 1 = Good, 2 = Fair, 3 = Moderate, "
    "4 = Poor, 5 = Very Poor."
)