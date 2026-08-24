import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/processed/karachi_cleaned.csv"
OUTPUT_PATH = "data/processed/karachi_features.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df = df.sort_values("timestamp").reset_index(drop=True)

print("Original shape:", df.shape)


# ============================================================
# 1. TIME FEATURES
# ============================================================

print("\nCreating time features...")

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["day_of_month"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_year"] = df["timestamp"].dt.dayofyear
df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

# Weekend indicator
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ============================================================
# 2. CYCLICAL TIME FEATURES
# ============================================================

print("Creating cyclical time features...")

# Hour
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

# Day of week
df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

# Month
df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)


# ============================================================
# 3. AQI LAG FEATURES
# ============================================================

print("Creating AQI lag features...")

# Previous hourly AQI
df["aqi_lag_1h"] = (
    df["openweather_aqi"].shift(1)
)

df["aqi_lag_3h"] = (
    df["openweather_aqi"].shift(3)
)

df["aqi_lag_6h"] = (
    df["openweather_aqi"].shift(6)
)

df["aqi_lag_12h"] = (
    df["openweather_aqi"].shift(12)
)

df["aqi_lag_24h"] = (
    df["openweather_aqi"].shift(24)
)

df["aqi_lag_48h"] = (
    df["openweather_aqi"].shift(48)
)

df["aqi_lag_72h"] = (
    df["openweather_aqi"].shift(72)
)


# ============================================================
# 4. POLLUTANT LAG FEATURES
# ============================================================

print("Creating pollutant lag features...")

pollutants = [
    "pm2_5",
    "pm10",
    "co",
    "no2",
    "o3",
    "so2",
    "nh3"
]

for column in pollutants:

    df[f"{column}_lag_24h"] = (
        df[column].shift(24)
    )

    df[f"{column}_lag_72h"] = (
        df[column].shift(72)
    )


# ============================================================
# 5. ROLLING AQI FEATURES
# ============================================================

print("Creating rolling AQI features...")

df["aqi_rolling_mean_6h"] = (
    df["openweather_aqi"]
    .shift(1)
    .rolling(6)
    .mean()
)

df["aqi_rolling_mean_12h"] = (
    df["openweather_aqi"]
    .shift(1)
    .rolling(12)
    .mean()
)

df["aqi_rolling_mean_24h"] = (
    df["openweather_aqi"]
    .shift(1)
    .rolling(24)
    .mean()
)

df["aqi_rolling_mean_72h"] = (
    df["openweather_aqi"]
    .shift(1)
    .rolling(72)
    .mean()
)

df["aqi_rolling_std_24h"] = (
    df["openweather_aqi"]
    .shift(1)
    .rolling(24)
    .std()
)


# ============================================================
# 6. POLLUTANT ROLLING FEATURES
# ============================================================

print("Creating pollutant rolling features...")

for column in [
    "pm2_5",
    "pm10",
    "co",
    "no2",
    "o3"
]:

    df[f"{column}_rolling_mean_24h"] = (
        df[column]
        .shift(1)
        .rolling(24)
        .mean()
    )


# ============================================================
# 7. WEATHER CHANGE FEATURES
# ============================================================

print("Creating weather change features...")

df["temperature_change_24h"] = (
    df["temperature"]
    - df["temperature"].shift(24)
)

df["humidity_change_24h"] = (
    df["humidity"]
    - df["humidity"].shift(24)
)

df["pressure_change_24h"] = (
    df["pressure"]
    - df["pressure"].shift(24)
)

df["wind_speed_change_24h"] = (
    df["wind_speed"]
    - df["wind_speed"].shift(24)
)


# ============================================================
# 8. POLLUTION CHANGE FEATURES
# ============================================================

print("Creating pollution change features...")

df["pm2_5_change_24h"] = (
    df["pm2_5"]
    - df["pm2_5"].shift(24)
)

df["pm10_change_24h"] = (
    df["pm10"]
    - df["pm10"].shift(24)
)

df["co_change_24h"] = (
    df["co"]
    - df["co"].shift(24)
)

df["no2_change_24h"] = (
    df["no2"]
    - df["no2"].shift(24)
)

df["o3_change_24h"] = (
    df["o3"]
    - df["o3"].shift(24)
)


# ============================================================
# 9. FUTURE AQI TARGETS
# ============================================================

print("Creating future AQI targets...")

# IMPORTANT:
# These are future values and are used ONLY as targets.
# They must NOT be used as input features.

df["target_aqi_24h"] = (
    df["openweather_aqi"].shift(-24)
)

df["target_aqi_48h"] = (
    df["openweather_aqi"].shift(-48)
)

df["target_aqi_72h"] = (
    df["openweather_aqi"].shift(-72)
)


# ============================================================
# 10. REMOVE ROWS WITH UNAVAILABLE HISTORY/TARGET
# ============================================================

print("\nRemoving rows without sufficient history/targets...")

before = len(df)

df = df.dropna().reset_index(drop=True)

after = len(df)

print("Rows before:", before)
print("Rows after:", after)
print("Rows removed:", before - after)


# ============================================================
# 11. FINAL VALIDATION
# ============================================================

print("\n========================================")
print("FEATURE ENGINEERING VALIDATION")
print("========================================")

print("\nFinal shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nTarget distributions:")

print("\n24-hour target:")
print(
    df["target_aqi_24h"]
    .value_counts()
    .sort_index()
)

print("\n48-hour target:")
print(
    df["target_aqi_48h"]
    .value_counts()
    .sort_index()
)

print("\n72-hour target:")
print(
    df["target_aqi_72h"]
    .value_counts()
    .sort_index()
)

print("\nNumber of features:")
print(len(df.columns))


# ============================================================
# 12. SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("FEATURE DATASET SAVED")
print("========================================")

print(OUTPUT_PATH)

print("\nFirst 5 rows:")
print(df.head())