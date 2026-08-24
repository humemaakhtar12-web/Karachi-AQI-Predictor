import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/karachi_combined_historical.csv"
OUTPUT_PATH = "data/processed/karachi_cleaned.csv"

# Load dataset
df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)

# ---------------------------------------------------------
# 1. Convert timestamp
# ---------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df = df.sort_values("timestamp").reset_index(drop=True)

# ---------------------------------------------------------
# 2. Replace -9999 sentinel values with NaN
# ---------------------------------------------------------

pollutant_columns = [
    "pm2_5",
    "pm10",
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "nh3"
]

df[pollutant_columns] = df[pollutant_columns].replace(
    -9999,
    np.nan
)

print("\nMissing values after replacing -9999:")
print(df[pollutant_columns].isnull().sum())

# ---------------------------------------------------------
# 3. Time-based interpolation for missing pollutants
# ---------------------------------------------------------

df = df.set_index("timestamp")

df[pollutant_columns] = (
    df[pollutant_columns]
    .interpolate(method="time")
)

# Safety fallback
df[pollutant_columns] = (
    df[pollutant_columns]
    .ffill()
    .bfill()
)

df = df.reset_index()

# ---------------------------------------------------------
# 4. Normalize AQI
# ---------------------------------------------------------

# OpenWeather AQI should represent categories 1–5.
# Convert fractional/invalid category values to nearest
# valid category.

df["openweather_aqi"] = (
    df["openweather_aqi"]
    .round()
    .clip(1, 5)
    .astype(int)
)

# ---------------------------------------------------------
# 5. Final validation
# ---------------------------------------------------------

print("\nFinal shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nAQI distribution:")
print(
    df["openweather_aqi"]
    .value_counts()
    .sort_index()
)

print("\nAQI unique values:")
print(
    sorted(df["openweather_aqi"].unique())
)

print("\nDuplicate timestamps:")
print(
    df["timestamp"].duplicated().sum()
)

# ---------------------------------------------------------
# 6. Save cleaned dataset
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nCleaned dataset saved successfully:")
print(OUTPUT_PATH)