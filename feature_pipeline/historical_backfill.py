import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
# ============================================================

LAT = 24.8591
LON = 66.9983

# 3 YEARS OF DATA
START_DATE = "2023-08-21"
END_DATE = "2026-08-20"

WEATHER_RAW_PATH = "data/raw/karachi/historical_weather.csv"
POLLUTION_RAW_PATH = "data/raw/karachi/historical_pollution.csv"
COMBINED_PATH = "data/processed/karachi_combined_historical.csv"

os.makedirs("data/raw/karachi", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENWEATHER_API_KEY not found in .env file."
    )


# ============================================================
# WEATHER
# ============================================================

def fetch_weather():

    print("\nFetching historical weather...")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover"
        ],
        "timezone": "UTC"
    }

    response = requests.get(
        url,
        params=params,
        timeout=120
    )

    print("Weather API Status:", response.status_code)

    response.raise_for_status()

    data = response.json()
    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "wind_speed": hourly["wind_speed_10m"],
        "wind_direction": hourly["wind_direction_10m"],
        "cloud_cover": hourly["cloud_cover"]
    })

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.drop_duplicates(
        subset=["timestamp"]
    ).sort_values("timestamp")

    df.to_csv(
        WEATHER_RAW_PATH,
        index=False
    )

    print("Weather records:", len(df))
    print("Weather data saved.")

    return df


# ============================================================
# POLLUTION REQUEST WITH RETRY
# ============================================================

def request_pollution(start, end, max_retries=5):

    url = (
        "https://api.openweathermap.org/data/2.5/"
        "air_pollution/history"
    )

    params = {
        "lat": LAT,
        "lon": LON,
        "start": int(start.timestamp()),
        "end": int(end.timestamp()),
        "appid": API_KEY
    }

    for attempt in range(1, max_retries + 1):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=120
            )

            print(
                f"  Attempt {attempt}/{max_retries} "
                f"Status: {response.status_code}"
            )

            if response.status_code == 200:
                return response.json()

            # Temporary API errors
            if response.status_code in [429, 500, 502, 503, 504]:

                wait_time = attempt * 10

                print(
                    f"  Temporary API error. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)
                continue

            # Permanent/API-key error
            print("  API response:")
            print(response.text)

            return None

        except requests.exceptions.RequestException as e:

            print(
                f"  Network error on attempt "
                f"{attempt}/{max_retries}: {e}"
            )

            if attempt < max_retries:

                wait_time = attempt * 10

                print(
                    f"  Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    "  Maximum retries reached for this chunk."
                )

    return None


# ============================================================
# POLLUTION
# ============================================================

def fetch_pollution():

    print("\nFetching historical air pollution...")

    start = pd.Timestamp(
        START_DATE,
        tz="UTC"
    )

    end = (
        pd.Timestamp(
            END_DATE,
            tz="UTC"
        )
        + pd.Timedelta(days=1)
    )

    all_records = []

    current = start

    while current < end:

        # Keep requests relatively small
        request_end = min(
            current + pd.Timedelta(days=4),
            end
        )

        print(
            f"\nPollution request: "
            f"{current.date()} to "
            f"{request_end.date()}"
        )

        data = request_pollution(
            current,
            request_end
        )

        if data is None:

            print(
                "  WARNING: This chunk could not be fetched."
            )

            # Move forward instead of crashing
            current = request_end
            continue

        records = data.get("list", [])

        print(
            "  Records received:",
            len(records)
        )

        for item in records:

            components = item.get(
                "components",
                {}
            )

            all_records.append({
                "timestamp": pd.to_datetime(
                    item["dt"],
                    unit="s",
                    utc=True
                ),

                "openweather_aqi":
                    item["main"]["aqi"],

                "pm2_5":
                    components.get("pm2_5"),

                "pm10":
                    components.get("pm10"),

                "co":
                    components.get("co"),

                "no":
                    components.get("no"),

                "no2":
                    components.get("no2"),

                "o3":
                    components.get("o3"),

                "so2":
                    components.get("so2"),

                "nh3":
                    components.get("nh3")
            })

        # SAVE PROGRESS AFTER EVERY SUCCESSFUL CHUNK
        progress_df = pd.DataFrame(all_records)

        if not progress_df.empty:

            progress_df = (
                progress_df
                .drop_duplicates(
                    subset=["timestamp"]
                )
                .sort_values("timestamp")
            )

            progress_df.to_csv(
                POLLUTION_RAW_PATH,
                index=False
            )

            print(
                "  Progress saved:",
                len(progress_df),
                "records"
            )

        current = request_end

    pollution_df = pd.DataFrame(all_records)

    if pollution_df.empty:

        raise ValueError(
            "No historical pollution data was collected."
        )

    pollution_df = (
        pollution_df
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
    )

    pollution_df.to_csv(
        POLLUTION_RAW_PATH,
        index=False
    )

    print(
        "\nTotal pollution records:",
        len(pollution_df)
    )

    return pollution_df


# ============================================================
# COMBINE
# ============================================================

def combine_data(weather_df, pollution_df):

    print("\nCombining weather and pollution data...")

    combined = pd.merge(
        weather_df,
        pollution_df,
        on="timestamp",
        how="inner"
    )

    combined = (
        combined
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
    )

    print(
        "Combined records:",
        len(combined)
    )

    return combined


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("Karachi 3-Year Historical Data Backfill")
    print("========================================")

    print("\nStart Date:", START_DATE)
    print("End Date:  ", END_DATE)

    # Weather
    weather_df = fetch_weather()

    # Pollution
    pollution_df = fetch_pollution()

    # Combine
    combined_df = combine_data(
        weather_df,
        pollution_df
    )

    print("\n========================================")
    print("DATA QUALITY CHECK")
    print("========================================")

    print("\nTotal combined records:")
    print(len(combined_df))

    print("\nTotal columns:")
    print(len(combined_df.columns))

    print("\nMissing values:")
    print(combined_df.isnull().sum())

    print("\nDuplicate timestamps:")
    print(
        combined_df["timestamp"]
        .duplicated()
        .sum()
    )

    print("\nDate range:")
    print(
        combined_df["timestamp"].min(),
        "to",
        combined_df["timestamp"].max()
    )

    # Save final dataset
    combined_df.to_csv(
        COMBINED_PATH,
        index=False
    )

    print("\n========================================")
    print("FILES SAVED SUCCESSFULLY")
    print("========================================")

    print("\nWeather:")
    print(WEATHER_RAW_PATH)

    print("\nPollution:")
    print(POLLUTION_RAW_PATH)

    print("\nCombined:")
    print(COMBINED_PATH)

    print("\nFirst 5 records:")
    print(combined_df.head())

    print("\nLast 5 records:")
    print(combined_df.tail())


if __name__ == "__main__":
    main()