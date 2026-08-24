import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Karachi coordinates
LAT = 24.8591
LON = 66.9983

# Weather API
weather_url = "https://api.openweathermap.org/data/2.5/weather"

weather_params = {
    "lat": LAT,
    "lon": LON,
    "appid": API_KEY,
    "units": "metric"
}

# Air Pollution API
pollution_url = "https://api.openweathermap.org/data/2.5/air_pollution"

pollution_params = {
    "lat": LAT,
    "lon": LON,
    "appid": API_KEY
}

# Create raw data folder
os.makedirs("data/raw", exist_ok=True)

# Fetch weather
weather_response = requests.get(weather_url, params=weather_params)

print("Weather API Status:", weather_response.status_code)

if weather_response.status_code == 200:
    weather_data = weather_response.json()
    print("Weather data fetched successfully.")

    with open("data/raw/weather.json", "w") as f:
        json.dump(weather_data, f, indent=4)
else:
    print("Weather API Error:")
    print(weather_response.text)

# Fetch air pollution
pollution_response = requests.get(
    pollution_url,
    params=pollution_params
)

print("Air Pollution API Status:", pollution_response.status_code)

if pollution_response.status_code == 200:
    pollution_data = pollution_response.json()
    print("Air pollution data fetched successfully.")

    with open("data/raw/air_pollution.json", "w") as f:
        json.dump(pollution_data, f, indent=4)
else:
    print("Air Pollution API Error:")
    print(pollution_response.text)

print("Raw data saved in data/raw/")