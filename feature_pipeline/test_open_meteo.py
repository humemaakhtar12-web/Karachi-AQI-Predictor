import requests

LAT = 24.8591
LON = 66.9983

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": "2026-08-20",
    "end_date": "2026-08-20",
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

print("Testing Open-Meteo historical weather access...")

response = requests.get(url, params=params)

print("HTTP Status:", response.status_code)

if response.status_code == 200:
    data = response.json()

    times = data["hourly"]["time"]

    print("Historical Weather API: SUCCESS")
    print("Number of hourly records:", len(times))

    print("\nFirst record:")
    print("time:", times[0])
    print("temperature_2m:", data["hourly"]["temperature_2m"][0])
    print("relative_humidity_2m:", data["hourly"]["relative_humidity_2m"][0])
    print("surface_pressure:", data["hourly"]["surface_pressure"][0])
    print("wind_speed_10m:", data["hourly"]["wind_speed_10m"][0])
    print("wind_direction_10m:", data["hourly"]["wind_direction_10m"][0])
    print("cloud_cover:", data["hourly"]["cloud_cover"][0])

else:
    print("Historical Weather API: FAILED")
    print(response.text)