import requests
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.8591
LON = 66.9983

end = datetime.now(timezone.utc)
start = end - timedelta(days=1)

url = "https://api.openweathermap.org/data/2.5/air_pollution/history"

params = {
    "lat": LAT,
    "lon": LON,
    "start": int(start.timestamp()),
    "end": int(end.timestamp()),
    "appid": API_KEY
}

print("Testing historical air pollution access...")
print("Start:", start)
print("End:", end)

response = requests.get(url, params=params)

print("HTTP Status:", response.status_code)

if response.status_code == 200:
    data = response.json()

    print("Historical API access: SUCCESS")
    print("Number of records:", len(data.get("list", [])))

    if data.get("list"):
        print("First historical record:")
        print(data["list"][0])
else:
    print("Historical API access: FAILED")
    print("Response:")
    print(response.text)