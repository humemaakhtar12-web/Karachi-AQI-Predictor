import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

if API_KEY:
    print("API key loaded successfully!")
    print("Key length:", len(API_KEY))
else:
    print("API key NOT found.")