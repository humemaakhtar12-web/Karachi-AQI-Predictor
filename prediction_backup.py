import os
import json
import joblib
import pandas as pd


# ============================================================
# AQI PREDICTION PIPELINE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "karachi_features.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


MODEL_FILES = {
    "24h": os.path.join(MODEL_DIR, "aqi_model_24h.joblib"),
    "48h": os.path.join(MODEL_DIR, "aqi_model_48h.joblib"),
    "72h": os.path.join(MODEL_DIR, "aqi_model_72h.joblib"),
}


TARGETS = {
    "24h": "target_aqi_24h",
    "48h": "target_aqi_48h",
    "72h": "target_aqi_72h",
}


def load_data():
    print("Loading feature dataset...")

    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_FILE}"
        )

    df = pd.read_csv(FEATURE_FILE)

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    print(f"Dataset shape: {df.shape}")

    return df


def load_models():
    print("\nLoading trained models...")

    models = {}

    for horizon, path in MODEL_FILES.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        models[horizon] = joblib.load(path)

        print(f"{horizon} model loaded successfully.")

    return models


def get_feature_columns(df):
    """
    Use the same feature definition used during training.
    Target columns and timestamp are excluded.
    """

    target_columns = list(TARGETS.values())

    excluded = target_columns + ["timestamp"]

    feature_columns = [
        column for column in df.columns
        if column not in excluded
    ]

    return feature_columns


def validate_features(df, feature_columns):
    print("\nValidating prediction features...")

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    X = df[feature_columns].copy()

    missing_values = X.isna().sum().sum()

    if missing_values > 0:
        raise ValueError(
            f"Prediction features contain {missing_values} missing values."
        )

    print(f"Number of features: {len(feature_columns)}")
    print("Missing feature values: 0")

    return X


def make_predictions(df, models, feature_columns):

    X = validate_features(df, feature_columns)

    # Latest available observation
    latest_index = len(df) - 1

    latest_timestamp = df.iloc[latest_index]["timestamp"]

    latest_row = X.iloc[[latest_index]]

    print("\n" + "=" * 60)
    print("LATEST OBSERVATION")
    print("=" * 60)

    print(f"Timestamp: {latest_timestamp}")

    if "openweather_aqi" in df.columns:
        print(
            f"Current AQI category: "
            f"{int(df.iloc[latest_index]['openweather_aqi'])}"
        )

    predictions = {}

    print("\n" + "=" * 60)
    print("AQI FORECAST")
    print("=" * 60)

    for horizon in ["24h", "48h", "72h"]:

        model = models[horizon]

        prediction = model.predict(latest_row)[0]

        prediction = int(round(float(prediction)))

        # Keep AQI category within valid range
        prediction = max(1, min(5, prediction))

        predictions[horizon] = prediction

        print(
            f"{horizon} AQI Prediction: {prediction}"
        )

    return latest_timestamp, predictions


def get_aqi_category(aqi):

    categories = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }

    return categories.get(aqi, "Unknown")


def save_predictions(timestamp, predictions):

    output_file = os.path.join(
        MODEL_DIR,
        "latest_predictions.json"
    )

    result = {
        "location": "Karachi",
        "generated_from_timestamp": timestamp,
        "predictions": {
            "24h": {
                "aqi_category": predictions["24h"],
                "description": get_aqi_category(
                    predictions["24h"]
                )
            },
            "48h": {
                "aqi_category": predictions["48h"],
                "description": get_aqi_category(
                    predictions["48h"]
                )
            },
            "72h": {
                "aqi_category": predictions["72h"],
                "description": get_aqi_category(
                    predictions["72h"]
                )
            }
        }
    }

    with open(output_file, "w") as file:
        json.dump(result, file, indent=4)

    print("\nPrediction results saved:")
    print(output_file)


def main():

    print("=" * 60)
    print("AQI PREDICTION PIPELINE")
    print("=" * 60)

    df = load_data()

    models = load_models()

    feature_columns = get_feature_columns(df)

    timestamp, predictions = make_predictions(
        df,
        models,
        feature_columns
    )

    save_predictions(
        timestamp,
        predictions
    )

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()

