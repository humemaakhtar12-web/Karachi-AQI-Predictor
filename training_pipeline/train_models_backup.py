import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# AQI PREDICTION MODEL TRAINING PIPELINE
# ============================================================

DATA_PATH = "data/processed/karachi_features.csv"
MODEL_DIR = "models"
METRICS_DIR = "models/metrics"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

TARGETS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]

# Columns that must NOT be used as model inputs
EXCLUDE_COLUMNS = [
    "timestamp",
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("AQI MODEL TRAINING")
print("=" * 60)

print("\nLoading feature dataset...")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Sort chronologically — essential for time-series forecasting
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)

missing = df.isna().sum().sum()
duplicates = df["timestamp"].duplicated().sum()

print(f"Missing values: {missing}")
print(f"Duplicate timestamps: {duplicates}")

if missing > 0:
    raise ValueError("Dataset contains missing values.")

if duplicates > 0:
    raise ValueError("Dataset contains duplicate timestamps.")


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in EXCLUDE_COLUMNS
]

X = df[feature_columns]

print(f"\nNumber of input features: {len(feature_columns)}")

print("\nFirst few features:")
print(feature_columns[:10])


# ============================================================
# 4. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================
#
# 70% -> Training
# 15% -> Validation
# 15% -> Testing
#
# No random shuffling because this is time-series data.
# ============================================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
validation_df = df.iloc[train_end:validation_end].copy()
test_df = df.iloc[validation_end:].copy()

print("\n" + "=" * 60)
print("CHRONOLOGICAL DATA SPLIT")
print("=" * 60)

print(f"Total records:      {len(df)}")
print(f"Training records:   {len(train_df)}")
print(f"Validation records: {len(validation_df)}")
print(f"Testing records:    {len(test_df)}")

print("\nTraining period:")
print(train_df["timestamp"].min(), "->", train_df["timestamp"].max())

print("\nValidation period:")
print(validation_df["timestamp"].min(), "->", validation_df["timestamp"].max())

print("\nTesting period:")
print(test_df["timestamp"].min(), "->", test_df["timestamp"].max())


# ============================================================
# 5. TRAIN ONE MODEL FOR EACH FORECAST HORIZON
# ============================================================

all_metrics = {}

for target in TARGETS:

    print("\n\n" + "=" * 60)
    print(f"TRAINING MODEL: {target}")
    print("=" * 60)

    # Training data
    X_train = train_df[feature_columns]
    y_train = train_df[target].astype(int)

    # Validation data
    X_validation = validation_df[feature_columns]
    y_validation = validation_df[target].astype(int)

    # Test data
    X_test = test_df[feature_columns]
    y_test = test_df[target].astype(int)

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    print("Training completed.")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_predictions = model.predict(X_validation)

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="weighted",
        zero_division=0,
    )

    print("\nValidation Results:")
    print(f"Accuracy: {validation_accuracy:.4f}")
    print(f"Weighted F1: {validation_f1:.4f}")

    # --------------------------------------------------------
    # Final Test Evaluation
    # --------------------------------------------------------

    test_predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    precision = precision_score(
        y_test,
        test_predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        test_predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=[1, 2, 3, 4, 5],
    )

    print("\n" + "-" * 50)
    print("TEST RESULTS")
    print("-" * 50)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[1, 2, 3, 4, 5],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"aqi_model_{target.replace('target_aqi_', '')}.joblib"
    )

    joblib.dump(model, model_path)

    print(f"\nModel saved:")
    print(model_path)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    all_metrics[target] = {
        "validation_accuracy": float(validation_accuracy),
        "validation_f1": float(validation_f1),
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "training_records": int(len(train_df)),
        "validation_records": int(len(validation_df)),
        "testing_records": int(len(test_df)),
        "number_of_features": int(len(feature_columns)),
    }


# ============================================================
# 6. SAVE FEATURE INFORMATION
# ============================================================

feature_info = {
    "features": feature_columns,
    "number_of_features": len(feature_columns),
    "targets": TARGETS,
}

with open(
    os.path.join(MODEL_DIR, "feature_info.json"),
    "w",
) as file:
    json.dump(feature_info, file, indent=4)


# ============================================================
# 7. SAVE ALL METRICS
# ============================================================

metrics_path = os.path.join(
    METRICS_DIR,
    "model_metrics.json"
)

with open(metrics_path, "w") as file:
    json.dump(all_metrics, file, indent=4)


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

for target, metrics in all_metrics.items():

    print(f"\n{target}")
    print(
        f"Test Accuracy: {metrics['test_accuracy']:.4f}"
    )
    print(
        f"Test F1 Score: {metrics['test_f1']:.4f}"
    )

print("\nModels saved in:")
print(MODEL_DIR)

print("\nMetrics saved in:")
print(metrics_path)

print("\nFeature information saved in:")
print(os.path.join(MODEL_DIR, "feature_info.json"))

print("\nAll training tasks completed successfully.")
