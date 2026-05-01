import joblib
import numpy as np
import pandas as pd
from pathlib import Path


MODEL_PATH = Path("data/processed/model.joblib")
SCALER_PATH = Path("data/processed/scaler.joblib")
FEATURES = ["soil_moisture", "temperature", "humidity", "rainfall"]


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict(soil_moisture: float, temperature: float, humidity: float, rainfall: float):
    model, scaler = load_artifacts()

    X = pd.DataFrame([[soil_moisture, temperature, humidity, rainfall]], columns=FEATURES)
    X_scaled = scaler.transform(X)

    pred = int(model.predict(X_scaled)[0])
    prob = float(model.predict_proba(X_scaled)[0][1])

    return {
        "irrigate": pred,
        "probability": round(prob, 4),
        "label": "Irrigation Needed" if pred == 1 else "No Irrigation Needed",
    }


if __name__ == "__main__":
    result = predict(soil_moisture=30, temperature=38, humidity=25, rainfall=2)
    print(result)
