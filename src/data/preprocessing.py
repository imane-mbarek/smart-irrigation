import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROCESSED_DIR = Path("data/processed")
SCALER_PATH = PROCESSED_DIR / "scaler.joblib"
FEATURES = ["soil_moisture", "temperature", "humidity", "rainfall"]
TARGET = "irrigate"


def preprocess(df: pd.DataFrame, fit_scaler=True):
    df = df.dropna().copy()

    X = df[FEATURES]
    y = df[TARGET]

    scaler = StandardScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
    return X_scaled, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def load_scaler():
    return joblib.load(SCALER_PATH)


if __name__ == "__main__":
    from ingestion import load_data

    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
