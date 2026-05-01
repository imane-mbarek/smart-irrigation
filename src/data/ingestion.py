import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/irrigation_data.csv")
PROCESSED_DATA_PATH = Path("data/processed/")


def generate_sample_data(n=500, save=True):
    import numpy as np

    np.random.seed(42)
    soil_moisture = np.random.uniform(10, 90, n)
    temperature = np.random.uniform(15, 45, n)
    humidity = np.random.uniform(20, 95, n)
    rainfall = np.random.uniform(0, 30, n)

    irrigate = (
        (soil_moisture < 40) |
        ((temperature > 35) & (humidity < 40)) |
        ((rainfall < 5) & (soil_moisture < 55))
    ).astype(int)

    df = pd.DataFrame({
        "soil_moisture": soil_moisture,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "irrigate": irrigate,
    })

    if save:
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"Saved {n} samples to {RAW_DATA_PATH}")

    return df


def load_data(path=RAW_DATA_PATH):
    if not Path(path).exists():
        print("Raw data not found — generating sample dataset.")
        return generate_sample_data()
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(f"Shape: {df.shape} | Irrigate rate: {df['irrigate'].mean():.2%}")
