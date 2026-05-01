import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zenml import pipeline, step
from zenml.logger import get_logger
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data.ingestion import load_data
from data.preprocessing import preprocess, split_data
from models.train import train as run_training

logger = get_logger(__name__)


@step
def ingest_data() -> pd.DataFrame:
    df = load_data()
    logger.info(f"Loaded {len(df)} records.")
    return df


@step
def preprocess_data(df: pd.DataFrame):
    X, y = preprocess(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test


@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    from pathlib import Path

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "data/processed/model.joblib")
    logger.info("Model trained and saved.")
    return model


@step
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
    }
    logger.info(f"Evaluation: {metrics}")
    return metrics


@pipeline
def irrigation_pipeline():
    df = ingest_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    irrigation_pipeline()
