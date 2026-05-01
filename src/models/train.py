import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.ingestion import load_data
from data.preprocessing import preprocess, split_data


MODEL_PATH = Path("data/processed/model.joblib")
MLFLOW_TRACKING_URI = "mlruns"


def train(n_estimators=100, max_depth=6, random_state=42):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("smart-irrigation")

    df = load_data()
    X, y = preprocess(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = split_data(X, y)

    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        })

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print("Training complete.")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return model, metrics


if __name__ == "__main__":
    train()
