import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_ingestion():
    from data.ingestion import generate_sample_data
    df = generate_sample_data(n=100, save=False)
    assert len(df) == 100
    assert "irrigate" in df.columns
    assert df["irrigate"].isin([0, 1]).all()


def test_preprocessing():
    from data.ingestion import generate_sample_data
    from data.preprocessing import preprocess, split_data

    df = generate_sample_data(n=200, save=False)
    X, y = preprocess(df, fit_scaler=True)
    assert X.shape[1] == 4
    assert len(X) == len(y)

    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) > len(X_test)


def test_train_and_predict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)

    from data.ingestion import generate_sample_data
    from data.preprocessing import preprocess, split_data
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    df = generate_sample_data(n=200, save=False)
    X, y = preprocess(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, tmp_path / "data/processed/model.joblib")

    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})
