import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Chemins des artefacts ──────────────────────────────
BASE      = Path(__file__).resolve().parents[2]
PROCESSED = BASE / "data" / "processed"

# ── Charger les 2 modèles + artefacts ─────────────────
model_classification = joblib.load(PROCESSED / "model_classification.joblib")
model_regression     = joblib.load(PROCESSED / "model_regression.joblib")
scaler               = joblib.load(PROCESSED / "scaler.joblib")
le_soil              = joblib.load(PROCESSED / "le_soil.joblib")
le_crop              = joblib.load(PROCESSED / "le_crop.joblib")

FEATURES = ['temperature', 'humidity', 'soil_moisture',
            'rainfall', 'soil_type_enc', 'crop_type_enc']

DEBIT_POMPE = 10  # litres/minute

def predict(soil_moisture: float, temperature: float, 
            humidity: float, rainfall: float,
            soil_type: str = "Loamy", crop_type: str = "Maize"):

    # Encoder soil_type et crop_type
    soil_enc = int(le_soil.transform([soil_type])[0])
    crop_enc = int(le_crop.transform([crop_type])[0])

    # Créer le dataframe
    X = pd.DataFrame([[temperature, humidity, soil_moisture,
                       rainfall, soil_enc, crop_enc]],
                     columns=FEATURES)

    # Normaliser
    X_scaled = scaler.transform(X)

    # ── Modèle 1 — Classification ──────────────────────
    irrigate    = int(model_classification.predict(X_scaled)[0])
    probability = float(model_classification.predict_proba(X_scaled)[0][1])

    if irrigate == 1:
        # ── Modèle 2 — Régression ──────────────────────
        litres    = float(model_regression.predict(X_scaled)[0])
        litres    = round(max(5, min(80, litres)), 1)
        duree_min = round(litres / DEBIT_POMPE, 2)

        return {
            "irrigate"        : 1,
            "probability"     : round(probability, 4),
            "label"           : "Irrigation Nécessaire",
            "litres_eau"      : litres,
            "duree_pompe_min" : duree_min,
            "statut_pompe"    : "ON"
        }
    else:
        return {
            "irrigate"        : 0,
            "probability"     : round(probability, 4),
            "label"           : "Pas d'irrigation",
            "litres_eau"      : 0,
            "duree_pompe_min" : 0,
            "statut_pompe"    : "OFF"
        }

if __name__ == "__main__":
    # Test direct
    result = predict(
        soil_moisture=25,
        temperature=38,
        humidity=30,
        rainfall=2,
        soil_type="Sandy",
        crop_type="Cotton"
    )
    print(result)