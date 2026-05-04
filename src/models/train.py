import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("smart-irrigation-v2")

np.random.seed(42)

# Charger
df = pd.read_csv("data/raw/data_core.csv")
df = df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type']]
df.columns = ['temperature', 'humidity', 'soil_moisture', 'soil_type', 'crop_type']
df['rainfall'] = np.random.uniform(0, 50, len(df))

crop_coeff = {'Maize': 0.55, 'Sugarcane': 0.65, 'Cotton': 0.60, 'Tobacco': 0.45, 'Paddy': 0.70, 'Barley': 0.45, 'Wheat': 0.50, 'Millets': 0.40, 'Oil seeds': 0.42, 'Pulses': 0.38, 'Ground Nuts': 0.48}
soil_coeff = {'Sandy': 0.20, 'Loamy': 0.00, 'Black': -0.10, 'Red': 0.05, 'Clayey': -0.15}

def calc_litres(row):
    coeff = crop_coeff.get(row['crop_type'], 0.50)
    sol = soil_coeff.get(row['soil_type'], 0.00)
    litres = (100 - row['soil_moisture']) * coeff + max(0, row['temperature'] - 20) * 0.35 - row['rainfall'] * 0.60 + sol * 10 + np.random.normal(0, 2)
    return round(max(5, min(80, litres)), 1)

def calc_irrigate(row):
    score = 0
    if row['soil_moisture'] < 40: score += 3
    elif row['soil_moisture'] < 60: score += 1
    if row['temperature'] > 30: score += 2
    elif row['temperature'] > 25: score += 1
    if row['rainfall'] < 15: score += 2
    elif row['rainfall'] < 30: score += 1
    if row['humidity'] < 50: score += 1
    if row['crop_type'] in ['Sugarcane', 'Paddy', 'Cotton']: score += 1
    if row['soil_type'] == 'Sandy': score += 1
    score += np.random.normal(0, 0.5)
    return 1 if score >= 4 else 0

df['litres_eau'] = df.apply(calc_litres, axis=1)
df['irrigate'] = df.apply(calc_irrigate, axis=1)

# Encoder
le_soil = LabelEncoder()
le_crop = LabelEncoder()
df['soil_type_enc'] = le_soil.fit_transform(df['soil_type'])
df['crop_type_enc'] = le_crop.fit_transform(df['crop_type'])

# Nouvelles lignes
crop_types = df['crop_type'].unique()
soil_types = df['soil_type'].unique()
n_new = 4000
new_data = pd.DataFrame({'temperature': np.random.uniform(20, 45, n_new), 'humidity': np.random.uniform(30, 90, n_new), 'soil_moisture': np.random.uniform(10, 80, n_new), 'rainfall': np.random.uniform(0, 50, n_new), 'soil_type': np.random.choice(soil_types, n_new), 'crop_type': np.random.choice(crop_types, n_new)})
new_data['soil_type_enc'] = le_soil.transform(new_data['soil_type'])
new_data['crop_type_enc'] = le_crop.transform(new_data['crop_type'])
new_data['litres_eau'] = new_data.apply(calc_litres, axis=1)
new_data['irrigate'] = new_data.apply(calc_irrigate, axis=1)

df_combined = pd.concat([df, new_data]).reset_index(drop=True)

# SMOTE
X_temp = df_combined[['temperature', 'humidity', 'soil_moisture', 'rainfall', 'soil_type_enc', 'crop_type_enc']]
y_temp = df_combined['irrigate']
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_temp, y_temp)

df_final = pd.DataFrame(X_res, columns=['temperature', 'humidity', 'soil_moisture', 'rainfall', 'soil_type_enc', 'crop_type_enc'])
df_final['irrigate'] = y_res
df_final['litres_eau'] = df_final.apply(
    lambda r: round(max(5, min(80,
        (100 - r['soil_moisture']) * 0.50
        + max(0, r['temperature'] - 20) * 0.35
        - r['rainfall'] * 0.60
        + np.random.normal(0, 2)
    )), 1), axis=1
)

print(f"Dataset final : {len(df_final)} lignes")
print(f"Distribution:\n{df_final['irrigate'].value_counts()}")

X = df_final[['temperature', 'humidity', 'soil_moisture', 'rainfall', 'soil_type_enc', 'crop_type_enc']]
scaler = StandardScaler()

# Modèle 1
y1 = df_final['irrigate']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size=0.2, random_state=42, stratify=y1)
X_train1_sc = scaler.fit_transform(X_train1)
X_test1_sc = scaler.transform(X_test1)

with mlflow.start_run(run_name="model1_classification"):
    mlflow.log_params({"n_estimators": 100, "max_depth": 6})
    model1 = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model1.fit(X_train1_sc, y_train1)
    y_pred1 = model1.predict(X_test1_sc)
    acc = accuracy_score(y_test1, y_pred1)
    f1 = f1_score(y_test1, y_pred1)
    auc = roc_auc_score(y_test1, model1.predict_proba(X_test1_sc)[:,1])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model1, name="model_classification")
    print(f"\n=== Modèle 1 — Classification ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

# Modèle 2
y2 = df_final['litres_eau']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.2, random_state=42)
X_train2_sc = scaler.transform(X_train2)
X_test2_sc = scaler.transform(X_test2)

with mlflow.start_run(run_name="model2_regression"):
    mlflow.log_params({"n_estimators": 100, "max_depth": 6})
    model2 = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model2.fit(X_train2_sc, y_train2)
    y_pred2 = model2.predict(X_test2_sc)
    mae = mean_absolute_error(y_test2, y_pred2)
    r2 = r2_score(y_test2, y_pred2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model2, name="model_regression")
    print(f"\n=== Modèle 2 — Régression ===")
    print(f"MAE : {mae:.4f} litres")
    print(f"R²  : {r2:.4f}")

# Sauvegarder
Path("data/processed").mkdir(parents=True, exist_ok=True)
joblib.dump(model1, "data/processed/model_classification.joblib")
joblib.dump(model2, "data/processed/model_regression.joblib")
joblib.dump(scaler, "data/processed/scaler.joblib")
joblib.dump(le_soil, "data/processed/le_soil.joblib")
joblib.dump(le_crop, "data/processed/le_crop.joblib")
print("\nTous les artefacts sauvegardés !")