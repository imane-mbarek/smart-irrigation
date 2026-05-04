import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor ,RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import joblib


df=pd.read_csv('data_core.csv')
df = df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type']]
df.columns = ['temperature', 'humidity', 'soil_moisture', 'soil_type', 'crop_type']
df.head()
df['rainfall'] = np.random.uniform(0, 50, len(df))
df.head()
crop_coeff = {
    'Maize': 0.55, 'Sugarcane': 0.65, 'Cotton': 0.60,
    'Tobacco': 0.45, 'Paddy': 0.70, 'Barley': 0.45,
    'Wheat': 0.50, 'Millets': 0.40, 'Oil seeds': 0.42,
    'Pulses': 0.38, 'Ground Nuts': 0.48
}

soil_coeff = {
    'Sandy': 0.20, 'Loamy': 0.00,
    'Black': -0.10, 'Red': 0.05, 'Clayey': -0.15
}

def calc_litres(row):
    coeff = crop_coeff.get(row['crop_type'], 0.50)
    sol = soil_coeff.get(row['soil_type'], 0.00)
    litres = (
        (100 - row['soil_moisture']) * coeff
        + max(0, row['temperature'] - 20) * 0.35
        - row['rainfall'] * 0.60
        + sol * 10
        + np.random.normal(0, 2)
    )
    return round(max(5, min(80, litres)), 1)

def calc_irrigate(row):
    score = 0
    if row['soil_moisture'] < 30: score += 3
    elif row['soil_moisture'] < 50: score += 1
    if row['temperature'] > 35: score += 2
    elif row['temperature'] > 28: score += 1
    if row['rainfall'] < 10: score += 2
    elif row['rainfall'] < 25: score += 1
    if row['humidity'] < 40: score += 1
    if row['crop_type'] in ['Sugarcane', 'Paddy', 'Cotton']: score += 1
    if row['soil_type'] == 'Sandy': score += 1
    score += np.random.normal(0, 0.5)
    return 1 if score >= 4 else 0

df['litres_eau'] = df.apply(calc_litres, axis=1)
df['irrigate'] = df.apply(calc_irrigate, axis=1)

print(f"Distribution irrigate:\n{df['irrigate'].value_counts()}")
df.head()

import subprocess
subprocess.run(['pip', 'install', 'imbalanced-learn'])

# Générer nouvelles lignes synthétiques
crop_types = df['crop_type'].unique()
soil_types = df['soil_type'].unique()

n_new = 4000  # lignes à ajouter
new_data = {
    'temperature': np.random.uniform(20, 45, n_new),
    'humidity': np.random.uniform(30, 90, n_new),
    'soil_moisture': np.random.uniform(10, 80, n_new),
    'rainfall': np.random.uniform(0, 50, n_new),
    'soil_type': np.random.choice(soil_types, n_new),
    'crop_type': np.random.choice(crop_types, n_new),
}

df_new = pd.DataFrame(new_data)
df_new['soil_type_enc'] = le_soil.transform(df_new['soil_type'])
df_new['crop_type_enc'] = le_crop.transform(df_new['crop_type'])
df_new['litres_eau'] = df_new.apply(calc_litres, axis=1)
df_new['irrigate'] = df_new.apply(calc_irrigate, axis=1)

# Combiner avec dataset original
df_combined = pd.concat([df, df_new]).reset_index(drop=True)

# Appliquer SMOTE sur le combined
X_temp2 = df_combined[['temperature', 'humidity', 'soil_moisture', 'rainfall',
                         'soil_type_enc', 'crop_type_enc']]
y_temp2 = df_combined['irrigate']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_temp2, y_temp2)

df_final = pd.DataFrame(X_res, columns=['temperature', 'humidity', 'soil_moisture',
                                         'rainfall', 'soil_type_enc', 'crop_type_enc'])
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
df_final.head()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

X = df_final[['temperature', 'humidity', 'soil_moisture', 
              'rainfall', 'soil_type_enc', 'crop_type_enc']]
y1 = df_final['irrigate']

X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, 
                                                      random_state=42, stratify=y1)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model1 = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model1.fit(X_train_sc, y_train)

y_pred = model1.predict(X_test_sc)
print("=== Modèle 1 — Classification ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, model1.predict_proba(X_test_sc)[:,1]):.4f}")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

y2 = df_final['litres_eau']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2,
                                                          test_size=0.2,
                                                          random_state=42)
X_train2_sc = scaler.transform(X_train2)
X_test2_sc = scaler.transform(X_test2)

model2 = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model2.fit(X_train2_sc, y_train2)

y_pred2 = model2.predict(X_test2_sc)
print("=== Modèle 2 — Régression ===")
print(f"MAE : {mean_absolute_error(y_test2, y_pred2):.4f} litres")
print(f"R²  : {r2_score(y_test2, y_pred2):.4f}")