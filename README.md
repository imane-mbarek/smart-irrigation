# Smart Irrigation

Smart Irrigation est un projet MLOps de classification binaire qui predit si une parcelle doit etre irriguee a partir de donnees environnementales :

- humidite du sol ;
- temperature ;
- humidite de l'air ;
- pluie.

Le projet couvre toute la chaine simple d'un systeme machine learning : ingestion des donnees, preprocessing, entrainement, tracking des experiences, sauvegarde des artefacts, API de prediction, interface web, containerisation et CI.

## Architecture du projet

```text
smart-irrigation/
|-- app/
|   |-- main.py                 # API FastAPI et rendu de l'interface HTML
|   `-- templates/
|       `-- index.html          # Interface utilisateur
|-- data/
|   |-- raw/                    # Donnees brutes suivies avec DVC
|   `-- processed/              # Modele et scaler sauvegardes
|-- notebooks/
|   `-- exploration.ipynb       # Exploration des donnees
|-- src/
|   |-- data/
|   |   |-- ingestion.py        # Chargement/generation du dataset
|   |   `-- preprocessing.py    # Nettoyage, scaling et split
|   |-- models/
|   |   |-- train.py            # Entrainement + logging MLflow
|   |   `-- predict.py          # Chargement des artefacts + prediction
|   `-- pipeline/
|       `-- pipeline.py         # Pipeline ZenML
|-- tests/
|   `-- test_pipeline.py        # Tests automatises
|-- .github/workflows/ci.yml    # Pipeline CI GitHub Actions
|-- Dockerfile                  # Image de deploiement
|-- requirements.txt            # Dependances Python
`-- README.md
```

## Frameworks et librairies utilises

| Outil | Role dans le projet | Pourquoi il est utilise |
|---|---|---|
| Pandas | Manipulation des donnees tabulaires | Simple et efficace pour lire, nettoyer et preparer un CSV |
| NumPy | Generation de donnees numeriques | Permet de creer un dataset d'exemple reproductible |
| Scikit-learn | Preprocessing, split, modele ML et metriques | Fournit une implementation fiable de `StandardScaler`, `RandomForestClassifier` et des metriques |
| Joblib | Sauvegarde du modele et du scaler | Format pratique pour serialiser les artefacts scikit-learn |
| MLflow | Tracking des experiences ML | Garde l'historique des parametres, metriques et modeles entraines |
| ZenML | Orchestration du pipeline ML | Structure les etapes ingestion, preprocessing, entrainement et evaluation |
| FastAPI | API de prediction | Framework rapide pour exposer le modele avec validation automatique |
| Pydantic | Validation des entrees/sorties API | Controle les types et les bornes des variables envoyees a `/predict` |
| Uvicorn | Serveur ASGI | Lance l'application FastAPI en local ou dans Docker |
| DVC | Versioning des donnees | Suit le dataset brut sans mettre directement les gros fichiers dans Git |
| Docker | Containerisation | Permet d'executer l'application dans un environnement reproductible |
| GitHub Actions | Integration continue | Installe les dependances, verifie le code et lance l'entrainement automatiquement |

## Implementation des composants

### 1. Gestion et ingestion des donnees

Le fichier `src/data/ingestion.py` contient deux fonctions principales :

- `generate_sample_data()` genere un dataset synthetique avec NumPy ;
- `load_data()` charge `data/raw/irrigation_data.csv` avec Pandas, ou genere un dataset si le fichier n'existe pas.

Pourquoi ce choix :

- le projet peut fonctionner meme si le CSV brut n'est pas present localement ;
- les donnees restent simples a comprendre et a reproduire ;
- Pandas facilite la lecture et la manipulation du dataset.

### 2. Preprocessing

Le fichier `src/data/preprocessing.py` prepare les donnees avant l'entrainement :

- suppression des lignes vides avec `dropna()` ;
- separation des features et de la target ;
- normalisation avec `StandardScaler` de scikit-learn ;
- sauvegarde du scaler dans `data/processed/scaler.joblib` ;
- separation train/test avec `train_test_split`.

Pourquoi ce choix :

- le scaler applique la meme transformation pendant l'entrainement et pendant la prediction ;
- `train_test_split(..., stratify=y)` garde une repartition coherente des classes ;
- sauvegarder le scaler evite d'avoir des predictions incoherentes en production.

### 3. Modele machine learning

Le fichier `src/models/train.py` entraine un `RandomForestClassifier`.

Parametres utilises :

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
)
```

Pourquoi Random Forest :

- performant pour des donnees tabulaires ;
- robuste sur de petits datasets ;
- gere bien les relations non lineaires entre humidite, temperature, pluie et besoin d'irrigation ;
- demande peu de tuning pour obtenir un bon premier modele.

Les metriques calculees sont :

- `accuracy` ;
- `f1` ;
- `roc_auc`.

Le modele final est sauvegarde dans :

```text
data/processed/model.joblib
```

### 4. Tracking avec MLflow

MLflow est configure dans `src/models/train.py` avec :

```python
MLFLOW_TRACKING_URI = "mlruns"
mlflow.set_experiment("smart-irrigation")
```

Pendant chaque entrainement, le code enregistre :

- les parametres du modele ;
- les metriques ;
- le modele scikit-learn.

Pourquoi MLflow :

- comparer plusieurs executions d'entrainement ;
- garder une trace des performances ;
- retrouver quel modele a ete entraine avec quels parametres.

L'interface MLflow peut etre lancee avec :

```bash
mlflow ui --backend-store-uri mlruns
```

Puis ouvrir :

```text
http://localhost:5000
```

### 5. Pipeline ZenML

Le fichier `src/pipeline/pipeline.py` transforme le workflow ML en pipeline structure.

Etapes definies :

- `ingest_data()` : charge les donnees ;
- `preprocess_data()` : prepare les features et la target ;
- `train_model()` : entraine et sauvegarde le modele ;
- `evaluate_model()` : calcule les metriques.

Pourquoi ZenML :

- rendre le pipeline plus clair ;
- separer chaque etape du cycle ML ;
- faciliter une evolution future vers un pipeline MLOps plus complet.

Execution :

```bash
zenml init
python src/pipeline/pipeline.py
```

### 6. API avec FastAPI

Le fichier `app/main.py` expose le modele avec FastAPI.

Endpoints disponibles :

| Methode | Route | Description |
|---|---|---|
| GET | `/` | Interface web |
| POST | `/predict` | Prediction du besoin d'irrigation |
| GET | `/health` | Verification de l'etat de l'API |
| GET | `/docs` | Documentation Swagger automatique |

FastAPI utilise Pydantic pour valider les donnees envoyees :

```python
class PredictRequest(BaseModel):
    soil_moisture: float = Field(..., ge=0, le=100)
    temperature: float = Field(..., ge=0, le=60)
    humidity: float = Field(..., ge=0, le=100)
    rainfall: float = Field(..., ge=0, le=100)
```

Pourquoi FastAPI :

- validation automatique des entrees ;
- generation automatique de la documentation Swagger ;
- integration simple avec Pydantic ;
- tres adapte pour servir un modele ML sous forme d'API REST.

Exemple de requete :

```json
{
  "soil_moisture": 35.0,
  "temperature": 32.0,
  "humidity": 45.0,
  "rainfall": 2.0
}
```

Exemple de reponse :

```json
{
  "irrigate": 1,
  "probability": 0.87,
  "label": "Irrigation Needed"
}
```

### 7. Prediction

Le fichier `src/models/predict.py` charge :

- `data/processed/model.joblib` ;
- `data/processed/scaler.joblib`.

Ensuite il :

1. cree un DataFrame avec les valeurs envoyees ;
2. applique le scaler sauvegarde ;
3. lance `model.predict()` ;
4. calcule la probabilite avec `predict_proba()` ;
5. retourne une reponse lisible par l'API.

Cette separation permet de reutiliser la logique de prediction depuis l'API ou depuis un script Python.

### 8. DVC pour les donnees

DVC est utilise pour versionner le dataset brut sans stocker directement le CSV dans Git.

Le fichier suivi par Git est :

```text
data/raw/irrigation_data.csv.dvc
```

Le fichier CSV reel est ignore par Git grace a :

```text
data/raw/.gitignore
```

Pourquoi DVC :

- Git reste leger ;
- les donnees peuvent etre versionnees ;
- le projet peut etre reproduit avec les bons fichiers de donnees.

Commandes utiles :

```bash
dvc add data/raw/irrigation_data.csv
git add data/raw/irrigation_data.csv.dvc data/raw/.gitignore
git commit -m "Track raw dataset with DVC"
```

Sur une nouvelle machine :

```bash
dvc pull
```

### 9. Docker

Le `Dockerfile` construit une image Python 3.11 :

1. copie `requirements.txt` ;
2. installe les dependances ;
3. copie le projet ;
4. entraine le modele pendant le build ;
5. lance FastAPI avec Uvicorn.

Commandes :

```bash
docker build -t smart-irrigation .
docker run -p 8000:8000 smart-irrigation
```

Pourquoi Docker :

- meme environnement sur toutes les machines ;
- deploiement plus simple ;
- evite les problemes de versions locales de Python ou des dependances.

### 10. GitHub Actions

Le fichier `.github/workflows/ci.yml` configure une pipeline CI.

Actions realisees :

- checkout du code ;
- installation de Python 3.11 ;
- installation des dependances ;
- lint avec Flake8 ;
- execution de l'entrainement.

Pourquoi GitHub Actions :

- verifier automatiquement que le projet s'installe ;
- detecter les erreurs de style ou d'entrainement ;
- securiser les changements avant integration.

Note : le workflow actuel est configure pour la branche `main`. Si le depot utilise `master`, il faut adapter le fichier CI ou renommer la branche.

## Installation locale

```bash
python -m venv env_mlops
env_mlops\Scripts\activate
pip install -r requirements.txt
```

## Entrainer le modele

```bash
python src/models/train.py
```

Cette commande genere ou charge les donnees, applique le preprocessing, entraine le modele, enregistre les metriques dans MLflow et sauvegarde les artefacts dans `data/processed/`.

## Lancer l'application

```bash
uvicorn app.main:app --reload
```

Ouvrir ensuite :

```text
http://localhost:8000
```

La documentation interactive de l'API est disponible ici :

```text
http://localhost:8000/docs
```

## Tests

```bash
pytest
```

## Fichiers ignores par Git

Le fichier `.gitignore` evite de pousser les fichiers locaux ou sensibles :

- environnements virtuels : `env_mlops/`, `.venv/`, `venv/` ;
- secrets : `.env`, cles privees, certificats ;
- caches Python : `__pycache__/`, `.pytest_cache/` ;
- artefacts MLflow : `mlruns/`, `mlartifacts/` ;
- fichiers d'editeurs : `.vscode/`, `.idea/`.

Cela permet de garder le depot propre, leger et plus sur.

## Resume du workflow

```text
Donnees brutes
    -> ingestion.py
    -> preprocessing.py
    -> train.py
    -> MLflow tracking
    -> model.joblib + scaler.joblib
    -> predict.py
    -> FastAPI
    -> Interface web / API REST
```
