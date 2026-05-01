# Smart Irrigation System

Binary classification ML system that predicts whether field irrigation is needed based on environmental sensor data.

## Stack

| Layer | Tool |
|-------|------|
| Data versioning | DVC |
| Experiment tracking | MLflow |
| Pipeline orchestration | ZenML |
| API | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |

## Project Structure

```
smart-irrigation/
├── data/
│   ├── raw/                  # Raw CSV data
│   └── processed/            # Scaled data, model artifacts
├── src/
│   ├── data/
│   │   ├── ingestion.py      # Load / generate dataset
│   │   └── preprocessing.py  # Clean + scale
│   ├── models/
│   │   ├── train.py          # Train + MLflow logging
│   │   └── predict.py        # Load model + infer
│   └── pipeline/
│       └── pipeline.py       # ZenML pipeline
├── app/
│   ├── main.py               # FastAPI app
│   └── templates/
│       └── index.html        # Dashboard UI
├── notebooks/
├── tests/
├── Dockerfile
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate data + train model
python src/models/train.py

# Start API + UI
uvicorn app.main:app --reload
# Open http://localhost:8000
```

## Docker

```bash
docker build -t smart-irrigation .
docker run -p 8000:8000 smart-irrigation
```

## DVC — Dataset Versioning

```bash
dvc init
dvc add data/raw/irrigation_data.csv
git add data/raw/irrigation_data.csv.dvc .dvc/
git commit -m "track dataset"
dvc remote add -d myremote s3://your-bucket/path
dvc push
```

To pull data on a fresh clone:
```bash
dvc pull
```

## MLflow UI

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

## ZenML Pipeline

```bash
zenml init
python src/pipeline/pipeline.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard UI |
| POST | `/predict` | Irrigation prediction |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

**POST /predict**
```json
{
  "soil_moisture": 35.0,
  "temperature": 32.0,
  "humidity": 45.0,
  "rainfall": 2.0
}
```

## Features

- `soil_moisture` — 0 to 100 %
- `temperature` — °C
- `humidity` — 0 to 100 %
- `rainfall` — mm/day

## Model

RandomForest classifier with 100 estimators. Achieves ~94% accuracy on held-out test set. All experiments logged to MLflow.
