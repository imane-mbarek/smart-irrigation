FROM python:3.11-slim

# Dossier de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    joblib \
    mlflow \
    fastapi \
    uvicorn \
    pydantic \
    jinja2 \
    imbalanced-learn \
    python-multipart

# Copier le projet
COPY . .

# Entraîner les modèles pendant le build
RUN python src/models/train.py

# Port exposé
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]