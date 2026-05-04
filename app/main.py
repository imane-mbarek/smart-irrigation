import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Literal
from models.predict import predict

app = FastAPI(title="Smart Irrigation API", version="2.0.0")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

class PredictRequest(BaseModel):
    soil_moisture : float = Field(..., ge=0, le=100, example=35.0)
    temperature   : float = Field(..., ge=0, le=60,  example=32.0)
    humidity      : float = Field(..., ge=0, le=100, example=45.0)
    rainfall      : float = Field(..., ge=0, le=100, example=2.0)
    soil_type     : Literal["Sandy", "Loamy", "Black", "Red", "Clayey"] = "Loamy"
    crop_type     : Literal["Maize", "Sugarcane", "Cotton", "Tobacco",
                            "Paddy", "Barley", "Wheat", "Millets",
                            "Oil seeds", "Pulses", "Ground Nuts"] = "Maize"

class PredictResponse(BaseModel):
    irrigate        : int
    probability     : float
    label           : str
    litres_eau      : float
    duree_pompe_min : float
    statut_pompe    : str

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictResponse)
async def make_prediction(body: PredictRequest):
    result = predict(
        soil_moisture = body.soil_moisture,
        temperature   = body.temperature,
        humidity      = body.humidity,
        rainfall      = body.rainfall,
        soil_type     = body.soil_type,
        crop_type     = body.crop_type,
    )
    return result

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}