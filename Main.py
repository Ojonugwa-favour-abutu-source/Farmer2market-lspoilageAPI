from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("tomato_spoilage_model.pkl")

class ShipmentData(BaseModel):
    originState: str
    destinationCity: str
    season: str
    packagingType: str
    transportMode: str
    coldStorageAvailable: str
    storageType: str
    ripenessLevel: str
    initialDamageLevel: str
    routeDistanceKm: float
    quantitySentCrates: int
    hoursSinceHarvestAtDispatch: float

@app.post("/predict-spoilage")
def predict_spoilage(data: ShipmentData):
    sample = pd.DataFrame([data.dict()])
    probability = model.predict_proba(sample)[0][1]
    prediction = "spoiled" if probability >= 0.46 else "notSpoiled"
    return {
        "spoilageProbability": round(float(probability), 4),
        "prediction": prediction
    }
