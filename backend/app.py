"""
Diabetes Risk Prediction - FastAPI Backend
Run from project root: uvicorn backend.app:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Diabetes Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts once at startup
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_order.pkl")


class PatientData(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0, le=300)
    BloodPressure: float = Field(..., ge=0, le=200)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=900)
    BMI: float = Field(..., ge=0, le=80)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3)
    Age: int = Field(..., ge=1, le=120)


def get_recommendations(data: PatientData, risk_pct: float) -> list[str]:
    tips = []
    if data.Glucose > 140:
        tips.append("Sugar control: monitor and reduce refined carbohydrate intake")
    if data.BMI > 25:
        tips.append("Weight management: aim for gradual, sustainable weight loss")
    if data.BloodPressure > 80:
        tips.append("Blood pressure: reduce sodium intake, monitor regularly")
    if data.Age > 45:
        tips.append("Routine screening: recommended annually given age factor")
    if not tips:
        tips.append("Maintain current healthy lifestyle habits")
    tips.append("Regular exercise: at least 150 minutes/week of moderate activity")
    return tips


@app.get("/")
def root():
    return {"status": "Diabetes Risk Prediction API is running"}


@app.post("/predict")
def predict(data: PatientData):
    input_dict = data.model_dump()
    input_array = pd.DataFrame([[input_dict[feat] for feat in feature_order]], columns=feature_order)
    input_scaled = scaler.transform(input_array)

    prob = float(model.predict_proba(input_scaled)[0][1])
    prediction = int(model.predict(input_scaled)[0])
    risk_pct = round(prob * 100, 1)

    return {
        "risk_percentage": risk_pct,
        "risk_label": "High" if prediction == 1 else "Low",
        "recommendations": get_recommendations(data, risk_pct)
    }