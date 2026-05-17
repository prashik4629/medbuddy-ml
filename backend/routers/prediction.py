from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.core.security import verify_api_key
from backend.services.predictor import predict


router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"],
)


class HeartDiseaseInput(BaseModel):
    age:      int   = Field(..., ge=1,   le=120, example=52)
    sex:      int   = Field(..., ge=0,   le=1,   example=1)
    cp:       int   = Field(..., ge=0,   le=3,   example=0)
    trestbps: float = Field(..., ge=80,  le=250, example=125)
    chol:     float = Field(..., ge=100, le=600, example=212)
    fbs:      int   = Field(..., ge=0,   le=1,   example=0)
    restecg:  int   = Field(..., ge=0,   le=2,   example=1)
    thalach:  float = Field(..., ge=60,  le=250, example=168)
    exang:    int   = Field(..., ge=0,   le=1,   example=0)
    oldpeak:  float = Field(..., ge=0.0, le=10.0,example=1.0)
    slope:    int   = Field(..., ge=0,   le=2,   example=2)
    ca:       int   = Field(..., ge=0,   le=4,   example=0)
    thal:     int   = Field(..., ge=0,   le=3,   example=2)


class HeartDiseaseOutput(BaseModel):
    prediction:  int
    probability: float
    diagnosis:   str
    risk_level:  str


@router.post(
    "/predict",
    response_model=HeartDiseaseOutput,
    summary="Predict heart disease risk",
)
async def predict_heart_disease(
    input_data: HeartDiseaseInput,
    api_key: str = Depends(verify_api_key),
):
    result = predict(input_data.model_dump())

    prediction  = result["prediction"]
    probability = result["probability"]

    if probability >= 0.75:
        risk_level = "High Risk"
    elif probability >= 0.45:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    return HeartDiseaseOutput(
        prediction=prediction,
        probability=probability,
        diagnosis=(
            "Heart Disease Detected"
            if prediction == 1
            else "No Heart Disease Detected"
        ),
        risk_level=risk_level,
    )