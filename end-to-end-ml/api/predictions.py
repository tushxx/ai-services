from fastapi import APIRouter

from models.schemas import PredictionRequest, PredictionResponse
from services.prediction_service import prediction_service

router = APIRouter(prefix="/api/v1/predictions", tags=["Predictions"])


@router.post("/priority", response_model=PredictionResponse)
def predict_priority(payload: PredictionRequest):
    return prediction_service.predict(payload.model_dump())
