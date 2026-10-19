from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "artifacts" / "incident_priority_model.joblib"


class PredictionService:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Trained model not found at {self.model_path}. Run `python -m ml.train` first."
                )
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        features = [
            "category",
            "service",
            "impact",
            "urgency",
            "affected_users",
            "service_criticality",
            "reported_source",
        ]
        row = pd.DataFrame([{key: payload[key] for key in features}])
        prediction = self.model.predict(row)[0]

        probabilities = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(row)[0]
            probabilities = {
                str(label): round(float(prob), 6)
                for label, prob in zip(self.model.classes_, probs)
            }
            confidence = float(max(probs))
        else:
            confidence = 1.0

        return {
            "predicted_priority": str(prediction),
            "confidence": round(confidence, 6),
            "probabilities": probabilities,
        }


prediction_service = PredictionService()
