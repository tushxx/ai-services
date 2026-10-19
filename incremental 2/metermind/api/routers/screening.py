import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from config import constants
from metermind.api import schemas
from metermind.api.access_control import require_signed_in
from metermind.exception_models.champion_model import predict_with_rule
from metermind.exception_models.feature_engineering import widen_readings
from metermind.exception_models.feature_pipeline import modelling_columns

router = APIRouter()

champion_model_path = constants.CHAMPION_MODEL_PATH
operating_rule_path = constants.OPERATING_RULE_PATH
loaded_champion = {}


def load_champion_once():
    if "model" not in loaded_champion or "rule" not in loaded_champion:
        m_path = Path(champion_model_path)
        r_path = Path(operating_rule_path)
        if not m_path.is_file() or not r_path.is_file():
            raise HTTPException(
                status_code=503,
                detail="Champion model or operating rule not available",
            )
        try:
            model = joblib.load(m_path)
            rule = json.loads(r_path.read_text(encoding="utf-8"))
            loaded_champion["model"] = model
            loaded_champion["rule"] = rule
        except Exception as err:
            raise HTTPException(
                status_code=503,
                detail=f"Error loading champion model or operating rule: {err}",
            )
    return loaded_champion["model"], loaded_champion["rule"]


@router.post("/screening", response_model=schemas.ScreeningReply)
def screen_reading(
    submitted_reading: schemas.ScreeningRequest,
    carried=Depends(require_signed_in),
):
    model, rule = load_champion_once()
    data_dict = submitted_reading.model_dump()
    df = pd.DataFrame([data_dict])
    widened_df = widen_readings(df)
    features = widened_df[list(modelling_columns(widened=True))]

    tamper_floor = rule.get("tamper_floor", 0.0)
    fault_floor = rule.get("fault_floor", 0.0)
    decided_classes = predict_with_rule(model, features, tamper_floor, fault_floor)
    decided_class = decided_classes[0]

    probs = model.predict_proba(features)[0]
    classes_list = list(model.classes_)
    class_idx = classes_list.index(decided_class)
    confidence = round(
        float(probs[class_idx]), constants.MEASUREMENT_DECIMAL_PLACES
    )

    queue_path = constants.QUEUE_FILE_PATHS.get(decided_class, "")
    goes_to = Path(queue_path).name if queue_path else ""

    return schemas.ScreeningReply(
        account_number=submitted_reading.account_number,
        billing_period=submitted_reading.billing_period,
        exception_class=decided_class,
        confidence=confidence,
        goes_to=goes_to,
    )
