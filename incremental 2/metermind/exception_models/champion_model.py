import json
from pathlib import Path

import joblib
import pandas as pd

from config import constants
from metermind.exception_models.baseline_model import split_readings
from metermind.exception_models.cross_validation import cross_validate_models
from metermind.exception_models.feature_engineering import widen_readings
from metermind.exception_models.feature_pipeline import (
    EXCEPTION_COLUMN,
    modelling_columns,
)
from metermind.exception_models.tree_models import (
    train_decision_tree,
    train_random_forest,
)

COST_MATRIX = {
    "genuine_high_usage": {
        "genuine_high_usage": 0.0,
        "estimated_read": 1.0,
        "meter_fault": 2.0,
        "tamper_suspected": 10.0,
        "tariff_mismatch": 3.0,
    },
    "estimated_read": {
        "genuine_high_usage": 1.0,
        "estimated_read": 0.0,
        "meter_fault": 1.0,
        "tamper_suspected": 4.0,
        "tariff_mismatch": 2.0,
    },
    "meter_fault": {
        "genuine_high_usage": 2.0,
        "estimated_read": 1.0,
        "meter_fault": 0.0,
        "tamper_suspected": 3.0,
        "tariff_mismatch": 2.0,
    },
    "tamper_suspected": {
        "genuine_high_usage": 5.0,
        "estimated_read": 3.0,
        "meter_fault": 3.0,
        "tamper_suspected": 0.0,
        "tariff_mismatch": 4.0,
    },
    "tariff_mismatch": {
        "genuine_high_usage": 4.0,
        "estimated_read": 2.0,
        "meter_fault": 2.0,
        "tamper_suspected": 5.0,
        "tariff_mismatch": 0.0,
    },
}

TAMPER_FLOORS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8)
FAULT_FLOORS = (0.0, 0.3, 0.4, 0.5, 0.6)
MODELS_THAT_REPORT_CONFIDENCE = ("baseline", "decision_tree", "random_forest")


def choose_champion(fitting_columns, fitting_classes, folds=None):
    scored = cross_validate_models(fitting_columns, fitting_classes, folds=folds)
    considered_means = {}
    for name in MODELS_THAT_REPORT_CONFIDENCE:
        if name in scored:
            scores = scored[name]
            considered_means[name] = round(
                sum(scores) / len(scores), constants.MEASUREMENT_DECIMAL_PLACES
            )

    winning_name = max(considered_means, key=considered_means.get)
    return winning_name, considered_means


def expected_cost(truth, predicted):
    truth_list = list(truth)
    pred_list = list(predicted)
    if len(truth_list) != len(pred_list):
        raise ValueError("Truth and predicted sequences must have equal length.")
    if len(truth_list) == 0:
        return 0.0

    total_cost = sum(
        COST_MATRIX[str(t)][str(p)] for t, p in zip(truth_list, pred_list)
    )
    return float(total_cost / len(truth_list))


def predict_with_rule(model, columns, tamper_floor, fault_floor):
    probs_array = model.predict_proba(columns)
    classes = list(model.classes_)

    results = []
    for probs in probs_array:
        class_probs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        chosen = None
        for cls_name, prob in class_probs:
            if cls_name == "tamper_suspected":
                if prob >= tamper_floor:
                    chosen = cls_name
                    break
            elif cls_name == "meter_fault":
                if prob >= fault_floor:
                    chosen = cls_name
                    break
            else:
                chosen = cls_name
                break

        if chosen is None:
            chosen = class_probs[0][0]
        results.append(str(chosen))

    return results


def choose_operating_rule(model, held_out_columns, held_out_classes):
    best_cost = float("inf")
    best_rule = None

    for tf in TAMPER_FLOORS:
        for ff in FAULT_FLOORS:
            preds = predict_with_rule(model, held_out_columns, tf, ff)
            cost = expected_cost(held_out_classes, preds)
            if cost < best_cost:
                best_cost = cost
                best_rule = {
                    "tamper_floor": tf,
                    "fault_floor": ff,
                    "expected_cost": float(cost),
                }

    return best_rule


def save_champion(model, rule, model_path, rule_path):
    m_path = Path(model_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, m_path)

    r_path = Path(rule_path)
    r_path.parent.mkdir(parents=True, exist_ok=True)
    rounded_rule = {
        k: round(v, constants.MEASUREMENT_DECIMAL_PLACES) if isinstance(v, float) else v
        for k, v in rule.items()
    }
    r_path.write_text(json.dumps(rounded_rule, indent=2), encoding="utf-8")

    return m_path, r_path


if __name__ == "__main__":
    book = pd.read_csv(constants.CLEAN_READING_BOOK_PATH)
    widened_book = widen_readings(book)

    fitting_columns, held_out_columns, fitting_classes, held_out_classes = (
        split_readings(widened_book, columns=modelling_columns(widened=True))
    )

    champion_name, means = choose_champion(fitting_columns, fitting_classes)

    if champion_name == "random_forest":
        model = train_random_forest(fitting_columns, fitting_classes, widened=True)
    elif champion_name == "decision_tree":
        model = train_decision_tree(fitting_columns, fitting_classes, widened=True)
    else:
        from metermind.exception_models.baseline_model import train_baseline_model

        model = train_baseline_model(fitting_columns, fitting_classes, widened=True)

    unaided_preds = model.predict(held_out_columns)
    argmax_cost = expected_cost(held_out_classes, unaided_preds)

    rule = choose_operating_rule(model, held_out_columns, held_out_classes)

    save_champion(
        model,
        rule,
        constants.CHAMPION_MODEL_PATH,
        constants.OPERATING_RULE_PATH,
    )

    print(
        "only the families that report a confidence can carry an operating rule, so the support vector"
    )
    print("is not considered:")
    print(f"chosen: {champion_name} {means}")
    print(f"argmax cost: {argmax_cost:.4f}")
    print(f"rule: {rule}")
