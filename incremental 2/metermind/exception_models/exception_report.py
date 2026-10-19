import csv
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import constants
from metermind.exception_models.baseline_model import split_readings
from metermind.exception_models.champion_model import predict_with_rule
from metermind.exception_models.feature_engineering import widen_readings
from metermind.exception_models.feature_pipeline import modelling_columns

EXCEPTION_REPORT_FIELDS = ("exception_class", "precision", "recall", "support")


def per_class_scores(truth, predicted):
    precisions, recalls, _, supports = precision_recall_fscore_support(
        truth,
        predicted,
        labels=list(constants.EXCEPTION_CLASSES),
        zero_division=0,
    )
    scores = {}
    for cls_name, p, r, s in zip(
        constants.EXCEPTION_CLASSES, precisions, recalls, supports
    ):
        scores[cls_name] = {
            "precision": float(p),
            "recall": float(r),
            "support": int(s),
        }
    return scores


def confusion_frame(truth, predicted):
    matrix = confusion_matrix(
        truth,
        predicted,
        labels=list(constants.EXCEPTION_CLASSES),
    )
    return pd.DataFrame(
        matrix,
        index=list(constants.EXCEPTION_CLASSES),
        columns=list(constants.EXCEPTION_CLASSES),
    )


def write_exception_report(truth, predicted, where):
    scores = per_class_scores(truth, predicted)
    path = Path(where)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_support = 0
    for cls_name in constants.EXCEPTION_CLASSES:
        info = scores[cls_name]
        p = round(info["precision"], constants.MEASUREMENT_DECIMAL_PLACES)
        r = round(info["recall"], constants.MEASUREMENT_DECIMAL_PLACES)
        s = info["support"]
        total_support += s
        rows.append(
            {
                "exception_class": cls_name,
                "precision": p,
                "recall": r,
                "support": s,
            }
        )

    # Total row
    acc = float(accuracy_score(truth, predicted))
    tot_p = round(acc, constants.MEASUREMENT_DECIMAL_PLACES)
    tot_r = round(acc, constants.MEASUREMENT_DECIMAL_PLACES)
    rows.append(
        {
            "exception_class": "total",
            "precision": tot_p,
            "recall": tot_r,
            "support": total_support,
        }
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXCEPTION_REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return path


if __name__ == "__main__":
    book = pd.read_csv(constants.CLEAN_READING_BOOK_PATH)
    widened_book = widen_readings(book)

    fitting_columns, held_out_columns, fitting_classes, held_out_classes = (
        split_readings(widened_book, columns=modelling_columns(widened=True))
    )

    champion = joblib.load(constants.CHAMPION_MODEL_PATH)
    rule_info = json.loads(
        Path(constants.OPERATING_RULE_PATH).read_text(encoding="utf-8")
    )

    predictions = predict_with_rule(
        champion,
        held_out_columns,
        rule_info["tamper_floor"],
        rule_info["fault_floor"],
    )

    frame = confusion_frame(held_out_classes, predictions)
    print(frame)

    written_path = write_exception_report(
        held_out_classes, predictions, constants.EXCEPTION_REPORT_PATH
    )
    print(f"written: {constants.EXCEPTION_REPORT_PATH}")
