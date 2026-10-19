import json
from pathlib import Path

import joblib
import pandas as pd

from config import constants
from metermind.exception_models.champion_model import predict_with_rule
from metermind.exception_models.feature_engineering import widen_readings
from metermind.exception_models.feature_pipeline import modelling_columns

QUEUES = constants.EXCEPTION_CLASSES
QUEUE_BY_DECISION = constants.QUEUE_FILE_PATHS


def sort_into_queues(book, decided):
    df = book.copy()
    df["exception_class"] = list(decided)

    sorted_out = {}
    for queue_name in QUEUES:
        matching = df[df["exception_class"] == queue_name].copy()
        matching = matching.reset_index(drop=True)
        sorted_out[queue_name] = matching

    return sorted_out


def queue_counts(sorted_out):
    return {queue_name: len(df) for queue_name, df in sorted_out.items()}


def write_queue_files(sorted_out, paths=None):
    if paths is None:
        paths = constants.QUEUE_FILE_PATHS

    written_paths = {}
    for queue_name, df in sorted_out.items():
        if queue_name in paths:
            file_path = Path(paths[queue_name])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(file_path, index=False)
            written_paths[queue_name] = file_path

    return written_paths


if __name__ == "__main__":
    predict_book = pd.read_csv(constants.READING_BOOK_PREDICT_PATH)
    widened_predict = widen_readings(predict_book)

    champion_model = joblib.load(constants.CHAMPION_MODEL_PATH)
    rule_info = json.loads(Path(constants.OPERATING_RULE_PATH).read_text(encoding="utf-8"))

    decided = predict_with_rule(
        champion_model,
        widened_predict[list(modelling_columns(widened=True))],
        rule_info["tamper_floor"],
        rule_info["fault_floor"],
    )

    sorted_out = sort_into_queues(predict_book, decided)
    write_queue_files(sorted_out)

    print(queue_counts(sorted_out))
