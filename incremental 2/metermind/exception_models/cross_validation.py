import csv
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import constants
from metermind.exception_models.baseline_model import build_baseline_model
from metermind.exception_models.feature_pipeline import (
    EXCEPTION_COLUMN,
    build_feature_pipeline,
    modelling_columns,
)
from metermind.exception_models.tree_models import (
    build_decision_tree_model,
    build_random_forest_model,
)

CROSS_VALIDATION_FIELDS = ("model", "fold", "accuracy")


def build_support_vector_model(class_weight=None, widened=False):
    return Pipeline(
        steps=[
            ("prepare", build_feature_pipeline(widened=widened)),
            (
                "decide",
                LinearSVC(
                    C=constants.SUPPORT_VECTOR_REGULARISATION,
                    random_state=constants.MODEL_SEED,
                    class_weight=class_weight,
                    max_iter=2000,
                    dual=False,
                ),
            ),
        ]
    )


def every_candidate_model(class_weight=None, widened=False):
    return {
        "baseline": build_baseline_model(class_weight=class_weight, widened=widened),
        "decision_tree": build_decision_tree_model(
            class_weight=class_weight, widened=widened
        ),
        "random_forest": build_random_forest_model(
            class_weight=class_weight, widened=widened
        ),
        "support_vector": build_support_vector_model(
            class_weight=class_weight, widened=widened
        ),
    }


def cross_validate_models(fitting_columns, fitting_classes, folds=None):
    if folds is None:
        folds = constants.CROSS_VALIDATION_FOLDS

    skf = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=constants.MODEL_SEED
    )
    candidates = every_candidate_model()
    results = {name: [] for name in candidates}

    for train_idx, test_idx in skf.split(fitting_columns, fitting_classes):
        X_train, X_test = (
            fitting_columns.iloc[train_idx],
            fitting_columns.iloc[test_idx],
        )
        y_train, y_test = (
            fitting_classes.iloc[train_idx],
            fitting_classes.iloc[test_idx],
        )

        for name, model in candidates.items():
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            preds = fold_model.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            results[name].append(acc)

    return results


def fold_spread(fold_scores):
    return float(max(fold_scores) - min(fold_scores))


def write_cross_validation(scored, where):
    path = Path(where)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CROSS_VALIDATION_FIELDS)
        for model_name, scores in scored.items():
            for fold_num, score in enumerate(scores, 1):
                writer.writerow([model_name, fold_num, score])
    return path


if __name__ == "__main__":
    book = pd.read_csv(constants.CLEAN_READING_BOOK_PATH)
    fitting_columns = book[list(modelling_columns())]
    fitting_classes = book[EXCEPTION_COLUMN]

    scored = cross_validate_models(fitting_columns, fitting_classes)
    write_cross_validation(scored, constants.CROSS_VALIDATION_PATH)

    for name, scores in scored.items():
        mean_score = sum(scores) / len(scores)
        spread = fold_spread(scores)
        print(f"{name:<16} mean {mean_score:.4f}  spread {spread:.4f}")
