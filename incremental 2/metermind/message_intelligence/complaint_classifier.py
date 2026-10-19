import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from config import constants
from metermind.message_intelligence.text_cleaning import clean_messages


def build_text_pipeline():
    return Pipeline(
        steps=[
            ("clean", FunctionTransformer(clean_messages)),
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=constants.TEXT_MINIMUM_DOCUMENT_COUNT,
                    max_features=constants.TEXT_MAXIMUM_FEATURES,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classify",
                LogisticRegression(
                    random_state=constants.MODEL_SEED, max_iter=1000
                ),
            ),
        ]
    )


def split_complaints(corpus, label_column):
    text_col = None
    for col in ("message_text", "message", "text", "complaint_text"):
        if col in corpus.columns:
            text_col = col
            break
    if text_col is None:
        text_col = corpus.columns[0]

    raw_text = corpus[text_col]
    labels = corpus[label_column]

    return train_test_split(
        raw_text,
        labels,
        test_size=constants.HELD_OUT_SHARE,
        random_state=constants.MODEL_SEED,
        stratify=labels,
    )


def train_category_classifier(fitting_text, fitting_labels):
    model = build_text_pipeline()
    model.fit(fitting_text, fitting_labels)
    return model


def train_urgency_classifier(fitting_text, fitting_labels):
    model = build_text_pipeline()
    model.fit(fitting_text, fitting_labels)
    return model


def classify_message(category_model, urgency_model, message):
    cat = category_model.predict([message])[0]
    urg = urgency_model.predict([message])[0]
    return {"category": str(cat), "urgency": str(urg)}


if __name__ == "__main__":
    corpus = pd.read_csv(constants.COMPLAINT_MESSAGES_PATH)

    (
        cat_fit_text,
        cat_held_text,
        cat_fit_labels,
        cat_held_labels,
    ) = split_complaints(corpus, "category")
    category_model = train_category_classifier(cat_fit_text, cat_fit_labels)
    cat_score = round(
        float(category_model.score(cat_held_text, cat_held_labels)),
        constants.MEASUREMENT_DECIMAL_PLACES,
    )

    (
        urg_fit_text,
        urg_held_text,
        urg_fit_labels,
        urg_held_labels,
    ) = split_complaints(corpus, "urgency")
    urgency_model = train_urgency_classifier(urg_fit_text, urg_fit_labels)
    urg_score = round(
        float(urgency_model.score(urg_held_text, urg_held_labels)),
        constants.MEASUREMENT_DECIMAL_PLACES,
    )

    Path(constants.COMPLAINT_CATEGORY_MODEL_PATH).parent.mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(category_model, constants.COMPLAINT_CATEGORY_MODEL_PATH)

    Path(constants.COMPLAINT_URGENCY_MODEL_PATH).parent.mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(urgency_model, constants.COMPLAINT_URGENCY_MODEL_PATH)

    scores = {"category": cat_score, "urgency": urg_score}
    scores_path = Path(constants.COMPLAINT_SCORES_PATH)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    print(scores)
