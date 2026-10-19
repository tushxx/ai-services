from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "incidents_ml_dataset.csv"

FEATURES = [
    "category",
    "service",
    "impact",
    "urgency",
    "affected_users",
    "service_criticality",
    "reported_source",
]
TARGET = "priority"


def load_training_data(path: Path = DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop_duplicates().copy()

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing training columns: {missing}")

    df["affected_users"] = pd.to_numeric(df["affected_users"], errors="coerce")
    df = df.dropna(subset=required)

    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def build_preprocessor():
    categorical = [
        "category",
        "service",
        "impact",
        "urgency",
        "service_criticality",
        "reported_source",
    ]
    numeric = ["affected_users"]

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )
