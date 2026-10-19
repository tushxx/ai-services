import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ml.preprocess import FEATURES, load_training_data, build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "incident_priority_model.joblib"


def train():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    classifier = DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    labels = sorted(y.unique().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "weighted_precision": round(float(precision), 6),
        "weighted_recall": round(float(recall), 6),
        "weighted_f1": round(float(f1), 6),
        "test_records": int(len(y_test)),
        "train_records": int(len(y_train)),
        "classes": labels,
        "classification_report": classification_report(
            y_test, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "actual"
    cm_df.to_csv(ARTIFACT_DIR / "confusion_matrix.csv")

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    feature_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    feature_df.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)

    joblib.dump(pipeline, MODEL_PATH)
    (ARTIFACT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Model saved: {MODEL_PATH}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    return metrics


if __name__ == "__main__":
    train()
