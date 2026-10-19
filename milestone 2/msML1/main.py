import joblib
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, f1_score


TRAIN_PATH = "data/service_ticket_train.csv"
TEST_PATH = "data/service_ticket_test.csv"
PREDICT_PATH = "data/service_ticket_predict.csv"
TRAIN_CLEANED = "processed_data/train_cleaned.csv"
TEST_CLEANED = "processed_data/test_cleaned.csv"
PREDICT_CLEANED = "processed_data/predict_cleaned.csv"
TREE_MODEL = "artifacts/decision_tree_model.pkl"
SVC_MODEL = "artifacts/svc_model.pkl"
OUTPUT_PATH = "output/service_ticket_predictions.csv"


RECORD_COLUMNS = [                 #columns which needs to be deleted
    "ticket_id",
    "raised_date",
    "reporting_location",
    "assigned_group",
    "requester_department"
]


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates()
    df = df.drop(columns = RECORD_COLUMNS)

    for i in ["affected_user_count", "average_resolution_hours"]:
        df[i] = df[i].fillna(df[i].median())


    for i in ["service_criticality", "is_business_hours"]:
        df[i] = (
            df[i]
            .str.strip()
            .str.lower()
        )
    df["service_criticality"] = df["service_criticality"].map(
        {
            "low": 0, "medium": 1, "high": 2, "critical": 3
        }
    )
    df["is_business_hours"] = df["is_business_hours"].map(
        {
            "no": 0, "yes": 1
        }
    )

    if "priority_level" in df.columns:
        df["priority_level"] = (
            df["priority_level"]
            .str.strip()
            .str.lower()
        )
        df["priority_level"] = df["priority_level"].map(
            {
                "p1": 0, "p2": 1, "p3": 2, "p4": 3
            }
        )

    # ENCODINGS = {
    #     "service_criticality": {"low": 0, "medium": 1, "high": 2, "critical": 3},
    #     "is_business_hours": {"no": 0, "yes": 1},
    #     "priority_level": {"p1": 0, "p2": 1, "p3": 2, "p4": 3}, 
    # }
    # for col, mapping in ENCODINGS.items():
    #     if col in df.columns:  # the untriaged queue has no priority_level
    #         df[col] = df[col].str.strip().str.lower().map(mapping)


    df.to_csv(output_path, index = False)
    return df


def tree_classifier(processed_path, model_path):
    df = pd.read_csv(processed_path)

    X = df.drop(columns = ["priority_label"])
    y = df["priority_label"]

    model = DecisionTreeClassifier(max_depth = 10, random_state = 42)

    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

def svc_classifier(processed_path, model_path):
    df = pd.read_csv(processed_path)

    X = df.drop(columns = ["priority_label"])
    y = df["priority_label"]

    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, random_state=42))

    model = Pipeline([
        ('scaler', StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, random_state=42))
    ])
    # Use make_pipeline by default for cleaner, faster code. Switch to Pipeline only if you want explicit, custom names for your steps.

    model.fit(X, y)
    joblib.dump(model, model_path)
    return model


def compare_models(tree_model_path, svc_model_path, features_test, target_test):
    tree = joblib.load(tree_model_path)
    svc = joblib.load(svc_model_path)

    tree_predictions = tree.predict(features_test)
    svc_predictions = svc.predict(features_test)
    result_df = pd.DataFrame([
        {
            "model_name" : "DecisionTreeClassifier",
            "accuracy" : accuracy_score(target_test, tree_predictions),
            "f1_score" : f1_score(target_test, tree_predictions, average = "macro")
        },
        {
            "model_name" : "SupportVectorClassifier",
            "accuracy" : accuracy_score(target_test, svc_predictions),
            "f1_score" : f1_score(target_test, svc_predictions, average = "macro")
        }
    ])

    result_df = result_df.sort_values(by  = "accuracy", ascending = False).reset_index(drop = True)

    return result_df

def evaluate_model(model_path, features_test, target_test):
    predictions = joblib.load(model_path).predict(features_test)
    return (float(accuracy_score(target_test, predictions)),
            float(f1_score(target_test, predictions, average="macro")))


# def compare_models(tree_model_path, svc_model_path, features_test, target_test):
#     rows = []
#     for name, path in [("DecisionTreeClassifier", tree_model_path), ("SVC", svc_model_path)]:
#         accuracy, f1 = evaluate_model(path, features_test, target_test)
#         rows.append({"model_name": name, "accuracy": accuracy, "f1_score": f1})
#     return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)



def predict_new_data(model_path, input_path, processed_path, output_path):
    model = joblib.load(model_path)
    ids = pd.read_csv(input_path)["ticket_id"]
    features = clean_data(input_path, processed_path)
    result = pd.DataFrame({"ticket_id": ids.values,
                           "predicted_priority_level": [BANDS[code] for code in model.predict(features)]})
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    for folder in ["processed_data", "artifacts", "output"]:
        os.makedirs(folder, exist_ok=True)
    paths = {"DecisionTreeClassifier": "artifacts/decision_tree_model.pkl", "SVC": "artifacts/svc_model.pkl"}

    train = clean_data("data/service_ticket_train.csv", "processed_data/service_ticket_train.csv")
    test = clean_data("data/service_ticket_test.csv", "processed_data/service_ticket_test.csv")
    print("Cleaned training tickets:", train.shape)
    print("Cleaned held-back tickets:", test.shape)
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    train_tree_model("processed_data/service_ticket_train.csv", paths["DecisionTreeClassifier"])
    train_svc_model("processed_data/service_ticket_train.csv", paths["SVC"])
    print("Models trained and saved.")

    comparison = compare_models(paths["DecisionTreeClassifier"], paths["SVC"], X_test, y_test)
    print("Model comparison:")
    print(comparison.to_string(index=False))

    best_path = paths[comparison.loc[0, "model_name"]]
    accuracy, f1 = evaluate_model(best_path, X_test, y_test)
    print(f"Accuracy: {accuracy:.4f} | Macro F1: {f1:.4f}")

    predictions = predict_new_data(best_path, "data/service_ticket_predict.csv",
                                   "processed_data/service_ticket_predict.csv",
                                   "output/predicted_priority_levels.csv")
    print("Predictions written:", predictions.shape)



