import joblib
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline


TRAIN_PATH = "data/service_ticket_train.csv"
TEST_PATH = "data/service_ticket_test.csv"
PREDICT_PATH = "data/service_ticket_predict.csv"
TRAIN_CLEANED = "processed_data/train_cleaned.csv"
TEST_CLEANED = "processed_data/test_cleaned.csv"
PREDICT_CLEANED = "processed_data/predict_cleaned.csv"
TREE_MODEL = "artifacts/decision_tree_model.pkl"
SVC_MODEL = "artifacts/svc_model.pkl"
OUTPUT_PATH = "output/service_ticket_predictions.csv"


RECORD_COLUMNS = [
    "ticket_id",
    "raised_date",
    "reporting_location",
    "assigned_group",
    "requester_department"
]

ENCODINGS = {
    "service_criticality" : {"low": 0, "medium": 1, "high": 2, "critical": 3},
    "is_business_hours": {"no": 0, "yes": 1},
    "priority_level": {"p1": 0, "p2": 1, "p3": 2, "p4": 3}, 
}


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates()
    df = df.drop(columns = RECORD_COLUMNS)

    for i in ["affected_user_count", "average_resolution_hours"]:
        df[i] = df[i].fillna(df[i].median())

    
    for col, mapping in ENCODINGS.items():
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower().map(mapping)

    df.to_csv(output_path, index = False)
    return df


def tree_classifier(processed_path, model_path):
    df = df.read_csv(processed_path)
    X = df.drop(columns = ["priority_label"])
    y = df["priority_label"]

    model = DecisionTreeClassifier(max_depth= 10, random_state = 42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

def svc_classifier(processed_path, model_path):
    df = df.read_csv(processed_path)
    X = df.drop(columns = ["priority_label"])
    y = df["priority_label"]

    model = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', C = 1.0, random_state = 42))
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model


def compare_models(tree_model_path, svc_model_path, features_test, target_test):
    tree_model = joblib.load(tree_model_path)
    svc_model = joblib.load(svc_model_path)

    tree_predictions = tree_model.predict(features_test)
    svc_predictions = svc_model.predict(features_test)
    result_df = pd.DataFrame([
        {
            "model_name" : "DecisionTreeClassifier",
            "accuracy" : accuracy_score(target_test, tree_predictions),
            "f1_score" : f1_score(target_test, tree_predictions, average = "macro")
        },
        {
            "model_name" : "SVC",
            "accuracy" : accuracy_score(target_test, svc_predictions),
            "f1_score" : f1_score(target_test, svc_predictions, average = "macro")
        }
    ])
    return result_df.sort_values(by = "accuracy", ascending = False).reset_index(drop = True)


def evaluate_model(model_path, features_test, target_test):
    model = joblib.load(model_path)
    predictions = model.predict(features_test)
    accuracy = accuracy_score(target_test, predictions)
    f1 = f1_score(target_test, predictions, average = "macro")
    return accuracy, f1

def predict_new_data(model_path, input_path, processed_path, output_path):
    model = joblib.load(model_path)
    ids = pd.read_csv(input_path)["ticket_id"]
    cleaned_df = clean_data(input_path, processed_path)
    predictions = model.predict(cleaned_df)

    levels = {
        0 : "p1", 1 : "p2", 2 : "p3", 3 : "p4"
    }
    result = pd.DataFrame({
        "ticket_ids" : ids,
        "predicted_priority_level": [levels[int(i)] for i in predictions]
    })

    result.to_csv(output_path, index = False)
    return result


if __name__ ==  "__main__":
    train_df = clean_data(TRAIN_PATH, TRAIN_CLEANED)
    test_df = clean_data(TEST_PATH, TEST_CLEANED)

    X_test = test_df.drop(columns = ["ticket_id"])
    y_test = test_df["ticket_id"]

    train_tree_model(TRAIN_CLEANED, TREE_PATH)
    train_svc_model(TRAIN_CLEANED, SVC_PATH)

    comparison_df = compare_models(TREE_PATH, SVC_PATH, X_test, y_test)
    top_model = comparison_df.iloc[0]["model_name"]
    selected_model_path = TREE_PATH if top_model == "DecisionTreeClassifier" else SVC_PATH

    
    acc,f1 = evaluate_model(selected_model_path, X_test, y_test)

    predictions_df = predict_new_data(selected_model_path, PREDICT_PATH, PREDICT_CLEANED, OUTPUT_PATH)
    print("kaam hogaya")