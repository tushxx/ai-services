"""Implement the Logistics Shipment Delay Risk Triage pipeline described in QUESTION.md."""

import joblib
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

TRAIN_DATA = "data/shipment_delay_train.csv"
TEST_PATH = "data/shipment_delay_test.csv"
PREDICT_PATH = "data/shipment_delay_predict.csv"
TRAIN_CLEANED = "processed_data/shipment_delay_train_cleaned.csv"
TEST_CLEANED = "processed_data/shipment_delay_test_cleaned.csv"
PREDICT_CLEANED = "processed_data/shipment_delay_predict_cleaned.csv"
TREE_MODEL_PATH = "artifacts/decision_tree_model.pkl"
SVC_MODEL_PATH = "artifacts/svc_model.pkl"
OUTPUT_PATH = "output/shipment_delay_predictions.csv"


RECORD_COLUMNS = [
    "shipment_id",
    "dispatch_date",
    "origin_hub_code",
    "destination_hub_code",
    "carrier_reference",
]

ENCODINGS = {
    "weather_severity" : {"clear" : 0, "rain" : 1, "storm" : 2, "extreme" : 3},
    "service_tier" : {"economy" : 0, "standard" : 1, "express" : 2},
    "is_weekend" : {"no" : 0, "yes" : 1},
    "delay_risk_level" : {"r1" : 0, "r2" : 1, "r3" : 2, "r4" : 3}
}

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop(columns = RECORD_COLUMNS)
    df = df.drop_duplicates()

    for i in ["hub_load_percent", "historical_delay_rate"]:
        df[i] = df[i].fillna(df[i].median())
    
    for col, mapping in ENCODINGS.items():
        
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower().map(mapping)

    df.to_csv(output_path, index = False)
    return df


def train_tree_model(processed_path, model_path):
    df = pd.read_csv(processed_path)
    X = df.drop(columns = "delay_risk_level")
    y = df["delay_risk_level"]

    model = DecisionTreeClassifier(max_depth = 10, random_state = 42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

def train_svc_model(processed_path, model_path):
    df = pd.read_csv(processed_path)
    X = df.drop(columns = "delay_risk_level")
    y = df["delay_risk_level"]

    model = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', C = 1.0, random_state = 42))
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model


def compare_models(tree_model_path, svc_model_path, features_test, target_test):
    tree_model = joblib.load(tree_model_path)
    svc_model = joblib.load(svc_model_path)

    tree_pred = tree_model.predict(features_test)
    svc_pred = svc_model.predict(features_test)

    result = pd.DataFrame([
        {
            "model_name" : "DecisionTreeClassifier",
            "accuracy" : accuracy_score(tree_pred, target_test),
            "f1_score" : f1_score(tree_pred, target_test, average = "macro")
        },
        {
            "model_name" : "SVC",
            "accuracy" : accuracy_score(svc_pred, target_test),
            "f1_score" : f1_score(svc_pred, target_test, average = "macro")
        }
    ])
    return result.sort_values(by = "accuracy", ascending = False).reset_index(drop = True)


def evaluate_model(model_path, features_test, target_test):
    pred = joblib.load(model_path).predict(features_test)
    return accuracy_score(pred, target_test), f1_score(pred, target_test, average = "macro")


def predict_new_data(model_path, input_path, processed_path, output_path):
    model = joblib.load(model_path)
    ids = pd.read_csv(input_path)["shipment_id"]
    cleaned_df = clean_data(input_path, processed_path)
    pred = model.predict(cleaned_df)
    band = {
        0 : "R1",
        1 : "R2",
        2 : "R3",
        3 : "R4"
    }
    result = pd.DataFrame({
        "shipment_id" : ids,
        "predicted_delay_risk_level" : [band[p] for p in pred]
    })
    result.to_csv(output_path, index = False)
    return result


if __name__ == "__main__":
    train_df = clean_data(TRAIN_DATA, TRAIN_CLEANED)
    test_df = clean_data(TEST_PATH, TEST_CLEANED)
    features_test = test_df.drop(columns = "delay_risk_level")
    target_test = test_df["delay_risk_level"]

    train_tree_model(TRAIN_CLEANED, TREE_MODEL_PATH)
    train_svc_model(TRAIN_CLEANED, SVC_MODEL_PATH)

    compare = compare_models(TREE_MODEL_PATH, SVC_MODEL_PATH, features_test, target_test)
    best = compare.iloc[0]["model_name"]
    selected_model_path = TREE_MODEL_PATH if best == "DecisionTreeClassifier" else SVC_MODEL_PATH

    acc, f1 = evaluate_model(selected_model_path, features_test, target_test)
    predictions = predict_new_data(selected_model_path, PREDICT_PATH,PREDICT_CLEANED, OUTPUT_PATH)
    print("model done")