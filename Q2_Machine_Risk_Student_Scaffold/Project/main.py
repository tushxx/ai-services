import os
import tempfile
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

TRAIN_PATH='data/machine_condition_train.csv'
TEST_PATH='data/machine_condition_test.csv'
PREDICT_PATH='data/machine_condition_predict.csv'
CLEAN_TRAIN_PATH='processed_data/machine_condition_train_cleaned.csv'
CLEAN_TEST_PATH='processed_data/machine_condition_test_cleaned.csv'
CLEAN_PREDICT_PATH='processed_data/machine_condition_predict_cleaned.csv'
LOGISTIC_MODEL_PATH='artifacts/logistic_regression_model.pkl'
SVC_MODEL_PATH='artifacts/svc_model.pkl'
OUTPUT_PATH = "output/machine_failure_risk_predictions.csv "

RECORD=['machine_id','inspection_date','plant_location','production_line','shift_engineer','maintenance_order_ref']

ENCODINGS = {
    "machine_type" : {"cnc_lathe" : 0, "milling_center" : 1, "hydraulic_press" : 2},
    "load_profile" : {"light" : 0, "normal" : 1, "heavy" : 2, "cyclic" : 3},
    "lubrication_system" : {"manual" : 0, "automatic" : 1},
    "failure_risk_band" : {"very_low" : 0, "low" : 1, "moderate" : 2, "high" : 3, "critical" : 4 }
}

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates()
    df = df.drop(columns = RECORD)
    for i in ["vibration_mm_s", "bearing_temperature_celsius", "average_daily_runtime_hours"]:
        df[i] = df[i].fillna(df[i].median())

    for col, mapping in ENCODINGS.items():
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower().map(mapping)

    df.to_csv(output_path, index = False)
    return df


def train_logistic_model(processed_path, model_path):
    df = pd.read_csv(processed_path)
    x = df.drop(columns = "failure_risk_band")
    y = df["failure_risk_band"]
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(x, y)
    joblib.dump(model, model_path)
    return model

def train_svc_model(processed_path, model_path):
    df = pd.read_csv(processed_path)
    x = df.drop(columns = "failure_risk_band")
    y = df["failure_risk_band"]
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', C=1.0, random_state=42))
    ])
    
    model.fit(x, y)
    joblib.dump(model, model_path)
    return model


def compare_models(logistic_model_path, svc_model_path, features_test, target_test):
    log_model = joblib.load(logistic_model_path)
    svc_model = joblib.load(svc_model_path)

    log_pred = log_model.predict(features_test)
    svc_pred = svc_model.predict(features_test)

    results = pd.DataFrame([
        {
            "model_name" : "LogisticRegression",
            "accuracy" : accuracy_score(target_test, log_pred),
            "f1_score" : f1_score(target_test, log_pred, average = 'macro')
        },
        {
            "model_name" : "SVC",
            "accuracy" : accuracy_score(target_test, svc_pred),
            "f1_score" : f1_score(target_test, svc_pred, average = 'macro')
        }
    ])
    return results.sort_values(by = 'accuracy', ascending = False).reset_index(drop = True)


def evaluate_model(model_path, features_test, target_test):
    pred = joblib.load(model_path).predict(features_test)
    return accuracy_score(target_test, pred), f1_score(target_test, pred, average = 'macro')


def predict_new_data(model_path, input_path, processed_path, output_path):
    model = joblib.load(model_path)
    ids = pd.read_csv(input_path)["machine_id"]
    cleaned_df = clean_data(input_path, processed_path)
    pred = model.predict(cleaned_df)
    band = {
        0 : "VERY_LOW",
        1 : "LOW",
        2 : "MODERATE",
        3 : "HIGH", 
        4 : "CRITICAL"
    }
    result = pd.DataFrame({
        "machine_id" : ids, 
        "predicted_failure_risk_band" : [band[p] for p in pred]
    })
    result.to_csv(output_path, index = False)
    return result


if __name__ == "__main__":
    train_df = clean_data(TRAIN_PATH, CLEAN_TRAIN_PATH)
    test_df = clean_data(TEST_PATH, CLEAN_TEST_PATH)

    features_test = test_df.drop(columns = "failure_risk_band")
    target_test = test_df["failure_risk_band"]

    train_logistic_model(CLEAN_TRAIN_PATH, LOGISTIC_MODEL_PATH)
    train_svc_model(CLEAN_TRAIN_PATH, SVC_MODEL_PATH)

    compare_df = compare_models(LOGISTIC_MODEL_PATH, SVC_MODEL_PATH, features_test, target_test)
    best = compare_df.iloc[0]["model_name"]
    selected_model_path = LOGISTIC_MODEL_PATH if best == "LogisticRegression" else SVC_MODEL_PATH

    acc, f1 = evaluate_model(selected_model_path, features_test, target_test)

    pred = predict_new_data(selected_model_path, PREDICT_PATH, CLEAN_PREDICT_PATH, OUTPUT_PATH)

    print("model done")