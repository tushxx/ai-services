import os, tempfile, joblib, pytest, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

TRAIN_PATH = 'data/wind_turbine_train.csv'
TEST_PATH = 'data/wind_turbine_test.csv'
PREDICT_PATH = 'data/wind_turbine_predict.csv'
CLEAN_TRAIN_PATH = 'processed_data/wind_turbine_train_cleaned.csv'
CLEAN_TEST_PATH = 'processed_data/wind_turbine_test_cleaned.csv'
CLEAN_PREDICT_PATH = 'processed_data/wind_turbine_predict_cleaned.csv'
SVR_MODEL_PATH = 'artifacts/svr_model.pkl'
FOREST_MODEL_PATH = 'artifacts/random_forest_model.pkl'
OUTPUT_PATH = "output/wind_turbine_predictions.csv"

RECORD = ['turbine_id', 'inspection_date', 'wind_farm', 'service_region', 'maintenance_vendor', 'work_order_ref']

ENCODINGS = {
    "drive_train_type": {"geared": 0, "direct_drive": 1, "hybrid": 2},
    "weather_exposure": {"low": 0, "moderate": 1, "high": 2, "extreme": 3},
    "offshore_installation": {"no": 0, "yes": 1}
}

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates()
    cols_to_drop = [c for c in RECORD if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    for i in ["operating_hours", "rated_capacity_mw", "gearbox_vibration_mm_s"]:
        if i in df.columns:
            df[i] = df[i].fillna(df[i].median())

    for col, mapping in ENCODINGS.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

def train_svr_model(processed_path, model_path):
    df = pd.read_csv(processed_path)
    x = df.drop(columns=["annual_maintenance_cost_lakh"])
    y = df["annual_maintenance_cost_lakh"]
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR(kernel='rbf'))
    ])
    model.fit(x, y)
    output_dir = os.path.dirname(model_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, model_path)
    return model

def train_forest_model(processed_path, model_path):
    df = pd.read_csv(processed_path)
    x = df.drop(columns=["annual_maintenance_cost_lakh"])
    y = df["annual_maintenance_cost_lakh"]
    model = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=15, random_state=42)
    model.fit(x, y)
    output_dir = os.path.dirname(model_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, model_path)
    return model

def compare_models(svr_model_path, forest_model_path, features_test, target_test):
    svr_model = joblib.load(svr_model_path)
    forest_model = joblib.load(forest_model_path)

    svr_pred = svr_model.predict(features_test)
    forest_pred = forest_model.predict(features_test)

    results = pd.DataFrame([
        {
            "model_name": "SVR",
            "rmse": root_mean_squared_error(target_test, svr_pred),
            "mae": mean_absolute_error(target_test, svr_pred),
            "r2_score": r2_score(target_test, svr_pred)
        },
        {
            "model_name": "RandomForestRegressor",
            "rmse": root_mean_squared_error(target_test, forest_pred),
            "mae": mean_absolute_error(target_test, forest_pred),
            "r2_score": r2_score(target_test, forest_pred)
        }
    ])
    return results.sort_values(
        by="r2_score", ascending=False
    ).reset_index(drop=True)

def evaluate_model(model_path, features_test, target_test):
    model = joblib.load(model_path)
    pred = model.predict(features_test)
    return root_mean_squared_error(target_test, pred), mean_absolute_error(target_test, pred), r2_score(target_test, pred)

def predict_new_data(model_path, input_path, processed_path, output_path):
    model = joblib.load(model_path)
    ids = pd.read_csv(input_path)["turbine_id"]
    cleaned_df = clean_data(input_path, processed_path)
    pred = model.predict(cleaned_df)
    result = pd.DataFrame({
        "turbine_id": ids,
        "predicted_annual_maintenance_cost_lakh": pred
    })
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result

if __name__ == "__main__":
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    train_df = clean_data(TRAIN_PATH, CLEAN_TRAIN_PATH)
    test_df = clean_data(TEST_PATH, CLEAN_TEST_PATH)

    features_test = test_df.drop(columns="annual_maintenance_cost_lakh")
    target_test = test_df["annual_maintenance_cost_lakh"]

    train_svr_model(CLEAN_TRAIN_PATH, SVR_MODEL_PATH)
    train_forest_model(CLEAN_TRAIN_PATH, FOREST_MODEL_PATH)

    compare_df = compare_models(SVR_MODEL_PATH, FOREST_MODEL_PATH, features_test, target_test)
    best_model = compare_df.iloc[0]["model_name"]
    selected_model_path = SVR_MODEL_PATH if best_model == "SVR" else FOREST_MODEL_PATH

    rmse, mae, r2 = evaluate_model(selected_model_path, features_test, target_test)

    pred = predict_new_data(selected_model_path, PREDICT_PATH, CLEAN_PREDICT_PATH, OUTPUT_PATH)
    print("model done")
