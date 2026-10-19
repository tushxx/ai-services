import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

FEATURES = ["virtual_machine_count", "average_vcpu_per_machine", "provisioned_storage_tb",
            "data_transfer_gb", "reserved_capacity_percent", "support_plan", "has_managed_services"]
TARGET = "monthly_cost_inr"
ENCODINGS = {
    "support_plan": {"basic": 0, "standard": 1, "premium": 2},
    "has_managed_services": {"no": 0, "yes": 1},
}


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path).drop_duplicates(keep="first")  # dedupe before any median
    df = df[[c for c in FEATURES + [TARGET] if c in df.columns]].copy()  # drops the 5 record columns

    for col in ["provisioned_storage_tb", "data_transfer_gb"]:
        df[col] = df[col].fillna(df[col].median())
    for col, mapping in ENCODINGS.items():
        df[col] = df[col].str.strip().str.lower().map(mapping)

    df.to_csv(output_path, index=False)
    return df


def split_features_target(processed_path):
    df = pd.read_csv(processed_path)
    return df.drop(columns=[TARGET]), df[TARGET]


def train_and_save(model, processed_path, model_path):
    X, y = split_features_target(processed_path)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model


def train_linear_model(processed_path, model_path):
    return train_and_save(LinearRegression(), processed_path, model_path)


def train_forest_model(processed_path, model_path):
    forest = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=10, random_state=42)
    return train_and_save(forest, processed_path, model_path)


def evaluate_model(model_path, features_test, target_test):
    predictions = joblib.load(model_path).predict(features_test)
    return (float(root_mean_squared_error(target_test, predictions)),
            float(mean_absolute_error(target_test, predictions)),
            float(r2_score(target_test, predictions)))


def compare_models(linear_model_path, forest_model_path, features_test, target_test):
    rows = []
    for name, path in [("LinearRegression", linear_model_path), ("RandomForestRegressor", forest_model_path)]:
        rmse, mae, r2 = evaluate_model(path, features_test, target_test)
        rows.append({"model_name": name, "rmse": rmse, "mae": mae, "r2_score": r2})
    return pd.DataFrame(rows).sort_values("r2_score", ascending=False).reset_index(drop=True)


def predict_new_data(model_path, input_path, processed_path, output_path):
    model = joblib.load(model_path)
    ids = pd.read_csv(input_path)["subscription_id"]
    features = clean_data(input_path, processed_path)
    result = pd.DataFrame({"subscription_id": ids.values,
                           "predicted_monthly_cost_inr": model.predict(features)})
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    for folder in ["processed_data", "artifacts", "output"]:
        os.makedirs(folder, exist_ok=True)
    paths = {"LinearRegression": "artifacts/linear_regression_model.pkl",
             "RandomForestRegressor": "artifacts/random_forest_model.pkl"}

    train = clean_data("data/cloud_spend_train.csv", "processed_data/cloud_spend_train.csv")
    test = clean_data("data/cloud_spend_test.csv", "processed_data/cloud_spend_test.csv")
    print("Cleaned training invoices:", train.shape)
    print("Cleaned held-back invoices:", test.shape)
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    train_linear_model("processed_data/cloud_spend_train.csv", paths["LinearRegression"])
    train_forest_model("processed_data/cloud_spend_train.csv", paths["RandomForestRegressor"])
    print("Models trained and saved.")

    comparison = compare_models(paths["LinearRegression"], paths["RandomForestRegressor"], X_test, y_test)
    print("Model comparison:")
    print(comparison.to_string(index=False))

    best_path = paths[comparison.loc[0, "model_name"]]
    rmse, mae, r2 = evaluate_model(best_path, X_test, y_test)
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")

    predictions = predict_new_data(best_path, "data/cloud_spend_predict.csv",
                                   "processed_data/cloud_spend_predict.csv",
                                   "output/predicted_monthly_costs.csv")
    print("Predictions written:", predictions.shape)