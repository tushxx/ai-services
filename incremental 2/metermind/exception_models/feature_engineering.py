import json
from pathlib import Path

import pandas as pd

from config import constants
from metermind.exception_models.baseline_model import split_readings
from metermind.exception_models.feature_pipeline import modelling_columns
from metermind.exception_models.tree_models import train_random_forest

DAYS_IN_A_YEAR = 365.25


def add_consumption_ratio_to_average(book):
    df = book.copy()
    if "units_consumed" in df.columns and "twelve_month_average_units" in df.columns:
        avg = pd.to_numeric(df["twelve_month_average_units"], errors="coerce")
        units = pd.to_numeric(df["units_consumed"], errors="coerce")
        ratio = units / avg
        ratio = ratio.replace([float("inf"), float("-inf")], float("nan")).fillna(1.0)
        df["consumption_ratio_to_average"] = ratio
    else:
        df["consumption_ratio_to_average"] = 1.0
    return df


def add_load_factor(book):
    df = book.copy()
    if all(
        col in df.columns
        for col in ("units_consumed", "sanctioned_load_kw", "cycle_days")
    ):
        units = pd.to_numeric(df["units_consumed"], errors="coerce")
        load = pd.to_numeric(df["sanctioned_load_kw"], errors="coerce")
        days = pd.to_numeric(df["cycle_days"], errors="coerce")
        max_draw = load * days * constants.HOURS_IN_A_DAY
        load_factor = units / max_draw
        load_factor = (
            load_factor.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
        )
        df["load_factor"] = load_factor
    else:
        df["load_factor"] = 0.0
    return df


def add_meter_age_years(book):
    df = book.copy()
    install_col = None
    for col in ("meter_install_date", "meter_installation_date", "connection_date"):
        if col in df.columns:
            install_col = col
            break

    end_col = None
    for col in ("cycle_end_date", "read_date", "read_on"):
        if col in df.columns:
            end_col = col
            break

    if install_col and end_col:
        install_dt = pd.to_datetime(df[install_col], errors="coerce")
        end_dt = pd.to_datetime(df[end_col], errors="coerce")
        days = (end_dt - install_dt).dt.days
        age_years = days / DAYS_IN_A_YEAR
        df["meter_age_years"] = age_years
    elif "meter_age_months" in df.columns:
        months = pd.to_numeric(df["meter_age_months"], errors="coerce")
        df["meter_age_years"] = months / 12.0
    else:
        df["meter_age_years"] = float("nan")

    return df


def widen_readings(book):
    df = add_consumption_ratio_to_average(book)
    df = add_load_factor(df)
    df = add_meter_age_years(df)
    return df


def measure_feature_lift(book):
    widened_book = widen_readings(book)

    # Narrow split & model
    fitting_n, held_n, classes_fit_n, classes_held_n = split_readings(
        book, columns=modelling_columns(widened=False)
    )
    model_n = train_random_forest(fitting_n, classes_fit_n, widened=False)
    score_n = float(model_n.score(held_n, classes_held_n))

    # Widened split & model
    fitting_w, held_w, classes_fit_w, classes_held_w = split_readings(
        widened_book, columns=modelling_columns(widened=True)
    )
    model_w = train_random_forest(fitting_w, classes_fit_w, widened=True)
    score_w = float(model_w.score(held_w, classes_held_w))

    narrow = round(score_n, constants.MEASUREMENT_DECIMAL_PLACES)
    widened = round(score_w, constants.MEASUREMENT_DECIMAL_PLACES)
    lift = round(widened - narrow, constants.MEASUREMENT_DECIMAL_PLACES)

    return {"narrow": narrow, "widened": widened, "lift": lift}


if __name__ == "__main__":
    book = pd.read_csv(constants.CLEAN_READING_BOOK_PATH)
    widened_book = widen_readings(book)

    Path(constants.WIDENED_READINGS_PATH).parent.mkdir(parents=True, exist_ok=True)
    widened_book.to_csv(constants.WIDENED_READINGS_PATH, index=False)

    lift_info = measure_feature_lift(book)

    lift_path = Path(constants.FEATURE_LIFT_PATH)
    lift_path.parent.mkdir(parents=True, exist_ok=True)
    lift_path.write_text(json.dumps(lift_info, indent=2), encoding="utf-8")

    print(f"narrow   {lift_info['narrow']}")
    print(f"widened  {lift_info['widened']}")
    print(f"lift     {lift_info['lift']}")
