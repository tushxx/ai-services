import json
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from config import constants
from metermind.core.exceptions import ToolError

readings_database_path = constants.READINGS_DATABASE_PATH
champion_model_path = constants.CHAMPION_MODEL_PATH
operating_rule_path = constants.OPERATING_RULE_PATH
exception_report_path = constants.EXCEPTION_REPORT_PATH
tariff_store_directory = constants.TARIFF_STORE_DIRECTORY
opened_resources = {}


class ReadingRequest(BaseModel):
    reading_identifier: str = Field(
        description="Find one reading on file by its identifier: the meter, the account, the dates, the units and everything recorded against the cycle."
    )


class BillRequest(BaseModel):
    consumer_category: str = Field(
        description="The category of consumer connection, one of domestic, commercial, agricultural."
    )
    units_consumed: float = Field(
        ge=0.0,
        description="The units of energy consumed in the billing cycle, not below zero."
    )
    sanctioned_load_kw: float = Field(
        gt=0.0,
        description="The approved load connected in kilowatts, above zero."
    )

    @field_validator("consumer_category")
    @classmethod
    def a_category_that_is_billed(cls, given):
        cat = str(given).strip().lower()
        if cat not in constants.CONSUMER_CATEGORIES:
            raise ValueError(
                f"category '{given}' is outside the published three: {', '.join(constants.CONSUMER_CATEGORIES)}"
            )
        return cat


class TariffRequest(BaseModel):
    question: str = Field(
        description="Find what the tariff order that was actually in force for that consumer category in that billing period says."
    )
    consumer_category: str = Field(
        description="The consumer category asked about, one of domestic, commercial, agricultural."
    )
    billing_period: str = Field(
        pattern=constants.BILLING_PERIOD_PATTERN,
        description="The billing period cycle asked about, in YYYY-MM format."
    )

    @field_validator("consumer_category")
    @classmethod
    def a_category_with_a_schedule(cls, given):
        cat = str(given).strip().lower()
        if cat not in constants.CONSUMER_CATEGORIES:
            raise ValueError(
                f"category '{given}' has no published schedule: {', '.join(constants.CONSUMER_CATEGORIES)}"
            )
        return cat


class AssistantTool:
    def __init__(self, name, description, args_schema, func):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func

    def invoke(self, input_dict):
        validated = self.args_schema.model_validate(input_dict)
        return self.func(**validated.model_dump())

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def _look_up_reading(reading_identifier):
    from metermind.data_access.database_loader import ReadingDatabase

    db = ReadingDatabase(readings_database_path)
    row = db.find_reading_by_identifier(reading_identifier)
    if row is None:
        raise ToolError(f"there is no reading {reading_identifier} on file")
    return dict(row)


def _calculate_bill(consumer_category, units_consumed, sanctioned_load_kw):
    from metermind.core.tariff_calculations import calculate_bill_amount

    amt = round(
        calculate_bill_amount(consumer_category, units_consumed, sanctioned_load_kw),
        constants.CURRENCY_DECIMAL_PLACES,
    )
    return {"current_demand_inr": amt}


def _predict_reading_exception(reading_identifier):
    import joblib
    from metermind.exception_models.champion_model import predict_with_rule
    from metermind.exception_models.feature_engineering import widen_readings
    from metermind.exception_models.feature_pipeline import modelling_columns

    reading = _look_up_reading(reading_identifier)

    if "champion" not in opened_resources:
        opened_resources["champion"] = joblib.load(champion_model_path)
    if "operating_rule" not in opened_resources:
        with open(operating_rule_path, "r", encoding="utf-8") as f:
            opened_resources["operating_rule"] = json.load(f)
    if "exception_report" not in opened_resources:
        opened_resources["exception_report"] = pd.read_csv(exception_report_path)

    df_read = pd.DataFrame([reading])
    df_widened = widen_readings(df_read)
    cols = list(modelling_columns(widened=True))

    champion = opened_resources["champion"]
    rule = opened_resources["operating_rule"]
    report = opened_resources["exception_report"]

    probas = champion.predict_proba(df_widened[cols])
    preds = predict_with_rule(rule, probas)
    pred_class = preds[0]

    match_rows = report.loc[report["exception_class"] == pred_class, "precision"]
    precision = float(match_rows.iloc[0]) if not match_rows.empty else 0.0

    return {
        "reading": reading,
        "exception_class": pred_class,
        "precision": precision,
    }


def _search_tariff_order(question, consumer_category, billing_period):
    from metermind.retrieval.filtered_search import NoOrderInForce, filtered_search
    from metermind.retrieval.vector_store import load_embedder, open_tariff_store

    if "store" not in opened_resources:
        opened_resources["store"] = open_tariff_store(tariff_store_directory)
    if "embedder" not in opened_resources:
        opened_resources["embedder"] = load_embedder()

    try:
        return filtered_search(
            opened_resources["store"],
            question,
            opened_resources["embedder"],
            consumer_category,
            billing_period,
        )
    except NoOrderInForce as e:
        raise ToolError(str(e))


look_up_reading = AssistantTool(
    name="look_up_reading",
    description="Find one reading on file by its identifier: the meter, the account, the dates, the units and everything recorded against the cycle.",
    args_schema=ReadingRequest,
    func=_look_up_reading,
)

calculate_bill = AssistantTool(
    name="calculate_bill",
    description="Work out what a billing cycle costs under the schedule in force: the slab-by-slab energy charge, the fixed charge on the sanctioned load, the electricity duty and the rebate. It does not evaluate reading plausibility.",
    args_schema=BillRequest,
    func=_calculate_bill,
)

predict_reading_exception = AssistantTool(
    name="predict_reading_exception",
    description="Say which of the five exception classes a reading on file belongs to, and how often that answer is right.",
    args_schema=ReadingRequest,
    func=_predict_reading_exception,
)

search_tariff_order = AssistantTool(
    name="search_tariff_order",
    description="Find what the tariff order that was actually in force for that consumer category in that billing period says. Never answer tariff questions from memory.",
    args_schema=TariffRequest,
    func=_search_tariff_order,
)

READING_TOOLS = [
    look_up_reading,
    calculate_bill,
    predict_reading_exception,
    search_tariff_order,
]


if __name__ == "__main__":
    for tool in READING_TOOLS:
        print(tool.name, list(tool.args_schema.model_fields.keys()))
