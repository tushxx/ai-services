import csv
import datetime
from pathlib import Path
import pandas as pd

from config import constants
from metermind.message_intelligence.batch_pipeline import MessageFailure, run_batch
from metermind.message_intelligence.llm_client import LanguageModelError, ask_model, read_answer

TONE_PROMPT = """Read the tone of this complaint:
"{message_text}"

Answer with a JSON object holding:
"tone": one of calm, frustrated, angry
"reason": a phrase from the complaint as evidence
"""

ESCALATION_FIELDS = (
    "message_id",
    "age_days",
    "urgency",
    "tone",
    "escalate",
    "signals",
)


def complaint_age_days(received_date, today=None):
    if today is None:
        today = datetime.date.today()
    if isinstance(received_date, (datetime.date, datetime.datetime)):
        r_date = received_date if isinstance(received_date, datetime.date) else received_date.date()
    else:
        r_date = datetime.date.fromisoformat(str(received_date).strip()[:10])
    return (today - r_date).days


def decide_escalation(age_days, urgency, tone):
    signals = []
    if age_days >= constants.COMPLAINT_AGE_ESCALATION_DAYS:
        signals.append("waiting too long")
    if urgency in constants.ESCALATING_URGENCIES:
        signals.append("urgency")
    if tone in constants.ESCALATING_TONES:
        signals.append("tone")

    escalate = len(signals) >= constants.ESCALATION_SIGNALS_NEEDED
    return {"escalate": bool(escalate), "signals": signals}


def read_tone(message_text, cache_path=constants.LANGUAGE_MODEL_CACHE_PATH):
    prompt = TONE_PROMPT.format(message_text=message_text)
    said = ask_model(prompt, cache_path=cache_path)
    data = read_answer(said, constants.TONE_REPLY_KEYS)
    tone = str(data["tone"]).strip().lower()
    if tone not in constants.TONE_READINGS:
        raise LanguageModelError(f"Unexpected tone: {tone}")
    return {"tone": tone, "reason": data["reason"]}


def escalate_one_message(
    message,
    urgency_model,
    today=None,
    cache_path=constants.LANGUAGE_MODEL_CACHE_PATH,
):
    msg_id = message["message_id"]
    msg_text = message["message_text"]
    age = complaint_age_days(message["received_date"], today)
    urgency = urgency_model.predict([msg_text])[0]
    tone_info = read_tone(msg_text, cache_path=cache_path)
    decision = decide_escalation(age, urgency, tone_info["tone"])

    return {
        "message_id": msg_id,
        "age_days": age,
        "urgency": urgency,
        "tone": tone_info["tone"],
        "escalate": decision["escalate"],
        "signals": decision["signals"],
    }


def write_escalations(
    messages,
    urgency_model,
    escalations_path,
    today=None,
    cache_path=constants.LANGUAGE_MODEL_CACHE_PATH,
):
    def handle_one(msg):
        return escalate_one_message(
            msg, urgency_model, today=today, cache_path=cache_path
        )

    results = run_batch(messages, handle_one)
    decided = 0
    failed = 0
    escalated = 0
    rows = []

    for res in results:
        if isinstance(res, MessageFailure):
            failed += 1
        else:
            decided += 1
            if res["escalate"]:
                escalated += 1
            row = dict(res)
            row["signals"] = ";".join(res["signals"])
            row["escalate"] = str(res["escalate"])
            rows.append(row)

    out = Path(escalations_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ESCALATION_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    return {"decided": decided, "failed": failed, "escalated": escalated}


if __name__ == "__main__":
    import joblib

    try:
        urgency_model = joblib.load(constants.COMPLAINT_URGENCY_MODEL_PATH)
        df = pd.read_csv(constants.COMPLAINT_MESSAGES_PATH).head(
            constants.ESCALATION_SAMPLE_MESSAGES
        )
        sample = df.to_dict(orient="records")
        res = write_escalations(
            sample, urgency_model, constants.ESCALATIONS_PATH
        )
        print(res)
    except Exception as e:
        print(f"Skipping main execution: {e}")
