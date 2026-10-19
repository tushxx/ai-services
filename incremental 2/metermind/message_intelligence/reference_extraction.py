import csv
from pathlib import Path
import re
import pandas as pd

from config import constants

ACCOUNT_NUMBER_PATTERN = re.compile(r"\bACC\d{8}\b")
METER_SERIAL_PATTERN = re.compile(r"\bMTR-[A-Z]{2}-\d{6}\b")
COMPLAINT_REFERENCE_PATTERN = re.compile(r"\bCMP\d{7}\b")

EXTRACTED_REFERENCE_FIELDS = (
    "message_id",
    "account_numbers",
    "meter_serials",
    "complaint_references",
    "unknown_accounts",
    "serials_on_another_account",
)


def find_account_numbers(message):
    text = str(message) if message is not None else ""
    matches = ACCOUNT_NUMBER_PATTERN.findall(text)
    return list(dict.fromkeys(matches))


def find_meter_serials(message):
    text = str(message) if message is not None else ""
    matches = METER_SERIAL_PATTERN.findall(text)
    return list(dict.fromkeys(matches))


def find_complaint_references(message):
    text = str(message) if message is not None else ""
    matches = COMPLAINT_REFERENCE_PATTERN.findall(text)
    return list(dict.fromkeys(matches))


def check_against_connection_register(accounts, serials, known_accounts, serial_owner):
    unknown_accounts = [acc for acc in accounts if acc not in known_accounts]
    if not accounts:
        serials_on_another_account = []
    else:
        serials_on_another_account = [
            s for s in serials if s in serial_owner and serial_owner[s] not in accounts
        ]
    return {
        "unknown_accounts": unknown_accounts,
        "serials_on_another_account": serials_on_another_account,
    }


def extract_and_check(messages_path, register_path, extracted_path):
    reg_df = pd.read_csv(register_path)
    known_accounts = set(reg_df["account_number"].dropna().astype(str))
    serial_owner = dict(
        zip(
            reg_df["meter_serial"].dropna().astype(str),
            reg_df["account_number"].dropna().astype(str),
        )
    )

    msg_df = pd.read_csv(messages_path)
    rows = []
    faults_count = 0

    text_col = "message_text" if "message_text" in msg_df.columns else msg_df.columns[1]
    id_col = "message_id" if "message_id" in msg_df.columns else msg_df.columns[0]

    for _, row in msg_df.iterrows():
        mid = str(row[id_col])
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        accs = find_account_numbers(text)
        serials = find_meter_serials(text)
        refs = find_complaint_references(text)
        checked = check_against_connection_register(
            accs, serials, known_accounts, serial_owner
        )
        if checked["unknown_accounts"] or checked["serials_on_another_account"]:
            faults_count += 1

        rows.append({
            "message_id": mid,
            "account_numbers": ";".join(accs),
            "meter_serials": ";".join(serials),
            "complaint_references": ";".join(refs),
            "unknown_accounts": ";".join(checked["unknown_accounts"]),
            "serials_on_another_account": ";".join(checked["serials_on_another_account"]),
        })

    out = Path(extracted_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(EXTRACTED_REFERENCE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    return {"messages": len(msg_df), "faults": faults_count}


if __name__ == "__main__":
    result = extract_and_check(
        constants.COMPLAINT_MESSAGES_PATH,
        constants.CONNECTION_REGISTER_PATH,
        constants.EXTRACTED_REFERENCES_PATH,
    )
    print(result)
