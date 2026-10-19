import re
from pathlib import Path

import pandas as pd

from config import constants

QUOTED_REPLY_PATTERN = re.compile(
    r"(?:\n|^)\s*(?:>|On\s+.*wrote:|From:|Sent:|To:|Subject:).*",
    re.DOTALL | re.IGNORECASE,
)

GREETING_PATTERN = re.compile(
    r"^\s*(?:dear\s+sir|dear\s+madam|dear\s+team|dear\s+officer|respected\s+sir|respected\s+madam|hello|hi)[,\s]*$",
    re.IGNORECASE | re.MULTILINE,
)

SIGN_OFF_PATTERN = re.compile(
    r"^\s*(?:thanks\s+and\s+regards|thanks|regards|thanking\s+you|thank\s+you|yours\s+sincerely|yours\s+faithfully)[,\s]*$",
    re.IGNORECASE | re.MULTILINE,
)

SIGNATURE_PATTERN = re.compile(
    r"^\s*(?:mobile|phone|contact|tel|mob|staff\s*id|emp\s*id)?[:\s]*\(?\+?\d[\d\s\-\(\)]{7,}\d\s*$",
    re.IGNORECASE | re.MULTILINE,
)

COMPLAINT_REFERENCE_PATTERN = re.compile(r"\bCMP\d{7}\b", re.IGNORECASE)

WHITESPACE_PATTERN = re.compile(r"\s+")


def strip_quoted_reply(message):
    if not isinstance(message, str):
        message = str(message) if pd.notna(message) else ""
    return QUOTED_REPLY_PATTERN.sub("", message)


def strip_greeting_and_sign_off(message):
    if not isinstance(message, str):
        message = str(message) if pd.notna(message) else ""
    text = GREETING_PATTERN.sub("", message)
    text = SIGN_OFF_PATTERN.sub("", text)
    return text


def strip_signature(message):
    if not isinstance(message, str):
        message = str(message) if pd.notna(message) else ""
    return SIGNATURE_PATTERN.sub("", message)


def normalise_whitespace(message):
    if not isinstance(message, str):
        message = str(message) if pd.notna(message) else ""
    cleaned = WHITESPACE_PATTERN.sub(" ", message)
    return cleaned.strip()


def clean_message(message):
    if not isinstance(message, str):
        message = str(message) if pd.notna(message) else ""
    original = message

    text = strip_quoted_reply(message)
    text = strip_greeting_and_sign_off(text)
    text = strip_signature(text)
    text = COMPLAINT_REFERENCE_PATTERN.sub("", text)
    text = normalise_whitespace(text)

    if not text:
        text = normalise_whitespace(original)

    return text


def clean_messages(messages):
    return [clean_message(m) for m in messages]


def clean_corpus(messages_path, clean_path):
    df = pd.read_csv(messages_path)

    text_col = None
    for col in ("message_text", "message", "text", "complaint_text"):
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]

    cleaned_list = clean_messages(df[text_col])
    df["clean_text"] = cleaned_list

    out_path = Path(clean_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    shortest_len = min(len(t) for t in cleaned_list) if cleaned_list else 0

    return {"messages": int(len(df)), "shortest": int(shortest_len)}


if __name__ == "__main__":
    result = clean_corpus(
        constants.COMPLAINT_MESSAGES_PATH, constants.CLEAN_COMPLAINTS_PATH
    )
    print(result)
