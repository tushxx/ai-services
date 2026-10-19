import csv
from pathlib import Path

from config import constants

BATCH_FAILURE_FIELDS = ("message_id", "reason")


class MessageFailure(Exception):
    def __init__(self, message_id, reason):
        super().__init__(f"Message {message_id} failed: {reason}")
        self.message_id = message_id
        self.reason = reason


def run_batch(messages, handle_one, failures_path=constants.BATCH_FAILURES_PATH):
    results = []
    failures = []
    for msg in messages:
        try:
            res = handle_one(msg)
            results.append(res)
        except Exception as e:
            mid = msg.get("message_id", "") if isinstance(msg, dict) else getattr(msg, "message_id", "")
            fail = MessageFailure(mid, str(e))
            results.append(fail)
            failures.append({"message_id": mid, "reason": str(e)})

    out = Path(failures_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(BATCH_FAILURE_FIELDS))
        writer.writeheader()
        writer.writerows(failures)

    return results


if __name__ == "__main__":
    test_msgs = [
        {"message_id": "MSG000001", "units": 120},
        {"message_id": "MSG000002"},
    ]
    def example_handler(one):
        return one["units"] * 2
    res = run_batch(test_msgs, example_handler)
    print(f"handled and returned: {res[0]} {res[1].message_id} failed, and the batch carried on: '{res[1].reason}'")
    print(f"what failed is written to: {constants.BATCH_FAILURES_PATH}")
