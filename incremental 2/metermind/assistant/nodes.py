from pathlib import Path
import joblib

from config import constants
from metermind.core.exceptions import ToolError

category_model_path = constants.COMPLAINT_CATEGORY_MODEL_PATH
urgency_model_path = constants.COMPLAINT_URGENCY_MODEL_PATH
loaded_classifiers = {}


def triage_the_question(state):
    q_text = state.get("question", "")
    if not q_text or not str(q_text).strip():
        return {
            "refusal": "nothing was asked",
            "finished": True,
            "steps": ["triage"],
        }

    if "category" not in loaded_classifiers:
        cat_file = Path(category_model_path)
        if not cat_file.exists():
            raise ToolError(f"Classifier not found at {category_model_path}")
        loaded_classifiers["category"] = joblib.load(category_model_path)

    if "urgency" not in loaded_classifiers:
        urg_file = Path(urgency_model_path)
        if not urg_file.exists():
            raise ToolError(f"Classifier not found at {urgency_model_path}")
        loaded_classifiers["urgency"] = joblib.load(urgency_model_path)

    cat = loaded_classifiers["category"].predict([q_text])[0]
    urg = loaded_classifiers["urgency"].predict([q_text])[0]

    return {
        "category": cat,
        "urgency": urg,
        "steps": ["triage"],
    }


def run_tools(state):
    from metermind.assistant.tools import READING_TOOLS

    tool_map = {one.name: one for one in READING_TOOLS}
    plan = state.get("plan", [])

    for step in plan:
        tname = step.get("tool", "")
        if tname not in tool_map:
            return {
                "refusal": f"tool '{tname}' does not exist in the catalogue",
                "finished": True,
                "steps": ["tools"],
            }

    tool_calls = []
    for step in plan:
        tname = step.get("tool", "")
        args = step.get("arguments", {})
        tool = tool_map[tname]
        try:
            res = tool.invoke(args)
            tool_calls.append({"tool": tname, "arguments": args, "result": res})
        except Exception as e:
            tool_calls.append({"tool": tname, "arguments": args, "error": str(e)})

    return {"tool_calls": tool_calls, "steps": ["tools"]}


def approval_gate(state):
    hold_reasons = []

    # 1. Check if reading exception class is held
    tool_calls = state.get("tool_calls", [])
    for call in tool_calls:
        res = call.get("result", {})
        if isinstance(res, dict):
            ex_class = res.get("exception_class")
            if ex_class in constants.HELD_EXCEPTION_CLASSES:
                hold_reasons.append(f"exception class {ex_class} requires person confirmation")

    # 2. Check critic objection
    veto = state.get("veto", "")
    if veto and str(veto).strip():
        hold_reasons.append(str(veto).strip())

    # 3. Check empty draft
    draft = state.get("draft", "")
    if not draft or not str(draft).strip():
        hold_reasons.append("nothing was drafted")

    if hold_reasons:
        held_text = "; ".join(hold_reasons)
        return {
            "held": held_text,
            "answer": constants.HELD_FOR_A_PERSON_REPLY,
            "steps": ["gate"],
        }

    return {"held": "", "steps": ["gate"]}


if __name__ == "__main__":
    from metermind.assistant.state import new_assistant_state

    for q in (
        "Why is my bill for June three times the usual amount",
        "There has been no supply in our street since last ev",
        "",
    ):
        st = new_assistant_state(q)
        try:
            res = triage_the_question(st)
            print(f"'{q}' -> {res}")
        except Exception as e:
            print(f"'{q}' -> error {e}")
