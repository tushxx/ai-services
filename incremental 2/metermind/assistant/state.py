import operator
from typing import Annotated, Any, Dict, List, TypedDict

ACCUMULATING_CHANNELS = ("steps", "tool_calls")


class AssistantState(TypedDict):
    question: str
    consumer_category: str
    billing_period: str
    category: str
    urgency: str
    plan: List[Dict[str, Any]]
    draft: str
    answer: str
    refusal: str
    finished: bool
    steps: Annotated[List[str], operator.add]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    veto: str
    held: str


def new_assistant_state(question, consumer_category="domestic", billing_period=""):
    return {
        "question": question,
        "consumer_category": consumer_category,
        "billing_period": billing_period,
        "category": "",
        "urgency": "",
        "plan": [],
        "draft": "",
        "answer": "",
        "refusal": "",
        "finished": False,
        "steps": [],
        "tool_calls": [],
        "veto": "",
        "held": "",
    }


if __name__ == "__main__":
    state = new_assistant_state("Why is my bill so high this month?")
    print(sorted(state.keys()))
    print(state)
