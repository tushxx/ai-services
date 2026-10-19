import json
import re

from config import constants
from metermind.assistant.tools import READING_TOOLS
from metermind.message_intelligence.llm_client import (
    LanguageModelError,
    ask_model_with_retry,
    read_answer,
)

language_model_cache_path = constants.LANGUAGE_MODEL_CACHE_PATH

PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z0-9_]+)\}")
PLANNER_KEYS = ("plan", "reason")
RESOLVER_KEYS = ("draft", "reason")
CRITIC_KEYS = ("veto", "reason")
RESPONDER_KEYS = ("reply", "reason")

TOOL_CATALOGUE = {
    one.name: {
        "does": one.description,
        "arguments": {k: (v.description or "") for k, v in one.args_schema.model_fields.items()},
    }
    for one in READING_TOOLS
}

PLANNER_PROMPT = """You are an assistant planning which tools to call to answer a consumer query.
You have access to this catalogue of tools:
{catalogue}

You may use at most {rounds} rounds of tool calls.
Query: "{question}"
Category: {category}
Urgency: {urgency}
Consumer Category: {consumer_category}
Billing Period: {billing_period}

Instructions:
- Reply with a JSON object holding "plan" and "reason".
- "plan" must be a list of at most {rounds} steps, where each step is an object with "tool" (exact tool name) and "arguments" (exact argument names as in catalogue).
- Forbid naming any tool outside the catalogue.
- Require the exact argument names listed against each tool.
- Leave a step out rather than guess an argument you cannot fill.
- An empty plan [] is allowed where no tool is needed.
"""

RESOLVER_PROMPT = """You are drafting an answer to a consumer's query based solely on the results returned by tools.
Question: "{question}"
Tool Calls and Results:
{tool_calls}

Instructions:
- Confine your draft to what the tools returned.
- Forbid working out any figure the tools did not return.
- Say plainly where the tools did not answer the question rather than fill the gap.
- Reply with a JSON object holding "draft" and "reason".
"""

CRITIC_PROMPT = """You are a critic reviewing a draft answer against the tool results.
Question: "{question}"
Tool Calls:
{tool_calls}
Draft:
"{draft}"

Instructions:
- You may only object; you may NOT rewrite the draft and may NOT answer the question yourself.
- Object if:
  1. A figure is quoted that the tools did not return.
  2. A statement contradicts what the tools returned.
  3. An accusation is made against the consumer that the tools did not establish.
- Reply with a JSON object holding:
  "veto": true if you object, false otherwise
  "reason": reason for objection if veto is true, or why every figure came from a tool if false
"""

RESPONDER_PROMPT = """You are formatting the final reply to the consumer based on the approved draft.
Question: "{question}"
Draft:
"{draft}"

Instructions:
- Confine the reply to what the draft says, in plain language, without jargon and without any figure the draft does not carry.
- Reply with a JSON object holding:
  "reply": final consumer response text
  "reason": short reason
"""


def build_agent_prompt(template, filling):
    required = set(PLACEHOLDER_PATTERN.findall(template))
    missing = required - set(filling.keys())
    if missing:
        raise LanguageModelError(f"Missing placeholders for template: {sorted(missing)}")
    return template.format(**filling)


def plan_tools(state, cache_path=None):
    cache = cache_path or language_model_cache_path
    filling = {
        "catalogue": json.dumps(TOOL_CATALOGUE, indent=2),
        "rounds": constants.ASSISTANT_TOOL_ROUNDS,
        "question": state.get("question", ""),
        "category": state.get("category", ""),
        "urgency": state.get("urgency", ""),
        "consumer_category": state.get("consumer_category", ""),
        "billing_period": state.get("billing_period", ""),
    }
    prompt = build_agent_prompt(PLANNER_PROMPT, filling)
    said = ask_model_with_retry(prompt, cache_path=cache)
    data = read_answer(said, PLANNER_KEYS)
    plan = data.get("plan")
    if not isinstance(plan, list):
        raise LanguageModelError(f"Expected plan to be a list, got {type(plan).__name__}")
    return {"plan": plan, "steps": ["plan"]}


def draft_answer(state, cache_path=None):
    cache = cache_path or language_model_cache_path
    tool_calls_str = json.dumps(state.get("tool_calls", []), indent=2, default=str)
    filling = {
        "question": state.get("question", ""),
        "tool_calls": tool_calls_str,
    }
    prompt = build_agent_prompt(RESOLVER_PROMPT, filling)
    said = ask_model_with_retry(prompt, cache_path=cache)
    data = read_answer(said, RESOLVER_KEYS)
    return {"draft": str(data.get("draft", "")), "steps": ["draft"]}


def critique_draft(state, cache_path=None):
    cache = cache_path or language_model_cache_path
    tool_calls_str = json.dumps(state.get("tool_calls", []), indent=2, default=str)
    filling = {
        "question": state.get("question", ""),
        "draft": state.get("draft", ""),
        "tool_calls": tool_calls_str,
    }
    prompt = build_agent_prompt(CRITIC_PROMPT, filling)
    said = ask_model_with_retry(prompt, cache_path=cache)
    data = read_answer(said, CRITIC_KEYS)
    veto_flag = bool(data.get("veto", False))
    veto_text = str(data.get("reason", "")).strip() if veto_flag else ""
    return {"veto": veto_text, "steps": ["critique"]}


def write_response(state, cache_path=None):
    cache = cache_path or language_model_cache_path
    filling = {
        "question": state.get("question", ""),
        "draft": state.get("draft", ""),
    }
    prompt = build_agent_prompt(RESPONDER_PROMPT, filling)
    said = ask_model_with_retry(prompt, cache_path=cache)
    data = read_answer(said, RESPONDER_KEYS)
    return {"answer": str(data.get("reply", "")).strip(), "finished": True, "steps": ["respond"]}


if __name__ == "__main__":
    print(f"planner answers with {PLANNER_KEYS} and is told {sorted(PLACEHOLDER_PATTERN.findall(PLANNER_PROMPT))}")
    print(f"resolver answers with {RESOLVER_KEYS} and is told {sorted(PLACEHOLDER_PATTERN.findall(RESOLVER_PROMPT))}")
    print("the planner may name these 4 tools and no others, and is given the argument names each one takes:")
    for name, info in TOOL_CATALOGUE.items():
        print(name, info["does"][:60], "takes", sorted(info["arguments"].keys()))
    print("the resolver is handed what the tools returned, and nothing else")
