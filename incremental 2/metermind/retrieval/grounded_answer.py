import csv
from pathlib import Path
import re

from config import constants
from metermind.message_intelligence.llm_client import ask_model, read_answer

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")
ANSWER_FIELDS = (
    "question_id",
    "question",
    "answer",
    "citations",
    "chunks_handed_over",
    "refused_because",
)

ANSWER_TEMPLATE = """You are answering questions about electrical tariff orders.
Use ONLY the extracts provided below. Do not use outside knowledge.
Cite every extract you rely on using its bracketed identifier, e.g. [{sample_id}].
If the extracts do not answer the question, answer with "{unanswerable_reply}".

EXTRACTS:
{extracts}

QUESTION:
{question}

Reply with a JSON object holding:
"answer": your grounded answer quoting extract identifiers in brackets
"citations": list of extract identifiers you relied on
"""


class UngroundedAnswer(Exception):
    """Raised where an answer does not stand on what was retrieved."""
    pass


def check_citations(answer_text, chunks):
    available = {c["chunk_id"] for c in chunks}
    found = CITATION_PATTERN.findall(str(answer_text))
    valid = []
    for citation in found:
        if citation not in available:
            raise UngroundedAnswer(f"Citation {citation} was never handed over.")
        if citation not in valid:
            valid.append(citation)
    return valid


def build_answer_prompt(question, chunks):
    if not chunks:
        raise UngroundedAnswer("Nothing was retrieved; cannot build prompt with zero extracts.")

    extract_lines = []
    for c in chunks:
        line = (
            f"[{c['chunk_id']}] Order: {c.get('order_id', '')}, Section: {c.get('section', '')}, "
            f"In force: {c.get('effective_from', '')} to {c.get('effective_to', '')}\n"
            f"{c.get('text', '')}"
        )
        extract_lines.append(line)

    sample_id = chunks[0]["chunk_id"] if chunks else "CHUNK_ID"
    return ANSWER_TEMPLATE.format(
        sample_id=sample_id,
        unanswerable_reply=constants.UNANSWERABLE_REPLY,
        extracts="\n\n".join(extract_lines),
        question=question,
    )


def answer_from_chunks(question, chunks, cache_path=constants.LANGUAGE_MODEL_CACHE_PATH):
    prompt = build_answer_prompt(question, chunks)
    said = ask_model(prompt, cache_path=cache_path)
    data = read_answer(said, constants.GROUNDED_ANSWER_KEYS)

    available = {c["chunk_id"] for c in chunks}
    text_cites = check_citations(data["answer"], chunks)

    field_cites = data.get("citations", [])
    if isinstance(field_cites, str):
        field_cites = [field_cites]
    for c in field_cites:
        if c not in available:
            raise UngroundedAnswer(f"Listed citation {c} was not among handed-over extracts.")

    all_citations = list(dict.fromkeys(text_cites + field_cites))
    return {"answer": data["answer"], "citations": all_citations}


def write_answers(
    questions,
    collection,
    embedder,
    answers_path,
    cache_path=constants.LANGUAGE_MODEL_CACHE_PATH,
):
    from metermind.retrieval.vector_store import search_tariff

    rows = []
    answered_count = 0
    refused_count = 0

    for q in questions:
        qid = q.get("question_id", "")
        q_text = q.get("question", "")
        chunks = search_tariff(collection, q_text, embedder)
        chunk_ids_str = ";".join(c["chunk_id"] for c in chunks)
        try:
            res = answer_from_chunks(q_text, chunks, cache_path=cache_path)
            ans = res["answer"]
            cites = ";".join(res["citations"])
            refused_because = ""
            answered_count += 1
        except Exception as e:
            ans = ""
            cites = ""
            refused_because = str(e)
            refused_count += 1

        rows.append({
            "question_id": qid,
            "question": q_text,
            "answer": ans,
            "citations": cites,
            "chunks_handed_over": chunk_ids_str,
            "refused_because": refused_because,
        })

    out = Path(answers_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ANSWER_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "questions": len(questions),
        "answered": answered_count,
        "refused": refused_count,
    }


if __name__ == "__main__":
    import json
    from metermind.retrieval.vector_store import load_embedder, open_tariff_store

    try:
        collection = open_tariff_store()
        embedder = load_embedder()
        eval_path = Path(constants.RETRIEVAL_EVALUATION_SET_PATH)
        if eval_path.is_file():
            questions = json.loads(eval_path.read_text(encoding="utf-8"))[:5]
            res = write_answers(
                questions, collection, embedder, constants.TARIFF_ANSWERS_PATH
            )
            print(res)
    except Exception as e:
        print(f"Skipping main execution: {e}")
