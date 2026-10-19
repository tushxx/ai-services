import csv
from pathlib import Path

from config import constants

FILTERED_ANSWER_FIELDS = (
    "question_id",
    "question",
    "consumer_category",
    "billing_period",
    "order_in_force",
    "answer",
    "citations",
    "refused_because",
)


class NoOrderInForce(Exception):
    """Raised where no order covered the period asked about."""
    pass


def order_in_force(manifest, consumer_category, billing_period):
    norm_cat = str(consumer_category).strip().lower()
    period_str = str(billing_period).strip()
    period_date = f"{period_str}-01" if len(period_str) == 7 else period_str

    for order in manifest:
        if order.get("consumer_category", "").strip().lower() == norm_cat:
            ef_from = str(order.get("effective_from", "")).strip()
            ef_to = str(order.get("effective_to", "")).strip()
            if ef_from <= period_date <= ef_to:
                return order

    raise NoOrderInForce(
        f"no {consumer_category} order was in force in {billing_period}; answering from an order that had ceased to have effect would be worse than not answering"
    )


def filtered_search(
    collection,
    question,
    embedder,
    consumer_category,
    billing_period,
    manifest=None,
    result_count=constants.SEARCH_RESULT_COUNT,
):
    if manifest is None:
        from metermind.retrieval.document_loader import read_manifest
        manifest = read_manifest()

    order = order_in_force(manifest, consumer_category, billing_period)
    from metermind.retrieval.vector_store import search_with_filter

    conditions = {"order_id": order["order_id"]}
    return search_with_filter(
        collection, question, embedder, conditions=conditions, result_count=result_count
    )


def write_filtered_answers(
    questions,
    collection,
    embedder,
    answers_path,
    manifest=None,
    cache_path=constants.LANGUAGE_MODEL_CACHE_PATH,
):
    if manifest is None:
        from metermind.retrieval.document_loader import read_manifest
        manifest = read_manifest()

    from metermind.retrieval.grounded_answer import answer_from_chunks

    rows = []
    answered_count = 0
    refused_count = 0

    for q in questions:
        qid = q.get("question_id", "")
        q_text = q.get("question", "")
        cat = q.get("consumer_category", "")
        period = q.get("billing_period", "")
        try:
            order = order_in_force(manifest, cat, period)
            order_id = order["order_id"]
            chunks = filtered_search(
                collection,
                q_text,
                embedder,
                cat,
                period,
                manifest=manifest,
            )
            result = answer_from_chunks(q_text, chunks, cache_path=cache_path)
            answer = result["answer"]
            citations = ";".join(result["citations"])
            refused_because = ""
            answered_count += 1
        except Exception as e:
            order_id = locals().get("order_id", "")
            answer = ""
            citations = ""
            refused_because = str(e)
            refused_count += 1

        rows.append({
            "question_id": qid,
            "question": q_text,
            "consumer_category": cat,
            "billing_period": period,
            "order_in_force": order_id,
            "answer": answer,
            "citations": citations,
            "refused_because": refused_because,
        })

    out = Path(answers_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FILTERED_ANSWER_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "questions": len(questions),
        "answered": answered_count,
        "refused": refused_count,
    }


if __name__ == "__main__":
    from metermind.retrieval.document_loader import read_manifest
    from metermind.retrieval.vector_store import (
        load_embedder,
        open_tariff_store,
        search_tariff,
    )

    try:
        manifest = read_manifest()
        collection = open_tariff_store()
        embedder = load_embedder()
        q = "what is the rate for the first hundred units"
        print(
            "nearest, any order at all :",
            [one["order_id"] for one in search_tariff(collection, q, embedder)],
        )
        print(
            "nearest, domestic 2024-09 :",
            [
                one["order_id"]
                for one in filtered_search(
                    collection, q, embedder, "domestic", "2024-09", manifest
                )
            ],
        )
        try:
            filtered_search(collection, q, embedder, "domestic", "2019-01", manifest)
        except NoOrderInForce as e:
            print("asked about 2019-01 :", e)
    except Exception as e:
        print(f"Skipping main execution: {e}")
