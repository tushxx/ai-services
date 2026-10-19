import json
from pathlib import Path

from config import constants

CHUNK_FIELDS = ("chunk_id", "text") + constants.STORED_CHUNK_FIELDS


def split_into_chunks(text, size=constants.CHUNK_CHARACTERS, overlap=constants.CHUNK_OVERLAP_CHARACTERS):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    if not text:
        return []

    step = size - overlap
    pieces = []
    start = 0
    text_len = len(text)

    while start < text_len:
        piece = text[start : start + size]
        if piece.strip():
            pieces.append(piece)
        if len(piece) < size:
            break
        start += step

    return pieces


def chunk_metadata(order, chunk_text, position):
    sec = ""
    for s in order.get("sections", []):
        if s in chunk_text:
            sec = s
            break

    return {
        "chunk_id": f"{order['order_id']}-{position:03d}",
        "text": chunk_text,
        "order_id": order.get("order_id", ""),
        "title": order.get("title", ""),
        "consumer_category": order.get("consumer_category", ""),
        "effective_from": order.get("effective_from", ""),
        "effective_to": order.get("effective_to", ""),
        "section": sec,
    }


def write_chunks(orders, chunks_path=constants.TARIFF_CHUNKS_PATH):
    all_chunks = []
    without_a_section = 0

    for order in orders:
        text = order.get("text", "")
        pieces = split_into_chunks(text)
        current_section = ""
        for pos, piece in enumerate(pieces, start=1):
            meta = chunk_metadata(order, piece, pos)
            if meta["section"]:
                current_section = meta["section"]
            else:
                meta["section"] = current_section

            if not meta["section"]:
                without_a_section += 1

            all_chunks.append(meta)

    out = Path(chunks_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")

    return {
        "orders": len(orders),
        "chunks": len(all_chunks),
        "without_a_section": without_a_section,
    }


if __name__ == "__main__":
    from metermind.retrieval.document_loader import load_tariff_orders

    orders = load_tariff_orders()
    res = write_chunks(orders)
    print(res)
