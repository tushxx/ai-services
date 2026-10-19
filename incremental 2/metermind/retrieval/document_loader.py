import json
from pathlib import Path

from config import constants


class UnreadableOrder(Exception):
    """Raised where an order cannot be read."""
    pass


def read_pdf_text(pdf_path):
    p = Path(pdf_path)
    if not p.is_file():
        raise UnreadableOrder(f"Order file does not exist: {pdf_path}")
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text() or ""
            pages_text.append(t)
        text = "\n".join(pages_text).strip()
    except Exception as e:
        if isinstance(e, UnreadableOrder):
            raise
        raise UnreadableOrder(f"Could not read PDF {pdf_path}: {e}")

    if not text:
        raise UnreadableOrder(f"No text extracted from order: {pdf_path}")
    return text


def read_manifest(manifest_path=constants.TARIFF_ORDERS_MANIFEST_PATH):
    p = Path(manifest_path)
    if not p.is_file():
        raise UnreadableOrder(f"Manifest file does not exist: {manifest_path}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise UnreadableOrder(f"Could not parse manifest at {manifest_path}: {e}")


def load_tariff_orders(
    orders_directory=constants.TARIFF_ORDERS_DIRECTORY,
    manifest_path=constants.TARIFF_ORDERS_MANIFEST_PATH,
):
    manifest = read_manifest(manifest_path)
    orders_dir = Path(orders_directory)
    loaded = []
    for order in manifest:
        order_id = order["order_id"]
        pdf_path = orders_dir / f"{order_id}.pdf"
        text = read_pdf_text(pdf_path)
        data = dict(order)
        data["text"] = text
        loaded.append(data)
    return loaded


if __name__ == "__main__":
    orders = load_tariff_orders()
    total_chars = sum(len(o["text"]) for o in orders)
    print(f"{len(orders)} orders, {total_chars} characters of text")
