"""
Semantic product search using Voyage AI embeddings + LanceDB.
"""

import os
import json
import lancedb
import pyarrow as pa
import voyageai

# ── Configuration ────────────────────────────────────────────────────────────
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
ZEALT_RUN_ID   = os.environ["ZEALT_RUN_ID"]

PRODUCTS_PATH  = "/home/user/myproject/products.json"
DB_PATH        = "/home/user/myproject/lancedb_data"
TABLE_NAME     = f"products_{ZEALT_RUN_ID}"
EMBED_MODEL    = "voyage-3"
VECTOR_DIM     = 1024

# ── Lazy singletons ──────────────────────────────────────────────────────────
_db    = None
_table = None
_client = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=VOYAGE_API_KEY)
    return _client


def _get_db():
    global _db
    if _db is None:
        os.makedirs(DB_PATH, exist_ok=True)
        _db = lancedb.connect(DB_PATH)
    return _db


def _build_table():
    """
    Embed all product descriptions and persist them to LanceDB.
    Called once; subsequent runs reuse the existing table.
    """
    db = _get_db()

    existing = db.table_names()
    if TABLE_NAME in existing:
        return db.open_table(TABLE_NAME)

    # Load catalogue
    with open(PRODUCTS_PATH, "r") as fh:
        products = json.load(fh)

    descriptions = [p["description"] for p in products]

    # Embed in a single batched call (60 items is well within Voyage limits)
    client = _get_client()
    result = client.embed(descriptions, model=EMBED_MODEL, input_type="document")
    embeddings = result.embeddings  # list of list[float], each len 1024

    # Build PyArrow table
    schema = pa.schema([
        pa.field("id",          pa.string()),
        pa.field("description", pa.string()),
        pa.field("vector",      pa.list_(pa.float32(), VECTOR_DIM)),
    ])

    rows = [
        {"id": p["id"], "description": p["description"], "vector": emb}
        for p, emb in zip(products, embeddings)
    ]

    table = db.create_table(TABLE_NAME, data=rows, schema=schema)
    return table


def _get_table():
    global _table
    if _table is None:
        _table = _build_table()
    return _table


# ── Public API ───────────────────────────────────────────────────────────────

def search(query: str, k: int) -> list[dict]:
    """
    Embed *query* with Voyage AI (input_type='query') and return the top-k
    most-relevant products from LanceDB, ordered rank-1 first.

    Each result dict contains at least 'id' and 'description'.
    """
    client = _get_client()
    result = client.embed([query], model=EMBED_MODEL, input_type="query")
    query_vec = result.embeddings[0]   # list[float] of length VECTOR_DIM

    table = _get_table()
    hits  = table.search(query_vec).limit(k).to_list()

    return [{"id": h["id"], "description": h["description"]} for h in hits]


# ── Ensure table is built on module import (idempotent) ──────────────────────
_get_table()
