"""
Semantic product search using Voyage AI embeddings + LanceDB.
"""

import json
import os
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa
import voyageai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRODUCTS_PATH = Path("/home/user/myproject/products.json")
LANCEDB_PATH = "/home/user/myproject/lancedb_data"
VECTOR_DIM = 1024
VOYAGE_MODEL = "voyage-3"


def _get_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name!r} is not set.")
    return value


def _table_name() -> str:
    run_id = _get_env("ZEALT_RUN_ID")
    return f"products_{run_id}"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _voyage_client() -> voyageai.Client:
    api_key = _get_env("VOYAGE_API_KEY")
    return voyageai.Client(api_key=api_key)


def _embed_documents(texts: list[str]) -> list[list[float]]:
    client = _voyage_client()
    result = client.embed(texts, model=VOYAGE_MODEL, input_type="document")
    return result.embeddings


def _embed_query(text: str) -> list[float]:
    client = _voyage_client()
    result = client.embed([text], model=VOYAGE_MODEL, input_type="query")
    return result.embeddings[0]


# ---------------------------------------------------------------------------
# LanceDB table management
# ---------------------------------------------------------------------------

def _build_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("description", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    ])


def _get_or_create_table(db: lancedb.DBConnection) -> lancedb.table.Table:
    """
    Return the existing products table or create and populate it from
    the products.json catalogue. Idempotent across calls.
    """
    table_name = _table_name()

    existing = db.table_names()
    if table_name in existing:
        return db.open_table(table_name)

    # Load catalogue
    with open(PRODUCTS_PATH) as f:
        products = json.load(f)

    descriptions = [p["description"] for p in products]

    # Embed in a single batched call (60 items — well within rate limits)
    print(f"[solution] Embedding {len(products)} products with Voyage AI…")
    vectors = _embed_documents(descriptions)

    # Build PyArrow table
    schema = _build_schema()
    pa_table = pa.table(
        {
            "id": [p["id"] for p in products],
            "description": descriptions,
            "vector": vectors,
        },
        schema=schema,
    )

    table = db.create_table(table_name, data=pa_table, schema=schema)
    print(f"[solution] Created LanceDB table '{table_name}' with {len(products)} rows.")
    return table


def _get_db() -> lancedb.DBConnection:
    return lancedb.connect(LANCEDB_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(query: str, k: int) -> list[dict]:
    """
    Embed *query* via Voyage AI (input_type='query') and return the top-k
    most similar products from LanceDB, ordered by relevance (rank 1 first).

    Each result dict contains at least 'id' and 'description'.
    """
    db = _get_db()
    table = _get_or_create_table(db)

    query_vec = _embed_query(query)

    results = (
        table.search(query_vec)
        .limit(k)
        .to_list()
    )

    return [{"id": row["id"], "description": row["description"]} for row in results]
