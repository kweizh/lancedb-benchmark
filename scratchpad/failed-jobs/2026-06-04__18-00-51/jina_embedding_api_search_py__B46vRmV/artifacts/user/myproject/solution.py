"""
Semantic search over news headlines using Jina Embeddings v3 + LanceDB.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import lancedb

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------
_HEADLINES_PATH = Path(__file__).parent / "headlines.json"
_LANCEDB_DIR = Path(__file__).parent / "lancedb_data"
_JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
_JINA_MODEL = "jina-embeddings-v3"


def _run_id() -> str:
    """Return the ZEALT_RUN_ID env var (required at runtime)."""
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return run_id


def _table_name() -> str:
    return f"headlines_{_run_id()}"


def _jina_api_key() -> str:
    key = os.environ.get("JINA_API_KEY", "")
    if not key:
        raise EnvironmentError("JINA_API_KEY environment variable is not set.")
    return key


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed(texts: list[str], task: str) -> list[list[float]]:
    """Call the Jina embeddings API and return a list of embedding vectors."""
    headers = {
        "Authorization": f"Bearer {_jina_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _JINA_MODEL,
        "task": task,
        "input": texts,
    }
    response = httpx.post(_JINA_EMBEDDINGS_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    # Response shape: {"data": [{"embedding": [...], "index": int}, ...], ...}
    # Sort by index to guarantee order matches input order.
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index() -> None:
    """
    Embed all headlines from headlines.json (task=retrieval.passage) and
    store them in a LanceDB table named `headlines_<ZEALT_RUN_ID>`.
    Overwrites the table if it already exists.
    """
    # Load headlines
    with open(_HEADLINES_PATH, "r", encoding="utf-8") as fh:
        headlines: list[dict[str, Any]] = json.load(fh)

    texts = [h["headline"] for h in headlines]

    print(f"Embedding {len(texts)} headlines via Jina API (task=retrieval.passage)…")
    embeddings = _embed(texts, task="retrieval.passage")
    dim = len(embeddings[0])
    print(f"Embedding dimension: {dim}")

    # Build PyArrow table
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("headline", pa.utf8()),
        pa.field("topic", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])

    records = []
    for h, emb in zip(headlines, embeddings):
        records.append({
            "id": int(h["id"]),
            "headline": h["headline"],
            "topic": h["topic"],
            "vector": [float(v) for v in emb],
        })

    arrow_table = pa.Table.from_pylist(records, schema=schema)

    # Write to LanceDB (overwrite existing table)
    _LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(_LANCEDB_DIR))

    tbl_name = _table_name()
    # Drop existing table if present
    existing = db.table_names()
    if tbl_name in existing:
        db.drop_table(tbl_name)

    db.create_table(tbl_name, data=arrow_table, schema=schema)
    print(f"LanceDB table '{tbl_name}' created with {len(records)} rows at {_LANCEDB_DIR}")


def search(query: str, k: int = 5, task: str = "retrieval.query") -> list[dict]:
    """
    Embed *query* (task defaults to retrieval.query) and return the top-k
    nearest neighbours from the LanceDB table as a list of dicts with keys
    `id`, `headline`, `topic`.
    """
    # Embed query
    query_vectors = _embed([query], task=task)
    query_vec = query_vectors[0]

    # Open LanceDB table
    db = lancedb.connect(str(_LANCEDB_DIR))
    tbl_name = _table_name()
    tbl = db.open_table(tbl_name)

    results = tbl.search(query_vec).limit(k).to_list()

    output = []
    for row in results:
        output.append({
            "id": int(row["id"]),
            "headline": row["headline"],
            "topic": row["topic"],
        })
    return output
