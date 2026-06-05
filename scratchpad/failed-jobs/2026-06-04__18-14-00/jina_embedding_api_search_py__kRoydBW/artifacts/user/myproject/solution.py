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
# Paths & constants
# ---------------------------------------------------------------------------

_PROJECT_DIR = Path(__file__).parent
_HEADLINES_PATH = _PROJECT_DIR / "headlines.json"
_LANCEDB_DIR = _PROJECT_DIR / "lancedb_data"

_JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"
_JINA_MODEL = "jina-embeddings-v3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_key() -> str:
    key = os.environ.get("JINA_API_KEY", "")
    if not key:
        raise EnvironmentError("JINA_API_KEY environment variable is not set.")
    return key


def _run_id() -> str:
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return run_id


def _table_name() -> str:
    return f"headlines_{_run_id()}"


def _embed(texts: list[str], task: str) -> list[list[float]]:
    """Call the Jina embeddings API and return a list of embedding vectors."""
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": _JINA_MODEL,
        "task": task,
        "input": texts,
    }
    response = httpx.post(_JINA_ENDPOINT, headers=headers, json=body, timeout=120.0)
    response.raise_for_status()
    data = response.json()
    # The response follows the OpenAI-compatible shape:
    # { "data": [ { "index": 0, "embedding": [...] }, ... ] }
    items = data["data"]
    # Sort by index to preserve order (API guarantees order but let's be safe)
    items_sorted = sorted(items, key=lambda x: x["index"])
    return [item["embedding"] for item in items_sorted]


def _load_headlines() -> list[dict]:
    with open(_HEADLINES_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index() -> None:
    """Embed all headlines and store them in a LanceDB table."""
    headlines = _load_headlines()
    texts = [h["headline"] for h in headlines]

    print(f"Embedding {len(texts)} headlines with task='retrieval.passage' …")
    embeddings = _embed(texts, task="retrieval.passage")
    dim = len(embeddings[0])
    print(f"Embedding dimension: {dim}")

    # Build PyArrow table
    ids = pa.array([h["id"] for h in headlines], type=pa.int64())
    headline_arr = pa.array([h["headline"] for h in headlines], type=pa.string())
    topic_arr = pa.array([h["topic"] for h in headlines], type=pa.string())
    vector_type = pa.list_(pa.float32(), dim)
    # Cast each embedding to float32
    vectors = pa.array(
        [emb for emb in embeddings],
        type=vector_type,
    )

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("headline", pa.string()),
            pa.field("topic", pa.string()),
            pa.field("vector", vector_type),
        ]
    )
    table = pa.table(
        {"id": ids, "headline": headline_arr, "topic": topic_arr, "vector": vectors},
        schema=schema,
    )

    # Connect to / create the LanceDB database
    _LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(_LANCEDB_DIR))

    tbl_name = _table_name()
    # Overwrite if exists
    existing = db.table_names()
    if tbl_name in existing:
        db.drop_table(tbl_name)
        print(f"Dropped existing table '{tbl_name}'.")

    db.create_table(tbl_name, data=table)
    print(f"Created LanceDB table '{tbl_name}' with {len(headlines)} rows.")


def search(query: str, k: int = 5, task: str = "retrieval.query") -> list[dict]:
    """Embed `query` and return the top-k matching headlines."""
    query_embedding = _embed([query], task=task)[0]

    db = lancedb.connect(str(_LANCEDB_DIR))
    tbl_name = _table_name()
    tbl = db.open_table(tbl_name)

    results = tbl.search(query_embedding).limit(k).to_list()

    output: list[dict] = []
    for row in results:
        output.append(
            {
                "id": int(row["id"]),
                "headline": row["headline"],
                "topic": row["topic"],
            }
        )
    return output
