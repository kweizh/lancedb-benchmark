"""Semantic search over news headlines using Jina Embeddings v3 + LanceDB."""

import json
import os
from pathlib import Path

import httpx
import lancedb
import pyarrow as pa

PROJECT_DIR = Path(__file__).parent
HEADLINES_PATH = PROJECT_DIR / "headlines.json"
LANCEDB_DIR = PROJECT_DIR / "lancedb_data"


def _get_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(f"Environment variable {key} is not set")
    return value


def _table_name() -> str:
    return f"headlines_{_get_env('ZEALT_RUN_ID')}"


def _jina_embed(texts: list[str], task: str) -> list[list[float]]:
    """Call the Jina embeddings API and return a list of embedding vectors."""
    api_key = _get_env("JINA_API_KEY")
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "jina-embeddings-v3",
        "task": task,
        "input": texts,
    }
    response = httpx.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()
    # Sort by index to preserve order
    embeddings = sorted(data["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in embeddings]


def build_index() -> None:
    """Embed all headlines and store them in a LanceDB table."""
    # Load headlines
    with open(HEADLINES_PATH, "r") as f:
        headlines = json.load(f)

    texts = [h["headline"] for h in headlines]
    ids = [h["id"] for h in headlines]
    topics = [h["topic"] for h in headlines]

    # Embed all headlines in a single batch call
    embeddings = _jina_embed(texts, task="retrieval.passage")

    # Determine vector dimension from the API response
    dim = len(embeddings[0])

    # Build PyArrow table
    vector_type = pa.list_(pa.float32(), dim)
    flat_values = pa.array([v for emb in embeddings for v in emb], type=pa.float32())
    vector_col = pa.FixedSizeListArray.from_arrays(flat_values, dim)

    table = pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "headline": pa.array(texts, type=pa.utf8()),
            "topic": pa.array(topics, type=pa.utf8()),
            "vector": vector_col,
        }
    )

    # Create / overwrite the LanceDB table
    db = lancedb.connect(str(LANCEDB_DIR))
    tbl_name = _table_name()
    db.create_table(tbl_name, table, mode="overwrite")


def search(query: str, k: int = 5, task: str = "retrieval.query") -> list[dict]:
    """Embed *query* and return the top-*k* matching headlines."""
    query_embedding = _jina_embed([query], task=task)[0]

    db = lancedb.connect(str(LANCEDB_DIR))
    tbl_name = _table_name()
    tbl = db.open_table(tbl_name)

    results = tbl.search(query_embedding).limit(k).to_list()

    return [
        {
            "id": row["id"],
            "headline": row["headline"],
            "topic": row["topic"],
        }
        for row in results
    ]