"""
solution.py — Semantic Search over ingested PDF chunks.

Exposes:
    search(query: str, k: int) -> list[dict]

Each result dict contains:
    doc_id  (str)  — PDF stem, e.g. "alpha"
    page    (int)  — 1-based page number
    snippet (str)  — short excerpt of the matched chunk
"""

import os
import lancedb
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration (must match ingest.py)
# ---------------------------------------------------------------------------
LANCEDB_DIR = "/home/user/myproject/lancedb"
EMBEDDING_MODEL = "text-embedding-3-small"
SNIPPET_MAX_LEN = 200  # characters

ZEALT_RUN_ID = os.environ.get("ZEALT_RUN_ID", "default")
TABLE_NAME = f"pdf_chunks_{ZEALT_RUN_ID}"

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialised on first call)
# ---------------------------------------------------------------------------
_client: OpenAI | None = None
_table = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
        else:
            _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _get_table():
    global _table
    if _table is None:
        db = lancedb.connect(LANCEDB_DIR)
        _table = db.open_table(TABLE_NAME)
    return _table


def _make_snippet(text: str, max_len: int = SNIPPET_MAX_LEN) -> str:
    """Return a short excerpt of *text*, stripped of excess whitespace."""
    text = " ".join(text.split())  # collapse whitespace / newlines
    if len(text) <= max_len:
        return text
    # Trim at a word boundary
    trimmed = text[:max_len]
    last_space = trimmed.rfind(" ")
    if last_space > max_len // 2:
        trimmed = trimmed[:last_space]
    return trimmed + "…"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(query: str, k: int) -> list[dict]:
    """
    Embed *query* with OpenAI and return the top-*k* most relevant chunks
    from the LanceDB table, ordered by descending relevance.

    Each result is a dict with keys: doc_id (str), page (int), snippet (str).
    """
    client = _get_client()
    table = _get_table()

    # Embed the query
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_embedding = response.data[0].embedding

    # Vector search in LanceDB
    results = (
        table.search(query_embedding)
        .limit(k)
        .select(["doc_id", "page", "text"])
        .to_list()
    )

    output: list[dict] = []
    for row in results:
        output.append({
            "doc_id": str(row["doc_id"]),
            "page": int(row["page"]),
            "snippet": _make_snippet(row["text"]),
        })

    return output


# ---------------------------------------------------------------------------
# Quick smoke-test when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    query = "coral reefs and marine biodiversity"
    print(f"Query: {query!r}\n")
    hits = search(query, k=3)
    for i, hit in enumerate(hits, 1):
        print(f"  {i}. [{hit['doc_id']} p{hit['page']}] {hit['snippet']}")
