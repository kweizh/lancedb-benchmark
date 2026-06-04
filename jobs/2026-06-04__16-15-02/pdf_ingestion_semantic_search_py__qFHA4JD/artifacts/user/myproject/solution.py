#!/usr/bin/env python3
"""Semantic search over PDF chunks stored in LanceDB with OpenAI embeddings."""

import os

import lancedb
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration – must match ingest.py
# ---------------------------------------------------------------------------
LANCEDB_URI = "/home/user/myproject/lancedb/"
TABLE_NAME = f"pdf_chunks_{os.environ['ZEALT_RUN_ID']}"
EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)


def _embed_query(query: str) -> list[float]:
    """Embed a single query string using the same model as ingestion."""
    response = _client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def search(query: str, k: int = 5) -> list[dict]:
    """
    Return the top-k most semantically relevant chunks for *query*.

    Each result dict contains:
        - doc_id  (str): the PDF file name without extension
        - page    (int): 1-based page number
        - snippet (str): a short excerpt of the chunk text
    """
    query_embedding = _embed_query(query)

    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table(TABLE_NAME)

    results = table.search(query_embedding).limit(k).to_list()

    output: list[dict] = []
    for row in results:
        text: str = row["text"]
        snippet = text[:200] + ("..." if len(text) > 200 else "")
        output.append(
            {
                "doc_id": row["doc_id"],
                "page": int(row["page"]),
                "snippet": snippet,
            }
        )

    return output


if __name__ == "__main__":
    # Quick smoke test
    results = search("wildlife migration", k=3)
    for r in results:
        print(r)