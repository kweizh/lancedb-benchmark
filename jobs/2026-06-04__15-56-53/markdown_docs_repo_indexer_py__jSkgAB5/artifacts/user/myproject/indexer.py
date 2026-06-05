"""
Markdown Docs Repo Indexer
--------------------------
Walks /app/docs/, splits each Markdown file into per-## section chunks,
embeds every chunk with OpenAI text-embedding-3-small, and persists the
rows into a LanceDB table.

Exposes:
    search(query: str, k: int) -> list[dict]
        Returns the top-k most relevant sections ordered by relevance.
        Each dict has: repo_path, doc_title, section_title, score.

Running this file as a script performs (idempotent) indexing:
    python3 /home/user/myproject/indexer.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCS_DIR = Path("/app/docs")
LANCEDB_DIR = "/home/user/myproject/lancedb"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # dimension for text-embedding-3-small

_run_id = os.environ.get("ZEALT_RUN_ID", "default")
TABLE_NAME = f"docs_sections_{_run_id}"

# ---------------------------------------------------------------------------
# OpenAI client (reads OPENAI_API_KEY / OPENAI_BASE_URL from environment)
# ---------------------------------------------------------------------------

_openai_kwargs: dict[str, Any] = {}
if os.environ.get("OPENAI_BASE_URL"):
    _openai_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

_client = OpenAI(**_openai_kwargs)


def embed(texts: list[str]) -> list[list[float]]:
    """Return embeddings for a batch of texts using the real OpenAI API."""
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # Sort by index to guarantee order matches input
    results = sorted(response.data, key=lambda e: e.index)
    return [r.embedding for r in results]


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

def parse_markdown(text: str) -> tuple[str, list[dict]]:
    """
    Parse a Markdown document and return:
        doc_title  - text of the first # heading (stripped)
        sections   - list of dicts with keys: section_title, content
    """
    lines = text.splitlines()
    doc_title = ""
    sections: list[dict] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def flush():
        if current_title is not None:
            sections.append(
                {
                    "section_title": current_title,
                    "content": "\n".join(current_lines).strip(),
                }
            )

    for line in lines:
        h1 = re.match(r"^#\s+(.+)$", line)
        h2 = re.match(r"^##\s+(.+)$", line)

        if h1 and not doc_title:
            doc_title = h1.group(1).strip()
            continue

        if h2:
            flush()
            current_title = h2.group(1).strip()
            current_lines = []
            continue

        if current_title is not None:
            current_lines.append(line)

    flush()
    return doc_title, sections


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _build_rows() -> list[dict]:
    """Walk DOCS_DIR and return all section rows (without embeddings yet)."""
    rows = []
    for md_path in sorted(DOCS_DIR.rglob("*.md")):
        repo_path = str(md_path.relative_to(DOCS_DIR))
        text = md_path.read_text(encoding="utf-8")
        doc_title, sections = parse_markdown(text)
        for sec in sections:
            rows.append(
                {
                    "repo_path": repo_path,
                    "doc_title": doc_title,
                    "section_title": sec["section_title"],
                    "content": sec["content"],
                    # text sent to the embedding model
                    "_embed_text": f"{sec['section_title']}\n\n{sec['content']}",
                }
            )
    return rows


def index():
    """
    Build embeddings for all sections and persist them to LanceDB.
    Idempotent: if the table already exists with the same number of rows,
    indexing is skipped.
    """
    db = lancedb.connect(LANCEDB_DIR)
    existing_tables = db.table_names()

    rows = _build_rows()

    if TABLE_NAME in existing_tables:
        tbl = db.open_table(TABLE_NAME)
        row_count = tbl.count_rows()
        if row_count == len(rows):
            print(
                f"[indexer] Table '{TABLE_NAME}' already exists with "
                f"{row_count} rows — skipping re-indexing."
            )
            return
        else:
            print(
                f"[indexer] Table '{TABLE_NAME}' exists but has {row_count} rows "
                f"(expected {len(rows)}) — dropping and re-indexing."
            )
            db.drop_table(TABLE_NAME)

    # Embed in one batch to minimise API round-trips
    print(f"[indexer] Embedding {len(rows)} sections …")
    embed_texts = [r["_embed_text"] for r in rows]
    embeddings = embed(embed_texts)

    # Build PyArrow table
    schema = pa.schema(
        [
            pa.field("repo_path", pa.string()),
            pa.field("doc_title", pa.string()),
            pa.field("section_title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ]
    )

    pa_rows = {
        "repo_path": [r["repo_path"] for r in rows],
        "doc_title": [r["doc_title"] for r in rows],
        "section_title": [r["section_title"] for r in rows],
        "content": [r["content"] for r in rows],
        "embedding": [list(map(float, v)) for v in embeddings],
    }

    table = pa.table(pa_rows, schema=schema)
    db.create_table(TABLE_NAME, data=table)
    print(f"[indexer] Created table '{TABLE_NAME}' with {len(rows)} rows.")


# ---------------------------------------------------------------------------
# Search API
# ---------------------------------------------------------------------------

def search(query: str, k: int) -> list[dict]:
    """
    Embed *query* and return the top-k most relevant sections.

    Returns a list of dicts with keys:
        repo_path     (str)
        doc_title     (str)
        section_title (str)
        score         (float)  — cosine similarity (higher = more relevant)
    """
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(TABLE_NAME)

    query_vec = embed([query])[0]

    results = (
        tbl.search(query_vec, vector_column_name="embedding")
        .metric("cosine")
        .limit(k)
        .select(["repo_path", "doc_title", "section_title", "_distance"])
        .to_list()
    )

    output = []
    for row in results:
        # LanceDB returns cosine *distance* (0 = identical, 2 = opposite).
        # Convert to similarity: score = 1 - distance
        distance = row.get("_distance", 0.0)
        score = float(1.0 - distance)
        output.append(
            {
                "repo_path": row["repo_path"],
                "doc_title": row["doc_title"],
                "section_title": row["section_title"],
                "score": score,
            }
        )

    # Results come back sorted by distance ascending (best first), so
    # after converting to similarity they are already sorted descending.
    return output


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    index()
