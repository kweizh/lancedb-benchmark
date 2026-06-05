#!/usr/bin/env python3
"""Markdown docs repo indexer with LanceDB and OpenAI embeddings."""

import os
import re
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCS_DIR = Path("/app/docs")
LANCEDB_DIR = Path("/home/user/myproject/lancedb")
EMBEDDING_MODEL = "text-embedding-3-small"
TABLE_NAME_PREFIX = "docs_sections"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_name() -> str:
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"{TABLE_NAME_PREFIX}_{run_id}"


def _get_openai_client() -> OpenAI:
    return OpenAI()  # reads OPENAI_API_KEY and optional OPENAI_BASE_URL


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI and return embedding vectors."""
    client = _get_openai_client()
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    # Sort by index to guarantee order matches input
    sorted_data = sorted(response.data, key=lambda d: d.index)
    return [d.embedding for d in sorted_data]


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


def parse_markdown(filepath: Path) -> list[dict]:
    """Parse a markdown file into sections.

    Returns a list of dicts, one per ## section, each containing:
      - repo_path: path relative to DOCS_DIR
      - doc_title: the top-level # Title (without #)
      - section_title: the ## Section header (without ##)
      - content: the body text of that section
    """
    text = filepath.read_text(encoding="utf-8")
    relative_path = str(filepath.relative_to(DOCS_DIR))

    # Extract the top-level title (# Title)
    title_match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else ""

    # Split by ## headers
    # Pattern: a line starting with ## followed by the section title,
    # then everything until the next ## or end of file.
    section_pattern = re.compile(
        r"^##\s+(.+?)$\n(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    sections = []
    for match in section_pattern.finditer(text):
        section_title = match.group(1).strip()
        content = match.group(2).strip()
        sections.append(
            {
                "repo_path": relative_path,
                "doc_title": doc_title,
                "section_title": section_title,
                "content": content,
            }
        )

    return sections


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def build_index() -> None:
    """Walk the docs directory, parse files, embed sections, and store in LanceDB."""
    # Collect all sections from all markdown files
    all_sections: list[dict] = []
    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        sections = parse_markdown(md_file)
        all_sections.extend(sections)

    if not all_sections:
        print("No sections found. Nothing to index.")
        return

    # Prepare texts for embedding: concatenate section title + content
    texts_to_embed = [
        f"{s['section_title']}\n{s['content']}" for s in all_sections
    ]

    # Compute embeddings
    print(f"Embedding {len(texts_to_embed)} sections with {EMBEDDING_MODEL}...")
    embeddings = _embed_texts(texts_to_embed)

    # Build table data
    table_data = []
    for s, emb in zip(all_sections, embeddings):
        table_data.append(
            {
                "repo_path": s["repo_path"],
                "doc_title": s["doc_title"],
                "section_title": s["section_title"],
                "content": s["content"],
                "embedding": emb,
            }
        )

    # Connect to LanceDB and create/overwrite the table
    db = lancedb.connect(str(LANCEDB_DIR))
    table_name = _table_name()

    # Drop existing table if it exists to ensure idempotency
    existing_tables = db.table_names()
    if table_name in existing_tables:
        db.drop_table(table_name)

    db.create_table(table_name, table_data)
    print(f"Indexed {len(table_data)} sections into table '{table_name}'.")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search(query: str, k: int = 5) -> list[dict]:
    """Search for the most relevant sections matching the query.

    Args:
        query: Natural language query string.
        k: Number of results to return.

    Returns:
        A list of dicts with keys: repo_path, doc_title, section_title, score.
        Ordered from most to least relevant.
    """
    # Embed the query
    query_embedding = _embed_texts([query])[0]

    # Open the LanceDB table
    db = lancedb.connect(str(LANCEDB_DIR))
    table_name = _table_name()
    table = db.open_table(table_name)

    # Perform vector search
    results = (
        table.search(query_embedding)
        .limit(k)
        .to_pandas()
    )

    # Convert to list of dicts
    output = []
    for _, row in results.iterrows():
        output.append(
            {
                "repo_path": row["repo_path"],
                "doc_title": row["doc_title"],
                "section_title": row["section_title"],
                "score": float(row["_distance"]),
            }
        )

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_index()