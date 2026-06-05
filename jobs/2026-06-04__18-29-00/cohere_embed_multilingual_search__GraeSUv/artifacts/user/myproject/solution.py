"""Cross-lingual semantic search using Cohere multilingual embeddings and LanceDB."""

import json
import os
from pathlib import Path

import cohere
import lancedb
import pyarrow as pa

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORPUS_PATH = Path(__file__).resolve().parent / "corpus.json"
LANCEDB_DIR = Path(__file__).resolve().parent / "lancedb_data"
MODEL = "embed-multilingual-v3.0"
EMBEDDING_DIM = 1024

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_table_name() -> str:
    """Return the run-scoped LanceDB table name."""
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"multilingual_{run_id}"


def _get_client() -> cohere.ClientV2:
    """Return a Cohere client using the env-var API key."""
    api_key = os.environ["COHERE_API_KEY"]
    return cohere.ClientV2(api_key=api_key)


def _embed_texts(client: cohere.ClientV2, texts: list[str], input_type: str) -> list[list[float]]:
    """Embed a list of texts using Cohere's multilingual model.

    Handles batching (Cohere limits to 96 texts per call) and returns
    a flat list of float32-castable embedding vectors.
    """
    all_embeddings: list[list[float]] = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.v2.embed(
            texts=batch,
            model=MODEL,
            input_type=input_type,
            embedding_types=["float"],
        )
        # v2 SDK: embeddings are in response.embeddings.float
        batch_embeddings = response.embeddings.float
        for vec in batch_embeddings:
            all_embeddings.append([float(x) for x in vec])
    return all_embeddings


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

_db = None
_table = None


def build_index() -> None:
    """Read corpus.json, embed every text, and persist rows into LanceDB."""
    global _db, _table

    # Load corpus
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    texts = [row["text"] for row in corpus]

    # Embed documents
    client = _get_client()
    embeddings = _embed_texts(client, texts, input_type="search_document")

    # Build PyArrow table with explicit schema
    table_name = _get_table_name()
    schema = pa.schema(
        [
            pa.field("concept_id", pa.int64()),
            pa.field("language", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ]
    )

    concept_ids = [row["concept_id"] for row in corpus]
    languages = [row["language"] for row in corpus]

    # Convert embeddings to fixed-size list arrays for LanceDB
    vectors = pa.FixedSizeListArray.from_arrays(
        pa.array([v for vec in embeddings for v in vec], type=pa.float32()),
        EMBEDDING_DIM,
    )

    pa_table = pa.table(
        {
            "concept_id": pa.array(concept_ids, type=pa.int64()),
            "language": pa.array(languages, type=pa.string()),
            "text": pa.array(texts, type=pa.string()),
            "vector": vectors,
        },
        schema=schema,
    )

    # Connect / create LanceDB
    db = lancedb.connect(str(LANCEDB_DIR))

    # Drop existing table if present so we can rebuild cleanly
    existing_tables = db.table_names()
    if table_name in existing_tables:
        db.drop_table(table_name)

    table = db.create_table(table_name, pa_table)
    _db = db
    _table = table


def _ensure_index() -> None:
    """Lazily build the index if it hasn't been built yet."""
    global _db, _table
    if _table is not None:
        return

    table_name = _get_table_name()
    db = lancedb.connect(str(LANCEDB_DIR))

    # Check if the table already exists on disk
    try:
        existing = db.table_names()
    except Exception:
        existing = []

    if table_name in existing:
        _db = db
        _table = db.open_table(table_name)
    else:
        build_index()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def cross_lingual_search(query: str, k: int = 3) -> list[dict]:
    """Search the multilingual corpus for the *query* and return top-k results.

    The query may be in any language; results come from all three languages
    (en, es, fr) sorted by ascending distance.
    """
    _ensure_index()

    client = _get_client()
    query_embeddings = _embed_texts(client, [query], input_type="search_query")
    query_vector = query_embeddings[0]

    results = _table.search(query_vector).metric("cosine").limit(k).to_arrow()

    output: list[dict] = []
    for i in range(results.num_rows):
        output.append(
            {
                "concept_id": int(results.column("concept_id")[i].as_py()),
                "language": str(results.column("language")[i].as_py()),
                "text": str(results.column("text")[i].as_py()),
            }
        )
    return output