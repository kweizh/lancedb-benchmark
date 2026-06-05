"""
Cross-lingual semantic search using Cohere multilingual embeddings and LanceDB.
"""

import json
import os
import numpy as np
import pyarrow as pa
import lancedb
import cohere

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus.json")
LANCEDB_DIR = os.path.join(os.path.dirname(__file__), "lancedb_data")
EMBED_MODEL = "embed-multilingual-v3.0"
VECTOR_DIM = 1024

# Read run-scoped table name from environment so concurrent runs don't collide.
_RUN_ID = os.environ.get("ZEALT_RUN_ID", "default")
TABLE_NAME = f"multilingual_{_RUN_ID}"

# ---------------------------------------------------------------------------
# Module-level singletons (lazily initialised)
# ---------------------------------------------------------------------------

_db: lancedb.DBConnection | None = None
_table = None


def _get_cohere_client() -> cohere.Client:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError("COHERE_API_KEY environment variable is not set.")
    return cohere.Client(api_key=api_key)


def _embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed a list of texts using Cohere's multilingual model.

    Handles both legacy (list-of-lists) and typed (EmbeddingsByType) response
    formats across SDK versions.
    """
    co = _get_cohere_client()

    # Process in batches of 96 (Cohere API limit per call)
    batch_size = 96
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = co.embed(
            texts=batch,
            model=EMBED_MODEL,
            input_type=input_type,
        )

        # The SDK may return either:
        #   EmbeddingsFloatsEmbedResponse  → response.embeddings is list[list[float]]
        #   EmbeddingsByTypeEmbedResponse  → response.embeddings.float_ or .float
        raw = response.embeddings
        if isinstance(raw, list):
            # Legacy / default float response
            vectors = [list(v) for v in raw]
        else:
            # Typed response – try .float_ then .float
            if hasattr(raw, "float_") and raw.float_ is not None:
                vectors = [list(v) for v in raw.float_]
            elif hasattr(raw, "float") and raw.float is not None:
                vectors = [list(v) for v in raw.float]
            else:
                raise RuntimeError(
                    f"Unexpected embeddings format returned by Cohere SDK: {type(raw)}"
                )

        all_vectors.extend(vectors)

    return all_vectors


def _get_db() -> lancedb.DBConnection:
    global _db
    if _db is None:
        os.makedirs(LANCEDB_DIR, exist_ok=True)
        _db = lancedb.connect(LANCEDB_DIR)
    return _db


def _get_or_open_table():
    """Return the LanceDB table, opening it if already created."""
    global _table
    if _table is not None:
        return _table
    db = _get_db()
    if TABLE_NAME in db.table_names():
        _table = db.open_table(TABLE_NAME)
    return _table


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_index() -> None:
    """Embed all corpus documents and persist them into LanceDB.

    This is idempotent: if the table already exists with 90 rows it is a no-op.
    """
    global _table

    db = _get_db()

    # If the table already exists and is fully populated, skip re-embedding.
    if TABLE_NAME in db.table_names():
        tbl = db.open_table(TABLE_NAME)
        if tbl.count_rows() == 90:
            _table = tbl
            return

    # Load corpus
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    texts = [row["text"] for row in corpus]
    concept_ids = [row["concept_id"] for row in corpus]
    languages = [row["language"] for row in corpus]

    # Embed all documents
    vectors = _embed_texts(texts, input_type="search_document")

    # Build PyArrow table with an explicit fixed-size list vector column
    schema = pa.schema(
        [
            pa.field("concept_id", pa.int32()),
            pa.field("language", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    )

    # Convert vectors to float32 numpy arrays then back to lists for PyArrow
    float32_vectors = [
        np.array(v, dtype=np.float32).tolist() for v in vectors
    ]

    arrow_table = pa.table(
        {
            "concept_id": pa.array(concept_ids, type=pa.int32()),
            "language": pa.array(languages, type=pa.string()),
            "text": pa.array(texts, type=pa.string()),
            "vector": pa.array(float32_vectors, type=pa.list_(pa.float32(), VECTOR_DIM)),
        },
        schema=schema,
    )

    # Create (or overwrite) the table
    _table = db.create_table(
        TABLE_NAME,
        data=arrow_table,
        mode="overwrite",
    )


def cross_lingual_search(query: str, k: int = 3) -> list[dict]:
    """Embed *query* and return the top-k closest corpus entries.

    Builds the index automatically on first call if it hasn't been built yet.

    Parameters
    ----------
    query:
        The search query (any supported language).
    k:
        Number of results to return (default 3).

    Returns
    -------
    list[dict]
        Each dict contains ``concept_id`` (int), ``language`` (str), and
        ``text`` (str), sorted from closest to farthest (ascending distance).
    """
    # Ensure the index exists
    tbl = _get_or_open_table()
    if tbl is None:
        build_index()
        tbl = _get_or_open_table()

    # Embed the query using the "search_query" input type
    query_vector = _embed_texts([query], input_type="search_query")[0]
    query_array = np.array(query_vector, dtype=np.float32).tolist()

    # Search across all languages, no filter
    results = (
        tbl.search(query_array)
        .metric("cosine")
        .limit(k)
        .select(["concept_id", "language", "text"])
        .to_list()
    )

    # Return as plain dicts with only the required keys (strip LanceDB internals)
    return [
        {
            "concept_id": int(row["concept_id"]),
            "language": row["language"],
            "text": row["text"],
        }
        for row in results
    ]
