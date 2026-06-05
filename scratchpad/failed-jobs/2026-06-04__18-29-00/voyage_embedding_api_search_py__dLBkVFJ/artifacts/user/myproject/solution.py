"""
Semantic product search using Voyage AI embeddings and LanceDB.

Provides:
  - build_catalogue(): embeds products and stores them in LanceDB (idempotent)
  - search(query, k): performs vector similarity search and returns ranked results
"""

import json
import os

import lancedb
import numpy as np
import pyarrow as pa
import voyageai

# ---------------------------------------------------------------------------
# Configuration read from environment
# ---------------------------------------------------------------------------
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
ZEALT_RUN_ID = os.environ["ZEALT_RUN_ID"]

DB_PATH = "/home/user/myproject/lancedb_data"
TABLE_NAME = f"products_{ZEALT_RUN_ID}"
PRODUCTS_PATH = "/home/user/myproject/products.json"
MODEL = "voyage-3"
EMBEDDING_DIM = 1024


def _load_products() -> list[dict]:
    """Read the product catalogue from the JSON file."""
    with open(PRODUCTS_PATH, "r") as f:
        return json.load(f)


def _embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed a list of texts using the Voyage AI API."""
    client = voyageai.Client(api_key=VOYAGE_API_KEY)
    result = client.embed(texts, model=MODEL, input_type=input_type)
    return result.embeddings


def build_catalogue() -> None:
    """
    Embed every product description and store it in LanceDB.

    This function is idempotent: if the table already exists it returns
    immediately without re-embedding.
    """
    db = lancedb.connect(DB_PATH)

    # Check if table already exists (idempotent)
    existing_tables = db.table_names()
    if TABLE_NAME in existing_tables:
        return

    products = _load_products()
    descriptions = [p["description"] for p in products]

    # Embed all descriptions with input_type="document"
    embeddings = _embed_texts(descriptions, input_type="document")

    # Build a pyarrow table with the required schema
    ids = [p["id"] for p in products]
    vectors = [np.array(e, dtype=np.float32) for e in embeddings]

    table_data = pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "description": pa.array(descriptions, type=pa.string()),
            "vector": pa.array(
                vectors,
                type=pa.list_(pa.float32(), EMBEDDING_DIM),
            ),
        }
    )

    db.create_table(TABLE_NAME, table_data)


def search(query: str, k: int) -> list[dict]:
    """
    Perform a semantic search over the product catalogue.

    Args:
        query: The search query text.
        k: Number of results to return.

    Returns:
        A list of dicts (length k), each with at least 'id' and 'description',
        ordered from most-relevant (rank 1) to least-relevant.
    """
    # Make sure the catalogue is built before searching
    build_catalogue()

    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)

    # Embed the query with input_type="query"
    query_embeddings = _embed_texts([query], input_type="query")
    query_vec = query_embeddings[0]

    # Perform vector search via LanceDB
    results = table.search(query_vec).limit(k).to_list()

    # Return only id and description, preserving relevance order
    return [
        {"id": row["id"], "description": row["description"]}
        for row in results
    ]