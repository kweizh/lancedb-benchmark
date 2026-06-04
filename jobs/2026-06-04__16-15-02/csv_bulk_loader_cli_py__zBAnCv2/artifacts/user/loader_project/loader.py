#!/usr/bin/env python3
"""CSV Bulk Loader CLI for LanceDB with OpenAI embeddings."""

import argparse
import json
import logging
import sys
import time

import lancedb
import numpy as np
import pandas as pd
from openai import OpenAI

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)

DB_PATH = "/home/user/loader_project/lance_db"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def get_openai_client() -> OpenAI:
    return OpenAI()


def embed_texts(client: OpenAI, texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Embed a list of texts using OpenAI, with simple retry logic."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            # Sort by index to guarantee order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            wait = 2 ** attempt
            logger.warning("Embedding call failed (attempt %d/%d): %s. Retrying in %ds...",
                           attempt + 1, max_retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to embed texts after {max_retries} attempts")


def embed_in_batches(client: OpenAI, texts: list[str], batch_size: int) -> list[list[float]]:
    """Embed texts in batches to avoid rate limits."""
    all_embeddings: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]
        logger.info("Embedding batch %d-%d of %d", start + 1, end, total)
        batch_embeddings = embed_texts(client, batch)
        all_embeddings.extend(batch_embeddings)
        # Small delay between batches to avoid rate limiting
        if end < total:
            time.sleep(0.5)
    return all_embeddings


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest CSV data into LanceDB with embeddings."""
    csv_path = args.csv
    table_name = args.table
    text_col = args.text_col
    batch_size = args.batch_size

    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info("Read %d rows from %s", len(df), csv_path)

    # Ensure required columns exist
    required_cols = {"id", "title", "body", "category", "published"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"CSV is missing required columns: {missing}")

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in CSV columns: {list(df.columns)}")

    # Prepare texts for embedding
    texts = df[text_col].fillna("").tolist()

    # Get embeddings
    client = get_openai_client()
    embeddings = embed_in_batches(client, texts, batch_size)

    if len(embeddings) != len(df):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(embeddings)}, expected {len(df)}"
        )

    # Build list-of-dicts for LanceDB
    records = []
    for i in range(len(df)):
        records.append({
            "id": int(df.iloc[i]["id"]),
            "title": str(df.iloc[i]["title"]),
            "body": str(df.iloc[i]["body"]),
            "category": str(df.iloc[i]["category"]),
            "published": str(df.iloc[i]["published"]),
            "vector": embeddings[i],
        })

    # Connect to LanceDB and create table
    db = lancedb.connect(DB_PATH)
    # Use create_table with mode="overwrite" for clean re-runs
    table = db.create_table(table_name, records, mode="overwrite")

    logger.info("Created table '%s' with %d rows", table_name, len(table))


def cmd_search(args: argparse.Namespace) -> None:
    """Search a LanceDB table using a query string."""
    table_name = args.table
    query = args.query
    k = args.k

    # Embed the query
    client = get_openai_client()
    query_embeddings = embed_texts(client, [query])
    query_vector = query_embeddings[0]

    # Open the table
    db = lancedb.connect(DB_PATH)
    table = db.open_table(table_name)

    # Run vector search
    results = table.search(query_vector).limit(k).to_pandas()

    # Build output JSON
    output_results = []
    for _, row in results.iterrows():
        entry = {
            "id": int(row["id"]),
            "title": str(row["title"]),
            "category": str(row["category"]),
            "published": str(row["published"]),
            "score": float(row["_distance"]),
        }
        output_results.append(entry)

    output = {
        "query": query,
        "k": k,
        "results": output_results,
    }

    print(json.dumps(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV Bulk Loader for LanceDB")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest CSV into LanceDB")
    ingest_parser.add_argument("--csv", required=True, help="Path to CSV file")
    ingest_parser.add_argument("--table", required=True, help="LanceDB table name")
    ingest_parser.add_argument("--text-col", required=True, help="Column to embed")
    ingest_parser.add_argument("--batch-size", type=int, default=100, help="Embedding batch size")

    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Search LanceDB table")
    search_parser.add_argument("--table", required=True, help="LanceDB table name")
    search_parser.add_argument("--query", required=True, help="Search query string")
    search_parser.add_argument("--k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()