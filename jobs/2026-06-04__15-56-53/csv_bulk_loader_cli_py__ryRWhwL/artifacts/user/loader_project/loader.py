#!/usr/bin/env python3
"""
CSV Bulk Loader CLI for LanceDB
Subcommands:
  ingest  -- reads a CSV, embeds a text column via OpenAI, writes to LanceDB
  search  -- embeds a query, runs vector search, prints JSON results to stdout
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import List, Dict, Any

import lancedb
import pyarrow as pa
from openai import OpenAI

# ── Constants ─────────────────────────────────────────────────────────────────
LANCE_DB_PATH = "/home/user/loader_project/lance_db"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: List[str], retries: int = 5) -> List[List[float]]:
    """Embed a list of texts, returning a list of float vectors."""
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            # sort by index to preserve order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as exc:
            wait = 2 ** attempt
            print(
                f"[embed] Attempt {attempt + 1}/{retries} failed: {exc}. "
                f"Retrying in {wait}s …",
                file=sys.stderr,
            )
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def read_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Read the CSV and return a list of row dicts (all values as strings)."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


# ── Subcommand: ingest ────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    csv_path: str = args.csv
    table_name: str = args.table
    text_col: str = args.text_col
    batch_size: int = args.batch_size

    print(f"[ingest] Reading CSV: {csv_path}", file=sys.stderr)
    rows = read_csv(csv_path)
    total = len(rows)
    print(f"[ingest] {total} rows loaded.", file=sys.stderr)

    if total == 0:
        print("[ingest] CSV has no data rows – nothing to do.", file=sys.stderr)
        sys.exit(0)

    if text_col not in rows[0]:
        print(
            f"[ingest] ERROR: text column '{text_col}' not found in CSV. "
            f"Available columns: {list(rows[0].keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    client = get_openai_client()

    # Embed in batches
    all_embeddings: List[List[float]] = []
    num_batches = (total + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_rows = rows[start:end]
        texts = [r[text_col] for r in batch_rows]
        print(
            f"[ingest] Embedding batch {batch_idx + 1}/{num_batches} "
            f"(rows {start + 1}–{end}) …",
            file=sys.stderr,
        )
        vectors = embed_texts(client, texts)
        all_embeddings.extend(vectors)

    print("[ingest] All embeddings obtained. Building Arrow table …", file=sys.stderr)

    # Build PyArrow table
    ids = pa.array([int(r["id"]) for r in rows], type=pa.int64())
    titles = pa.array([r["title"] for r in rows], type=pa.utf8())
    bodies = pa.array([r["body"] for r in rows], type=pa.utf8())
    categories = pa.array([r["category"] for r in rows], type=pa.utf8())
    published = pa.array([r["published"] for r in rows], type=pa.utf8())
    vectors = pa.array(all_embeddings, type=pa.list_(pa.float32(), EMBEDDING_DIM))

    arrow_table = pa.table(
        {
            "id": ids,
            "title": titles,
            "body": bodies,
            "category": categories,
            "published": published,
            "vector": vectors,
        }
    )

    # Write to LanceDB
    os.makedirs(LANCE_DB_PATH, exist_ok=True)
    db = lancedb.connect(LANCE_DB_PATH)
    print(
        f"[ingest] Writing {total} rows to table '{table_name}' …",
        file=sys.stderr,
    )
    db.create_table(table_name, data=arrow_table, mode="overwrite")
    print(f"[ingest] Done. Table '{table_name}' created with {total} rows.", file=sys.stderr)


# ── Subcommand: search ────────────────────────────────────────────────────────

def cmd_search(args: argparse.Namespace) -> None:
    table_name: str = args.table
    query_str: str = args.query
    k: int = args.k

    client = get_openai_client()

    print(f"[search] Embedding query …", file=sys.stderr)
    query_vector = embed_texts(client, [query_str])[0]

    print(f"[search] Searching table '{table_name}' for top-{k} …", file=sys.stderr)
    db = lancedb.connect(LANCE_DB_PATH)
    try:
        tbl = db.open_table(table_name)
    except Exception as exc:
        print(f"[search] ERROR: Cannot open table '{table_name}': {exc}", file=sys.stderr)
        sys.exit(1)

    results = (
        tbl.search(query_vector)
        .limit(k)
        .to_list()
    )

    # Build output (drop raw vector column, rename _distance -> score)
    output_results = []
    for row in results:
        # _distance is the L2 distance returned by LanceDB
        score = row.get("_distance", row.get("score", 0.0))
        output_results.append(
            {
                "id": int(row["id"]),
                "title": str(row["title"]),
                "category": str(row["category"]),
                "published": str(row["published"]),
                "score": float(score),
            }
        )

    output = {
        "query": query_str,
        "k": k,
        "results": output_results,
    }

    # Print ONLY the JSON to stdout
    print(json.dumps(output, ensure_ascii=False))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSV Bulk Loader CLI for LanceDB",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest sub-command
    ingest_p = subparsers.add_parser("ingest", help="Load CSV into LanceDB with embeddings")
    ingest_p.add_argument("--csv", required=True, help="Path to the CSV file")
    ingest_p.add_argument("--table", required=True, help="Target LanceDB table name")
    ingest_p.add_argument(
        "--text-col", required=True, help="CSV column to embed"
    )
    ingest_p.add_argument(
        "--batch-size", type=int, default=100, help="Embedding batch size (default: 100)"
    )

    # search sub-command
    search_p = subparsers.add_parser("search", help="Semantic search against a LanceDB table")
    search_p.add_argument("--table", required=True, help="LanceDB table name to search")
    search_p.add_argument("--query", required=True, help="Query string")
    search_p.add_argument("--k", type=int, default=10, help="Number of top results (default: 10)")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
