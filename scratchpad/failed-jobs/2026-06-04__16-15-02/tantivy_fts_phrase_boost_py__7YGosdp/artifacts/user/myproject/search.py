#!/usr/bin/env python3
"""Tantivy-backed full-text search over a LanceDB table of technical documents."""

import argparse
import json
import os
import sys

import lancedb
import pyarrow as pa


SEED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed", "docs.json")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lancedb_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tantivy FTS search over LanceDB")
    parser.add_argument("--query", required=True, help="Raw Tantivy query string")
    parser.add_argument("--k", type=int, required=True, help="Number of results to return")
    args = parser.parse_args()

    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    table_name = f"docs_{run_id}"

    db = lancedb.connect(DB_PATH)

    existing_tables = db.table_names()
    if table_name in existing_tables:
        table = db.open_table(table_name)
    else:
        with open(SEED_PATH, "r") as f:
            docs = json.load(f)

        table = db.create_table(table_name, docs)

        table.create_fts_index(
            field_names=["title", "body"],
            use_tantivy=True,
            replace=True,
        )

    results = table.search(args.query, query_type="fts").limit(args.k).to_list()

    cleaned = []
    for row in results:
        cleaned.append({
            "id": int(row["id"]),
            "title": row["title"],
            "body": row["body"],
        })

    json.dump(cleaned, sys.stdout)
    print()


if __name__ == "__main__":
    main()