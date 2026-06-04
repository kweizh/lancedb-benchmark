#!/usr/bin/env python3
"""
Tantivy-backed full-text search over a LanceDB table.

Usage:
    python3 search.py --query <tantivy-query-string> --k <int>

The script is idempotent:
  - On first run it loads docs from seed/docs.json, creates the LanceDB table,
    and builds a Tantivy-backed FTS index on both 'title' and 'body'.
  - On subsequent runs it reuses the existing table/index.

The table name is derived from the ZEALT_RUN_ID environment variable so that
parallel runs never collide on shared storage.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import lancedb

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()
SEED_FILE   = PROJECT_DIR / "seed" / "docs.json"
DB_DIR      = PROJECT_DIR / "lancedb_data"
FTS_COLUMNS = ["title", "body"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_table_name() -> str:
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"docs_{run_id}"


def load_or_create_table(db: lancedb.LanceDBConnection, table_name: str):
    """Return the LanceDB table, creating and indexing it if necessary."""
    existing = db.table_names()

    if table_name in existing:
        table = db.open_table(table_name)
        return table

    # ---- First run: seed the table ----------------------------------------
    with open(SEED_FILE, "r", encoding="utf-8") as fh:
        docs = json.load(fh)

    table = db.create_table(table_name, data=docs)

    # ---- Build Tantivy FTS index covering both text columns ----------------
    # use_tantivy=True → legacy Tantivy path with full query-parser syntax.
    # We pass both columns as a list so field-scoped queries work against
    # either title: or body: fields.
    table.create_fts_index(
        FTS_COLUMNS,
        use_tantivy=True,
        replace=True,
    )

    return table


def run_query(table, query: str, k: int) -> list[dict]:
    """Execute a Tantivy query and return the top-k hits as plain dicts."""
    results = (
        table.search(query, query_type="fts")
        .limit(k)
        .to_list()
    )

    # Strip any internal LanceDB columns (e.g. _score, _distance, _rowid)
    clean = []
    for row in results:
        clean.append({
            "id":    int(row["id"]),
            "title": str(row["title"]),
            "body":  str(row["body"]),
        })
    return clean


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a Tantivy FTS query against a LanceDB table."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Raw Tantivy query string (phrases, field boosts, boolean ops …)",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of top results to return.",
    )
    args = parser.parse_args()

    DB_DIR.mkdir(parents=True, exist_ok=True)
    db         = lancedb.connect(str(DB_DIR))
    table_name = get_table_name()
    table      = load_or_create_table(db, table_name)
    hits       = run_query(table, args.query, args.k)

    print(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
