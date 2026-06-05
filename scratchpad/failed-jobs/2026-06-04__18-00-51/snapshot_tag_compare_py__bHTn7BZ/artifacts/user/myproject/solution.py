"""
LanceDB Snapshot Tags & Diff Tool
==================================
Provides:
  - build_snapshots(db_path, table_name): creates three tagged snapshots
  - diff(db_path, table_name, tag_a, tag_b): compares two snapshots by tag name
"""

import os
import numpy as np
import pyarrow as pa
import lancedb


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 16)),
    ]
)


def _make_rows(ids: range, rng: np.random.Generator) -> list[dict]:
    """Generate rows for the given id range using *rng* (advances in-place)."""
    rows = []
    for i in ids:
        vec = rng.random(16).astype(np.float32).tolist()
        rows.append({"id": i, "text": f"doc-{i}", "vector": vec})
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_snapshots(db_path: str, table_name: str) -> None:
    """Create the table, perform the three mutations, and write three tags.

    Tags created:
      v1_baseline  – after inserting ids [0, 50)
      v2_extended  – after appending ids [50, 70)
      v3_pruned    – after deleting rows where id < 5
    """
    db = lancedb.connect(db_path)

    # Drop existing table so the function is idempotent when re-run.
    if table_name in db.table_names():
        db.drop_table(table_name)

    # Deterministic RNG (shared across phases so vectors are reproducible).
    rng = np.random.default_rng(seed=2026)

    # ------------------------------------------------------------------
    # Phase 1 – Seed: insert 50 rows with id in [0, 50)
    # ------------------------------------------------------------------
    seed_rows = _make_rows(range(50), rng)
    table = db.create_table(table_name, data=seed_rows, schema=SCHEMA)
    table.tags.create("v1_baseline", table.version)
    print(f"[v1_baseline] version={table.version}, rows={table.count_rows()}")

    # ------------------------------------------------------------------
    # Phase 2 – Extend: append 20 rows with id in [50, 70)
    # ------------------------------------------------------------------
    extend_rows = _make_rows(range(50, 70), rng)
    table.add(extend_rows)
    table.tags.create("v2_extended", table.version)
    print(f"[v2_extended] version={table.version}, rows={table.count_rows()}")

    # ------------------------------------------------------------------
    # Phase 3 – Prune: delete rows where id < 5  (removes exactly 5 rows)
    # ------------------------------------------------------------------
    table.delete("id < 5")
    table.tags.create("v3_pruned", table.version)
    print(f"[v3_pruned]   version={table.version}, rows={table.count_rows()}")

    # Confirm all three tags exist.
    tags = table.tags.list()
    print(f"Tags: {list(tags.keys())}")


def diff(db_path: str, table_name: str, tag_a: str, tag_b: str) -> dict:
    """Return the set-difference between two tagged snapshots.

    Parameters
    ----------
    db_path:    Path to the LanceDB database directory.
    table_name: Name of the table to inspect.
    tag_a:      Name of the "before" snapshot tag.
    tag_b:      Name of the "after"  snapshot tag.

    Returns
    -------
    dict with keys:
      added_ids    – sorted list of ints present in B but not in A
      removed_ids  – sorted list of ints present in A but not in B
      common_count – number of ids present in both snapshots
    """
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    # Read ids from snapshot A.
    table.checkout(tag_a)
    ids_a = set(table.to_arrow().column("id").to_pylist())

    # Read ids from snapshot B.
    table.checkout(tag_b)
    ids_b = set(table.to_arrow().column("id").to_pylist())

    # Restore to the live version so subsequent callers see current data.
    table.checkout_latest()

    added_ids = sorted(ids_b - ids_a)
    removed_ids = sorted(ids_a - ids_b)
    common_count = len(ids_a & ids_b)

    return {
        "added_ids": added_ids,
        "removed_ids": removed_ids,
        "common_count": common_count,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DB_PATH = "/app/db"
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    table_name = f"documents_{run_id}"

    print(f"Building snapshots in '{DB_PATH}', table='{table_name}' ...")
    build_snapshots(DB_PATH, table_name)

    print("\nVerifying diffs:")
    d1 = diff(DB_PATH, table_name, "v1_baseline", "v2_extended")
    print(f"  v1_baseline → v2_extended: {d1}")

    d2 = diff(DB_PATH, table_name, "v2_extended", "v3_pruned")
    print(f"  v2_extended → v3_pruned:   {d2}")

    print("\nDone.")
