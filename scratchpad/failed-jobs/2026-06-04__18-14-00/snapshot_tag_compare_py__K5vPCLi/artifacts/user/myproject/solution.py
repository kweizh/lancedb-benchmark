"""
LanceDB Snapshot Tags & Diff Tool
==================================
Provides:
  - build_snapshots(db_path, table_name): create table + 3 tagged snapshots.
  - diff(db_path, table_name, tag_a, tag_b): compare two snapshots by tag name.
"""

import os
import numpy as np
import pyarrow as pa
import lancedb


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 16)),
])


def _make_rows(ids, rng: np.random.Generator) -> list[dict]:
    """Build a list of row dicts for the given id range using the RNG."""
    vectors = rng.random((len(ids), 16)).astype(np.float32)
    return [
        {"id": int(i), "text": f"doc-{i}", "vector": vectors[idx].tolist()}
        for idx, i in enumerate(ids)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_snapshots(db_path: str, table_name: str) -> None:
    """
    Create the LanceDB table and produce three tagged snapshots:

      v1_baseline  – 50 rows  (id 0-49)
      v2_extended  – 70 rows  (id 0-69, appended 20 more)
      v3_pruned    – 65 rows  (id 5-69, deleted rows where id < 5)
    """
    db = lancedb.connect(db_path)

    # Drop pre-existing table so the function is idempotent
    if table_name in db.table_names():
        db.drop_table(table_name)

    rng = np.random.default_rng(seed=2026)

    # ------------------------------------------------------------------
    # Seed phase – v1_baseline (50 rows, id 0-49)
    # ------------------------------------------------------------------
    seed_rows = _make_rows(range(0, 50), rng)
    table = db.create_table(table_name, data=seed_rows, schema=SCHEMA)
    table.tags.create("v1_baseline", table.version)
    print(f"[build_snapshots] v1_baseline → version {table.version}, rows {table.count_rows()}")

    # ------------------------------------------------------------------
    # Extend phase – v2_extended (20 more rows, id 50-69)
    # ------------------------------------------------------------------
    extra_rows = _make_rows(range(50, 70), rng)
    table.add(extra_rows)
    table.tags.create("v2_extended", table.version)
    print(f"[build_snapshots] v2_extended → version {table.version}, rows {table.count_rows()}")

    # ------------------------------------------------------------------
    # Prune phase – v3_pruned (delete rows where id < 5)
    # ------------------------------------------------------------------
    table.delete("id < 5")
    table.tags.create("v3_pruned", table.version)
    print(f"[build_snapshots] v3_pruned  → version {table.version}, rows {table.count_rows()}")

    print(f"[build_snapshots] Tags: {list(table.tags.list().keys())}")


def diff(db_path: str, table_name: str, tag_a: str, tag_b: str) -> dict:
    """
    Compare two snapshots identified by *tag_a* and *tag_b*.

    Returns
    -------
    dict with keys:
        added_ids    – sorted list of ints present in B but not in A
        removed_ids  – sorted list of ints present in A but not in B
        common_count – number of ids present in both snapshots
    """
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    # Read snapshot A
    table.checkout(tag_a)
    ids_a = set(table.to_arrow().column("id").to_pylist())

    # Read snapshot B
    table.checkout(tag_b)
    ids_b = set(table.to_arrow().column("id").to_pylist())

    # Restore to latest so subsequent callers see the live version
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
    db_path = "/app/db"
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    table_name = f"documents_{run_id}"

    print(f"[main] db_path={db_path!r}  table_name={table_name!r}")
    build_snapshots(db_path, table_name)

    print("\n[main] Verifying diff v1_baseline → v2_extended:")
    result = diff(db_path, table_name, "v1_baseline", "v2_extended")
    print(result)

    print("\n[main] Verifying diff v2_extended → v3_pruned:")
    result = diff(db_path, table_name, "v2_extended", "v3_pruned")
    print(result)
