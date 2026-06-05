"""LanceDB Snapshot Tags & Diff Tool."""

import os
import lancedb
import numpy as np
import pyarrow as pa


def _table_name() -> str:
    run_id = os.environ["ZEALT_RUN_ID"]
    return f"documents_{run_id}"


def _schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 16)),
    ])


def _make_rows(rng: np.random.Generator, start: int, end: int) -> list[dict]:
    rows = []
    for i in range(start, end):
        rows.append({
            "id": i,
            "text": f"doc-{i}",
            "vector": rng.random(16).astype(np.float32).tolist(),
        })
    return rows


def build_snapshots(db_path: str, table_name: str) -> None:
    db = lancedb.connect(db_path)

    # Seed phase: 50 rows, tag v1_baseline
    rng = np.random.default_rng(seed=2026)
    rows_v1 = _make_rows(rng, 0, 50)
    table = db.create_table(table_name, rows_v1, schema=_schema())
    table.tags.create("v1_baseline", table.version)

    # Extend phase: 20 more rows, tag v2_extended
    rows_v2 = _make_rows(rng, 50, 70)
    table.add(rows_v2)
    table.tags.create("v2_extended", table.version)

    # Prune phase: delete id < 5, tag v3_pruned
    table.delete("id < 5")
    table.tags.create("v3_pruned", table.version)

    # Ensure we're on latest for any subsequent operations
    table.checkout_latest()


def diff(db_path: str, table_name: str, tag_a: str, tag_b: str) -> dict:
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    # Read snapshot A
    table.checkout(tag_a)
    ids_a = set(table.to_arrow()["id"].to_pylist())

    # Read snapshot B
    table.checkout(tag_b)
    ids_b = set(table.to_arrow()["id"].to_pylist())

    # Restore to latest before returning
    table.checkout_latest()

    added = sorted(ids_b - ids_a)
    removed = sorted(ids_a - ids_b)
    common = ids_a & ids_b

    return {
        "added_ids": added,
        "removed_ids": removed,
        "common_count": len(common),
    }


if __name__ == "__main__":
    db_path = "/app/db"
    table_name = _table_name()
    build_snapshots(db_path, table_name)

    # Verify
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    print(f"Table: {table_name}")
    print(f"Tags: {table.tags.list()}")
    print(f"Row count (latest): {table.count_rows()}")

    # Diff v1_baseline vs v2_extended
    result = diff(db_path, table_name, "v1_baseline", "v2_extended")
    print(f"diff(v1_baseline, v2_extended): {result}")

    # Diff v2_extended vs v3_pruned
    result = diff(db_path, table_name, "v2_extended", "v3_pruned")
    print(f"diff(v2_extended, v3_pruned): {result}")