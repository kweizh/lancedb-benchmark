"""Final-state verification for sqlite_to_lance_etl_py.

The verifier wires up a fresh SQLite source DB, calls the candidate's
``solution.sync`` for a full sync, mutates the SQLite DB (5 inserts,
10 updates, 7 soft-deletes), runs an incremental sync, and asserts both
the returned counters and the final LanceDB table contents.
"""

from __future__ import annotations

import importlib
import math
import os
import shutil
import sqlite3
import sys
import time

import lancedb
import numpy as np
import pyarrow as pa
import pytest
from openai import OpenAI

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
RUN_ID = os.environ.get("ZEALT_RUN_ID")
TABLE_NAME = f"articles_{RUN_ID}" if RUN_ID else "articles_unset"
SQLITE_PATH = os.path.join(PROJECT_DIR, f"source_{RUN_ID}.db") if RUN_ID else os.path.join(
    PROJECT_DIR, "source_unset.db"
)
CATEGORIES = ("tech", "science", "sports", "finance", "world")
EMBED_MODEL = "text-embedding-3-small"


def _seed_sqlite(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE articles ("
            "id INTEGER PRIMARY KEY, "
            "title TEXT, "
            "body TEXT, "
            "category TEXT, "
            "updated_at INTEGER, "
            "deleted INTEGER"
            ")"
        )
        rows = []
        for i in range(200):
            rows.append(
                (
                    i,
                    f"Title {i}",
                    f"Body content for article {i}.",
                    CATEGORIES[i % len(CATEGORIES)],
                    1_700_000_000 + i,
                    0,
                )
            )
        cur.executemany(
            "INSERT INTO articles (id, title, body, category, updated_at, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _mutate_sqlite(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        # 5 new inserts: ids 200..204, updated_at 1_700_002_000 + offset
        for offset, new_id in enumerate(range(200, 205)):
            cur.execute(
                "INSERT INTO articles (id, title, body, category, updated_at, deleted) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (
                    new_id,
                    f"New Title {new_id}",
                    f"Fresh body for article {new_id} written after the initial sync.",
                    CATEGORIES[new_id % len(CATEGORIES)],
                    1_700_002_000 + offset,
                ),
            )
        # 10 updates: ids [0,10,...,90], updated_at 1_700_003_000 + offset
        update_ids = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        for offset, uid in enumerate(update_ids):
            cur.execute(
                "UPDATE articles SET title=?, body=?, updated_at=? WHERE id=?",
                (
                    f"Updated Title {uid}",
                    f"Rewritten body for article {uid} after copy-editing pass.",
                    1_700_003_000 + offset,
                    uid,
                ),
            )
        # 7 soft-deletes: ids [5,15,...,65], updated_at 1_700_004_000 + offset
        delete_ids = [5, 15, 25, 35, 45, 55, 65]
        for offset, did in enumerate(delete_ids):
            cur.execute(
                "UPDATE articles SET deleted=1, updated_at=? WHERE id=?",
                (1_700_004_000 + offset, did),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(scope="session")
def solution_module():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    return importlib.import_module("solution")


@pytest.fixture(scope="session")
def initial_sync_result(solution_module):
    # Make sure we start from a clean LanceDB directory; SQLite re-seeded fresh.
    if os.path.isdir(LANCEDB_DIR):
        shutil.rmtree(LANCEDB_DIR)
    _seed_sqlite(SQLITE_PATH)
    result = solution_module.sync(SQLITE_PATH, TABLE_NAME, 0)
    return result


@pytest.fixture(scope="session")
def incremental_sync_result(solution_module, initial_sync_result):
    # initial_sync_result triggers the initial sync; now mutate and re-sync.
    _mutate_sqlite(SQLITE_PATH)
    result = solution_module.sync(SQLITE_PATH, TABLE_NAME, 1_700_001_000)
    return result


def test_solution_has_sync_function(solution_module):
    assert hasattr(solution_module, "sync"), "solution.py must define a top-level `sync` function."
    assert callable(solution_module.sync), "solution.sync must be callable."


def test_initial_sync_counts(initial_sync_result):
    assert isinstance(initial_sync_result, dict), (
        f"sync() must return a dict, got {type(initial_sync_result).__name__}."
    )
    for key in ("inserted", "updated", "deleted", "high_water_ts"):
        assert key in initial_sync_result, f"Initial sync result missing key {key!r}."
    assert initial_sync_result["inserted"] == 200, (
        f"Initial sync should insert 200 rows, got {initial_sync_result['inserted']}."
    )
    assert initial_sync_result["updated"] == 0, (
        f"Initial sync should update 0 rows, got {initial_sync_result['updated']}."
    )
    assert initial_sync_result["deleted"] == 0, (
        f"Initial sync should delete 0 rows, got {initial_sync_result['deleted']}."
    )
    assert initial_sync_result["high_water_ts"] == 1_700_000_199, (
        f"Initial high_water_ts should be 1700000199, got {initial_sync_result['high_water_ts']}."
    )


def test_initial_table_row_count(initial_sync_result):
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(TABLE_NAME)
    assert tbl.count_rows() == 200, (
        f"Expected 200 rows in {TABLE_NAME} after initial sync, got {tbl.count_rows()}."
    )


def test_initial_table_schema(initial_sync_result):
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(TABLE_NAME)
    schema = tbl.schema
    field_names = {f.name for f in schema}
    required = {"id", "title", "body", "category", "updated_at", "vector"}
    missing = required - field_names
    assert not missing, f"LanceDB table schema is missing required fields: {missing}."

    field_types = {f.name: f.type for f in schema}
    assert pa.types.is_integer(field_types["id"]), (
        f"`id` column must be an integer type, got {field_types['id']}."
    )
    assert pa.types.is_string(field_types["title"]) or pa.types.is_large_string(
        field_types["title"]
    ), f"`title` column must be a string type, got {field_types['title']}."
    assert pa.types.is_string(field_types["body"]) or pa.types.is_large_string(
        field_types["body"]
    ), f"`body` column must be a string type, got {field_types['body']}."
    assert pa.types.is_string(field_types["category"]) or pa.types.is_large_string(
        field_types["category"]
    ), f"`category` column must be a string type, got {field_types['category']}."
    assert pa.types.is_integer(field_types["updated_at"]), (
        f"`updated_at` column must be an integer type, got {field_types['updated_at']}."
    )

    vec_type = field_types["vector"]
    assert pa.types.is_fixed_size_list(vec_type), (
        f"`vector` column must be a fixed-size list, got {vec_type}."
    )
    assert vec_type.list_size == 1536, (
        f"`vector` column must have 1536 dimensions, got {vec_type.list_size}."
    )
    assert pa.types.is_floating(vec_type.value_type), (
        f"`vector` element type must be floating, got {vec_type.value_type}."
    )


def test_incremental_sync_counts(incremental_sync_result):
    assert isinstance(incremental_sync_result, dict), (
        f"sync() must return a dict, got {type(incremental_sync_result).__name__}."
    )
    for key in ("inserted", "updated", "deleted", "high_water_ts"):
        assert key in incremental_sync_result, (
            f"Incremental sync result missing key {key!r}."
        )
    assert incremental_sync_result["inserted"] == 5, (
        f"Incremental sync should insert 5 rows, got {incremental_sync_result['inserted']}."
    )
    assert incremental_sync_result["updated"] == 10, (
        f"Incremental sync should update 10 rows, got {incremental_sync_result['updated']}."
    )
    assert incremental_sync_result["deleted"] == 7, (
        f"Incremental sync should delete 7 rows, got {incremental_sync_result['deleted']}."
    )
    assert incremental_sync_result["high_water_ts"] == 1_700_004_006, (
        "Incremental high_water_ts should be 1700004006, got "
        f"{incremental_sync_result['high_water_ts']}."
    )


def test_final_table_row_count(incremental_sync_result):
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(TABLE_NAME)
    assert tbl.count_rows() == 198, (
        f"Final table row count should be 198 (200 + 5 - 7), got {tbl.count_rows()}."
    )


def test_deleted_ids_absent(incremental_sync_result):
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(TABLE_NAME)
    deleted_ids = [5, 15, 25, 35, 45, 55, 65]
    id_list_sql = ",".join(str(i) for i in deleted_ids)
    remaining = tbl.count_rows(filter=f"id IN ({id_list_sql})")
    assert remaining == 0, (
        f"Soft-deleted ids {deleted_ids} must not be present in the table; found {remaining}."
    )


def test_inserted_ids_present(incremental_sync_result):
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(TABLE_NAME)
    new_ids = [200, 201, 202, 203, 204]
    id_list_sql = ",".join(str(i) for i in new_ids)
    rows = tbl.search().where(f"id IN ({id_list_sql})").limit(50).to_list()
    seen = {row["id"] for row in rows}
    assert seen == set(new_ids), (
        f"Expected new ids {new_ids} in the table, got {sorted(seen)}."
    )
    for row in rows:
        nid = row["id"]
        assert row["title"] == f"New Title {nid}", (
            f"Inserted row id={nid} should have title 'New Title {nid}', got {row['title']!r}."
        )


def test_updated_rows_have_new_content(incremental_sync_result):
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(TABLE_NAME)
    update_ids = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    id_list_sql = ",".join(str(i) for i in update_ids)
    rows = tbl.search().where(f"id IN ({id_list_sql})").limit(50).to_list()
    by_id = {row["id"]: row for row in rows}
    for uid in update_ids:
        assert uid in by_id, f"Updated id={uid} missing from final table."
        row = by_id[uid]
        assert row["title"] == f"Updated Title {uid}", (
            f"Updated id={uid} should have title 'Updated Title {uid}', got {row['title']!r}."
        )
        assert "Rewritten body" in row["body"], (
            f"Updated id={uid} should have a rewritten body, got {row['body']!r}."
        )


def test_updated_vector_matches_openai_embedding(incremental_sync_result):
    """Cosine-similarity sanity check that the embedding was recomputed
    against the new (title, body) text for an updated row."""
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(TABLE_NAME)
    target_id = 0
    rows = tbl.search().where(f"id = {target_id}").limit(1).to_list()
    assert len(rows) == 1, f"Expected exactly one row with id={target_id}, got {len(rows)}."
    stored_vec = np.asarray(rows[0]["vector"], dtype=np.float32)
    assert stored_vec.shape == (1536,), (
        f"Stored vector for id={target_id} must be 1536-d, got shape={stored_vec.shape}."
    )
    assert np.any(stored_vec != 0), (
        f"Stored vector for id={target_id} must not be all zeros."
    )

    expected_title = f"Updated Title {target_id}"
    expected_body = f"Rewritten body for article {target_id} after copy-editing pass."
    expected_text = f"{expected_title}\n\n{expected_body}"

    client = OpenAI()
    last_err = None
    embedded = None
    for _attempt in range(3):
        try:
            embedded = client.embeddings.create(model=EMBED_MODEL, input=expected_text)
            break
        except Exception as exc:  # pragma: no cover - network resilience
            last_err = exc
            time.sleep(1.0)
    assert embedded is not None, f"OpenAI embedding call failed: {last_err}"
    expected_vec = np.asarray(embedded.data[0].embedding, dtype=np.float32)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        return float(np.dot(a, b)) / denom if denom else 0.0

    sim = cosine(stored_vec, expected_vec)
    assert math.isfinite(sim) and sim >= 0.99, (
        f"Stored vector for updated id={target_id} should match OpenAI embedding of the new "
        f"title+body (cosine >= 0.99); got sim={sim:.4f}."
    )
