import os
import importlib

import pytest

PROJECT_DIR = "/home/user/hybrid_bridge"
LANCEDB_DIR = "/home/user/hybrid_bridge/data/lancedb"

CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123


def test_lancedb_importable():
    assert importlib.util.find_spec("lancedb") is not None, \
        "The 'lancedb' package must be importable in the environment."


def test_clickhouse_client_importable():
    assert importlib.util.find_spec("clickhouse_connect") is not None, \
        "The 'clickhouse_connect' package must be importable in the environment."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), \
        f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), \
        f"LanceDB directory {LANCEDB_DIR} does not exist."


def test_documents_table_seeded():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    names = db.table_names()
    assert "documents" in names, \
        f"Expected a LanceDB table named 'documents', found: {names}"

    table = db.open_table("documents")
    schema_names = set(table.schema.names)
    for col in ("id", "text", "category", "vector"):
        assert col in schema_names, \
            f"Column '{col}' missing from 'documents' schema: {schema_names}"

    assert table.count_rows() > 0, "The 'documents' table must be seeded with rows."


def test_documents_fts_index_present():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    table = db.open_table("documents")
    indices = table.list_indices()
    covers_text = False
    for idx in indices:
        columns = getattr(idx, "columns", None)
        if columns and "text" in columns:
            covers_text = True
            break
    assert covers_text, \
        "An FTS index on the 'text' column of 'documents' must exist in the initial state."


def _clickhouse_client():
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username="default",
        password="",
        database="default",
    )


def test_clickhouse_server_reachable():
    client = _clickhouse_client()
    result = client.query("SELECT 1").result_rows
    assert result and int(result[0][0]) == 1, \
        "The local ClickHouse server must be reachable and answer 'SELECT 1'."


def test_clickhouse_events_seeded():
    client = _clickhouse_client()
    tables = {row[0] for row in client.query("SHOW TABLES").result_rows}
    assert "events" in tables, f"ClickHouse table 'events' must exist, found: {tables}"
    count = client.query("SELECT count() FROM events").result_rows[0][0]
    assert int(count) > 0, "The 'events' table must be seeded with rows."


def test_clickhouse_users_seeded():
    client = _clickhouse_client()
    tables = {row[0] for row in client.query("SHOW TABLES").result_rows}
    assert "users" in tables, f"ClickHouse table 'users' must exist, found: {tables}"
    count = client.query("SELECT count() FROM users").result_rows[0][0]
    assert int(count) > 0, "The 'users' table must be seeded with rows."

    tiers = {row[0] for row in client.query("SELECT DISTINCT tier FROM users").result_rows}
    assert "premium" in tiers, "The 'users' table must contain at least one 'premium' user."
