import json
import os
import sqlite3

import pytest

PROJECT_DIR = "/home/user/myproject"
CONFIG_PATH = "/app/data/config.json"
QUERIES_PATH = "/app/data/queries.json"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_sqlite_fts5_available():
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
    finally:
        con.close()


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_config_file_exists_and_has_keys():
    assert os.path.isfile(CONFIG_PATH), f"Config file {CONFIG_PATH} does not exist."
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    for key in ("sqlite_path", "lancedb_uri", "table_name", "vector_dim", "distance_metric", "rrf_k"):
        assert key in cfg, f"Config file is missing required key '{key}'."


def test_queries_fixture_exists():
    assert os.path.isfile(QUERIES_PATH), f"Query fixture {QUERIES_PATH} does not exist."
    with open(QUERIES_PATH) as f:
        queries = json.load(f)
    assert isinstance(queries, list) and len(queries) >= 1, "Query fixture must be a non-empty list."
    q0 = queries[0]
    assert "query" in q0 and "vector" in q0, "Each query fixture must have 'query' and 'vector'."


def test_sqlite_store_seeded():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    assert os.path.isfile(cfg["sqlite_path"]), f"SQLite DB {cfg['sqlite_path']} does not exist."
    con = sqlite3.connect(cfg["sqlite_path"])
    try:
        table = cfg["table_name"]
        cur = con.execute(f"SELECT count(*) FROM {table}")
        n = cur.fetchone()[0]
    finally:
        con.close()
    assert n > 0, f"SQLite FTS5 table {table} has no rows."


def test_lancedb_store_seeded():
    import lancedb

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    db = lancedb.connect(cfg["lancedb_uri"])
    assert cfg["table_name"] in db.table_names(), (
        f"LanceDB table {cfg['table_name']} not found in {cfg['lancedb_uri']}."
    )
    tbl = db.open_table(cfg["table_name"])
    assert tbl.count_rows() > 0, "LanceDB table has no rows."


def test_stores_row_counts_match():
    import lancedb

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    con = sqlite3.connect(cfg["sqlite_path"])
    try:
        sqlite_n = con.execute(f"SELECT count(*) FROM {cfg['table_name']}").fetchone()[0]
    finally:
        con.close()
    db = lancedb.connect(cfg["lancedb_uri"])
    lance_n = db.open_table(cfg["table_name"]).count_rows()
    assert sqlite_n == lance_n, (
        f"Row count mismatch between stores: sqlite={sqlite_n}, lancedb={lance_n}."
    )
