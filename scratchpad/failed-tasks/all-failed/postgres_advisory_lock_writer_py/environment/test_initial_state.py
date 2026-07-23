import os
import importlib

import pytest

PROJECT_DIR = "/home/user/myproject"


def test_lancedb_importable():
    assert importlib.util.find_spec("lancedb") is not None, "lancedb is not importable."


def test_psycopg2_importable():
    assert importlib.util.find_spec("psycopg2") is not None, "psycopg2 is not importable."


def test_numpy_importable():
    assert importlib.util.find_spec("numpy") is not None, "numpy is not importable."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_required_env_vars_present():
    for var in ("PG_LOCK_KEY", "LANCEDB_URI", "ZEALT_RUN_ID", "PGHOST", "PGPORT", "PGUSER", "PGDATABASE"):
        assert os.environ.get(var), f"Environment variable {var} is not set."


def test_pg_lock_key_is_integer():
    val = os.environ.get("PG_LOCK_KEY", "")
    assert val.lstrip("-").isdigit(), f"PG_LOCK_KEY must be an integer, got {val!r}."


def test_postgres_advisory_lock_available():
    import psycopg2

    conn = psycopg2.connect(
        host=os.environ["PGHOST"],
        port=os.environ["PGPORT"],
        user=os.environ["PGUSER"],
        dbname=os.environ["PGDATABASE"],
    )
    try:
        key = int(os.environ["PG_LOCK_KEY"])
        cur = conn.cursor()
        cur.execute("SELECT pg_try_advisory_lock(%s)", (key,))
        got = cur.fetchone()[0]
        assert got is True, "Could not acquire advisory lock on a fresh PostgreSQL server."
        cur.execute("SELECT pg_advisory_unlock(%s)", (key,))
        cur.close()
    finally:
        conn.close()


def test_lancedb_tables_exist_and_empty():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(os.environ["LANCEDB_URI"])
    names = set(db.table_names())

    data_name = f"records_{run_id}"
    audit_name = f"audit_{run_id}"

    assert data_name in names, f"Data table {data_name} does not exist. Present: {sorted(names)}"
    assert audit_name in names, f"Audit table {audit_name} does not exist. Present: {sorted(names)}"

    assert db.open_table(data_name).count_rows() == 0, f"Data table {data_name} must start empty."
    assert db.open_table(audit_name).count_rows() == 0, f"Audit table {audit_name} must start empty."
