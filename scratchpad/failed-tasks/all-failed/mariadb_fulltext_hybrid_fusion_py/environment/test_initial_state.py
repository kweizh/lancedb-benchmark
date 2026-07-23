import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
EXPECTED_JSON = "/app/expected.json"


def _run_id():
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id and os.path.isfile("/logs/artifacts/run-id"):
        with open("/logs/artifacts/run-id") as f:
            run_id = f.read().strip()
    assert run_id, "ZEALT_RUN_ID environment variable is not set."
    return run_id


def _table_name():
    return f"docs_{_run_id()}"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pymysql_importable():
    import pymysql  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_expected_fixture_exists():
    assert os.path.isfile(EXPECTED_JSON), f"Ground-truth fixture {EXPECTED_JSON} is missing."
    with open(EXPECTED_JSON) as f:
        data = json.load(f)
    for key in ("both_id", "keyword_only_id", "vector_only_id", "query_text", "query_vector"):
        assert key in data, f"Fixture {EXPECTED_JSON} missing key '{key}'."
    assert isinstance(data["query_vector"], list) and len(data["query_vector"]) > 0, \
        "query_vector must be a non-empty list."


def test_mariadb_running_and_seeded():
    import pymysql

    conn = pymysql.connect(
        host=os.environ.get("MARIADB_HOST", "127.0.0.1"),
        port=int(os.environ.get("MARIADB_PORT", "3306")),
        user=os.environ["MARIADB_USER"],
        password=os.environ["MARIADB_PASSWORD"],
        database=os.environ["MARIADB_DATABASE"],
    )
    try:
        table = _table_name()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
                (os.environ["MARIADB_DATABASE"], table),
            )
            assert cur.fetchone()[0] == 1, f"MariaDB table {table} does not exist."

            cur.execute(
                "SELECT COUNT(*) FROM information_schema.STATISTICS "
                "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND INDEX_TYPE='FULLTEXT'",
                (os.environ["MARIADB_DATABASE"], table),
            )
            assert cur.fetchone()[0] >= 1, f"MariaDB table {table} has no FULLTEXT index."

            cur.execute(f"SELECT COUNT(*) FROM `{table}`")
            assert cur.fetchone()[0] > 0, f"MariaDB table {table} has no rows."
    finally:
        conn.close()


def test_lancedb_table_seeded():
    import lancedb

    db = lancedb.connect(os.environ["LANCEDB_PATH"])
    table = _table_name()
    assert table in db.table_names(), f"LanceDB table {table} does not exist."
    tbl = db.open_table(table)
    assert tbl.count_rows() > 0, f"LanceDB table {table} has no rows."
