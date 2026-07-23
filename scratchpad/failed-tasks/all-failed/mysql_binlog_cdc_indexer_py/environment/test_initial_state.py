import os
import time

import pytest

PROJECT_DIR = "/home/user/myproject"


def _mysql_settings():
    return {
        "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.environ.get("MYSQL_PORT", "3306")),
        "user": os.environ.get("MYSQL_USER", "cdc"),
        "password": os.environ.get("MYSQL_PASSWORD", "cdcpass"),
    }


def _connect_mysql(retries=60, delay=1.0):
    import pymysql

    last_err = None
    for _ in range(retries):
        try:
            conn = pymysql.connect(
                host=_mysql_settings()["host"],
                port=_mysql_settings()["port"],
                user=_mysql_settings()["user"],
                password=_mysql_settings()["password"],
                autocommit=True,
            )
            return conn
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(delay)
    raise AssertionError(f"Could not connect to local MySQL server: {last_err}")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pymysqlreplication_importable():
    from pymysqlreplication import BinLogStreamReader  # noqa: F401
    from pymysqlreplication.row_event import (  # noqa: F401
        DeleteRowsEvent,
        UpdateRowsEvent,
        WriteRowsEvent,
    )


def test_pymysql_importable():
    import pymysql  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_mysql_env_vars_present():
    for var in ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD"):
        assert os.environ.get(var), f"Environment variable {var} is not set."


def test_mysql_server_running_and_reachable():
    conn = _connect_mysql()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1
    finally:
        conn.close()


def test_source_table_exists_and_empty():
    conn = _connect_mysql()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema='appdb' AND table_name='documents'"
            )
            assert cur.fetchone()[0] == 1, "Source table appdb.documents does not exist."
            cur.execute("SELECT COUNT(*) FROM appdb.documents")
            assert cur.fetchone()[0] == 0, "Source table appdb.documents should start empty."
    finally:
        conn.close()


def test_binlog_row_based_replication_enabled():
    conn = _connect_mysql()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW VARIABLES LIKE 'log_bin'")
            assert cur.fetchone()[1].upper() == "ON", "Binary logging (log_bin) is not enabled."
            cur.execute("SHOW VARIABLES LIKE 'binlog_format'")
            assert cur.fetchone()[1].upper() == "ROW", "binlog_format must be ROW."
            cur.execute("SHOW VARIABLES LIKE 'binlog_row_image'")
            assert cur.fetchone()[1].upper() == "FULL", "binlog_row_image must be FULL."
    finally:
        conn.close()


def test_replication_privileges_available():
    conn = _connect_mysql()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW GRANTS")
            grants = " ".join(row[0] for row in cur.fetchall()).upper()
            assert "REPLICATION SLAVE" in grants, "MySQL user lacks REPLICATION SLAVE privilege."
            assert "REPLICATION CLIENT" in grants, "MySQL user lacks REPLICATION CLIENT privilege."
    finally:
        conn.close()
