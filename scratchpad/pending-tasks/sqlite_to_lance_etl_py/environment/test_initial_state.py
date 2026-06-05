import importlib
import os
import sqlite3

PROJECT_DIR = "/home/user/myproject"


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert hasattr(mod, "connect"), "lancedb.connect is not available."


def test_pyarrow_importable():
    pa = importlib.import_module("pyarrow")
    assert hasattr(pa, "schema"), "pyarrow.schema is not available."


def test_numpy_importable():
    np = importlib.import_module("numpy")
    assert hasattr(np, "ndarray"), "numpy.ndarray is not available."


def test_pandas_importable():
    pd = importlib.import_module("pandas")
    assert hasattr(pd, "DataFrame"), "pandas.DataFrame is not available."


def test_openai_importable():
    openai = importlib.import_module("openai")
    assert hasattr(openai, "OpenAI"), "openai.OpenAI client is not available."


def test_sqlite3_available():
    # Built into Python's standard library; this should always work.
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE t (a INTEGER)")
    cur.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    rows = list(cur.execute("SELECT a FROM t"))
    conn.close()
    assert rows == [(1,)], "sqlite3 in-memory smoke test failed."


def test_openai_api_key_present():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY must be set in the task environment."


def test_zealt_run_id_present():
    assert os.environ.get("ZEALT_RUN_ID"), "ZEALT_RUN_ID must be set in the task environment."
