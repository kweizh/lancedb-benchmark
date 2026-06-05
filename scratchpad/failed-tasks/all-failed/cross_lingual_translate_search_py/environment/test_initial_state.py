import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb must be importable in the environment."


def test_openai_importable():
    mod = importlib.import_module("openai")
    assert mod is not None, "openai must be importable in the environment."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_openai_api_key_set():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY must be set in the environment."


def test_lancedb_uri_set():
    uri = os.environ.get("LANCEDB_URI")
    assert uri, "LANCEDB_URI must be set in the environment."
    assert os.path.isdir(uri), f"LANCEDB_URI directory {uri} must exist (pre-seeded corpus)."


def test_lancedb_table_prefix_set():
    assert os.environ.get("LANCEDB_TABLE_PREFIX"), "LANCEDB_TABLE_PREFIX must be set in the environment."


def test_zealt_run_id_set():
    assert os.environ.get("ZEALT_RUN_ID"), "ZEALT_RUN_ID must be set in the environment."


def test_pre_seeded_corpus_table():
    import lancedb

    uri = os.environ["LANCEDB_URI"]
    prefix = os.environ["LANCEDB_TABLE_PREFIX"]
    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"{prefix}{run_id}"
    db = lancedb.connect(uri)
    names = db.table_names()
    assert table_name in names, (
        f"Pre-seeded LanceDB table '{table_name}' must already exist under {uri}. "
        f"Found tables: {names}"
    )
    tbl = db.open_table(table_name)
    n = tbl.count_rows()
    assert n == 40, f"Pre-seeded corpus table must have exactly 40 rows; found {n}."
