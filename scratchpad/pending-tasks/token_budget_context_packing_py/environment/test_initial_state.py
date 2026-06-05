import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = "/app/lancedb_data"


def test_lancedb_importable():
    assert importlib.import_module("lancedb") is not None, "lancedb is not importable."


def test_openai_importable():
    assert importlib.import_module("openai") is not None, "openai SDK is not importable."


def test_tiktoken_importable():
    assert importlib.import_module("tiktoken") is not None, "tiktoken is not importable."


def test_pyarrow_importable():
    assert importlib.import_module("pyarrow") is not None, "pyarrow is not importable."


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), f"Seeded LanceDB data dir {LANCEDB_DIR} does not exist."


def test_zealt_run_id_env_set():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id and run_id.strip(), "ZEALT_RUN_ID environment variable is not set."


def test_openai_api_key_env_set():
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key and api_key.strip(), "OPENAI_API_KEY environment variable is not set."


def test_chunks_table_seeded():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(LANCEDB_DIR)
    expected = f"chunks_{run_id}"
    names = db.table_names()
    assert expected in names, (
        f"Expected seeded LanceDB table '{expected}' to exist, found tables: {names}"
    )
    tbl = db.open_table(expected)
    assert tbl.count_rows() == 150, (
        f"Expected 150 rows in '{expected}', found {tbl.count_rows()}."
    )
