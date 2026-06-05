import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "lancedb")
EXPECTED_PATH = os.path.join(PROJECT_DIR, ".expected.json")


def test_lancedb_import():
    import lancedb  # noqa: F401


def test_openai_import():
    import openai  # noqa: F401


def test_pyarrow_import():
    import pyarrow  # noqa: F401


def test_openai_api_key_env():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY environment variable is not set."


def test_zealt_run_id_env():
    assert os.environ.get("ZEALT_RUN_ID"), "ZEALT_RUN_ID environment variable is not set."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(DATA_DIR), f"LanceDB data directory {DATA_DIR} does not exist."


def test_expected_anchor_file_exists():
    assert os.path.isfile(EXPECTED_PATH), f"Expected anchors file {EXPECTED_PATH} does not exist."
    with open(EXPECTED_PATH) as f:
        payload = json.load(f)
    assert "anchors" in payload, "anchors key missing from .expected.json"
    assert isinstance(payload["anchors"], list), ".expected.json anchors must be a list"
    assert len(payload["anchors"]) == 8, "expected exactly 8 anchor queries"


def test_products_table_seeded():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(DATA_DIR)
    table_name = f"products_{run_id}"
    assert table_name in db.table_names(), f"Expected table {table_name} to exist."
    tbl = db.open_table(table_name)
    assert tbl.count_rows() == 500, "Expected exactly 500 seeded product rows."
    schema_names = {f.name for f in tbl.schema}
    expected = {"id", "sku", "name", "description", "category", "price", "release_date", "vector"}
    assert expected.issubset(schema_names), (
        f"Schema columns missing: {expected - schema_names}"
    )


def test_fts_index_built():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(DATA_DIR)
    tbl = db.open_table(f"products_{run_id}")
    index_names = [idx.name for idx in tbl.list_indices()]
    assert any("description" in name for name in index_names), (
        f"Expected an FTS index on description column, found {index_names}"
    )
