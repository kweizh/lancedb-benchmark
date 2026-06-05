import importlib
import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = "/app/lancedb_data"
EXPECTED_FIXTURE = "/app/.expected.json"


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb is not importable in the environment."


def test_openai_importable():
    mod = importlib.import_module("openai")
    assert mod is not None, "openai SDK is not importable in the environment."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), (
        f"LanceDB seeded data directory {LANCEDB_DIR} does not exist; "
        "the build-time seed step must have populated it."
    )


def test_expected_fixture_exists_and_valid():
    assert os.path.isfile(EXPECTED_FIXTURE), (
        f"Expected fixture {EXPECTED_FIXTURE} does not exist; "
        "the build-time seed step must have produced it."
    )
    with open(EXPECTED_FIXTURE) as f:
        data = json.load(f)
    assert "rigged_id" in data and isinstance(data["rigged_id"], int), (
        "Expected fixture must contain integer 'rigged_id'."
    )
    assert data.get("query") == "GC differences between languages?", (
        "Expected fixture must contain the exact verifier query."
    )


def test_programming_qa_table_present():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    names = db.table_names()
    assert "programming_qa" in names, (
        f"LanceDB table 'programming_qa' is missing from {LANCEDB_DIR} (found: {names})."
    )
    tbl = db.open_table("programming_qa")
    assert tbl.count_rows() == 30, (
        f"Expected the seeded 'programming_qa' table to have 30 rows, got {tbl.count_rows()}."
    )


def test_openai_api_key_available():
    key = os.environ.get("OPENAI_API_KEY", "")
    assert key and len(key) > 10, (
        "OPENAI_API_KEY environment variable is missing or implausibly short."
    )
