import importlib
import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "data", "lancedb")
EXPECTED_FIXTURE = os.path.join(PROJECT_DIR, ".expected.json")


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), (
        f"Seeded LanceDB directory {LANCEDB_DIR} does not exist; the build-time seed step did not run."
    )


def test_expected_fixture_exists_and_is_well_formed():
    assert os.path.isfile(EXPECTED_FIXTURE), (
        f"Build-time fixture {EXPECTED_FIXTURE} is missing; "
        "the verifier needs it to know the expected retrieval result."
    )
    with open(EXPECTED_FIXTURE) as f:
        payload = json.load(f)
    assert isinstance(payload, dict), "Fixture must be a JSON object."
    assert isinstance(payload.get("question"), str) and payload["question"].strip(), (
        "Fixture must contain a non-empty 'question' string."
    )
    sources = payload.get("expected_sources")
    assert isinstance(sources, list) and len(sources) == 3, (
        "Fixture must contain an 'expected_sources' list of length 3."
    )
    for s in sources:
        assert isinstance(s, str) and s, "Each expected source ID must be a non-empty string."


def test_lancedb_importable():
    importlib.import_module("lancedb")


def test_lancedb_docs_table_has_expected_rows():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    assert "docs" in db.table_names(), "LanceDB table 'docs' was not seeded."
    tbl = db.open_table("docs")
    n = tbl.count_rows()
    assert n >= 20, f"Expected at least 20 seeded rows in 'docs', got {n}."


def test_fastapi_importable():
    importlib.import_module("fastapi")
    importlib.import_module("uvicorn")


def test_openai_sdk_importable():
    importlib.import_module("openai")


def test_openai_api_key_env_present():
    val = os.environ.get("OPENAI_API_KEY", "")
    assert val and val.strip(), "OPENAI_API_KEY environment variable must be set for the candidate task."


def test_verifier_deps_importable():
    importlib.import_module("xprocess")
    importlib.import_module("requests")
