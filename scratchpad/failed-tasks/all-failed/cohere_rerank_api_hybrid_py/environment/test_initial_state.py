import importlib
import os

import pytest

PROJECT_DIR = "/home/user/cohere_rerank"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb")


def test_lancedb_importable():
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"lancedb is not importable: {exc!r}")


def test_cohere_importable():
    try:
        importlib.import_module("cohere")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"cohere is not importable: {exc!r}")


def test_openai_importable():
    try:
        importlib.import_module("openai")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"openai is not importable: {exc!r}")


def test_cohere_reranker_importable():
    try:
        from lancedb.rerankers import CohereReranker  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"lancedb.rerankers.CohereReranker is not importable: {exc!r}")


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_directory_exists():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB directory {LANCEDB_DIR} does not exist."


def test_docs_table_seeded_with_200_rows():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    assert "docs" in db.table_names(), "LanceDB table 'docs' is missing from the seeded database."
    tbl = db.open_table("docs")
    n = tbl.count_rows()
    assert n == 200, f"Expected 200 seeded rows in 'docs' table, found {n}."


def test_docs_table_has_three_languages():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table("docs")
    en = tbl.count_rows(filter="language = 'en'")
    es = tbl.count_rows(filter="language = 'es'")
    fr = tbl.count_rows(filter="language = 'fr'")
    assert en == 80, f"Expected 80 English rows, found {en}."
    assert es == 60, f"Expected 60 Spanish rows, found {es}."
    assert fr == 60, f"Expected 60 French rows, found {fr}."


def test_rigged_doc_present():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table("docs")
    rows = tbl.search().where("id = 'install-linux-guide'").limit(1).to_list()
    assert len(rows) == 1, "Rigged ground-truth document 'install-linux-guide' is missing from the table."
    assert rows[0]["language"] == "en", "Rigged ground-truth document must have language='en'."


def test_openai_api_key_present():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY environment variable is not set."


def test_cohere_api_key_present():
    assert os.environ.get("COHERE_API_KEY"), "COHERE_API_KEY environment variable is not set."
