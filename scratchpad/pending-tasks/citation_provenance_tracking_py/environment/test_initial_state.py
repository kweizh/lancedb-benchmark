import os

import pytest

PROJECT_DIR = "/home/user/myproject"
SOURCE_DOCS_DIR = "/app/source_documents"
LANCEDB_DIR = "/app/lancedb_data"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_openai_importable():
    import openai  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_source_documents_dir_seeded():
    assert os.path.isdir(SOURCE_DOCS_DIR), f"Seed directory {SOURCE_DOCS_DIR} is missing."
    txt_files = [f for f in os.listdir(SOURCE_DOCS_DIR) if f.endswith(".txt")]
    assert len(txt_files) == 15, (
        f"Expected 15 seeded source documents in {SOURCE_DOCS_DIR}, found {len(txt_files)}."
    )


def test_source_documents_non_empty():
    for fname in os.listdir(SOURCE_DOCS_DIR):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(SOURCE_DOCS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        assert len(text) > 200, f"Source document {fname} is too short ({len(text)} chars)."


def test_lancedb_data_seeded():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB directory {LANCEDB_DIR} is missing."


def test_chunks_table_seeded():
    import lancedb

    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable must be set."
    db = lancedb.connect(LANCEDB_DIR)
    table_name = f"chunks_{run_id}"
    names = db.table_names()
    assert table_name in names, (
        f"Expected seeded LanceDB table '{table_name}', found tables: {names}."
    )
    tbl = db.open_table(table_name)
    assert tbl.count_rows() >= 30, (
        f"Expected seeded table to have ≥30 chunk rows, found {tbl.count_rows()}."
    )


def test_openai_api_key_present():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY must be set at runtime."
