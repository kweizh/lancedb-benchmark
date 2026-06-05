import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_URI = "/app/lancedb_data"
LANCEDB_TABLE = "docs"


def test_lancedb_importable():
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"lancedb is not importable: {exc!r}")


def test_pyarrow_importable():
    try:
        importlib.import_module("pyarrow")
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"pyarrow is not importable: {exc!r}")


def test_nltk_importable():
    try:
        importlib.import_module("nltk")
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"nltk is not importable: {exc!r}")


def test_wordnet_corpus_available_offline():
    from nltk.corpus import wordnet as wn

    syns = wn.synsets("car")
    assert len(syns) > 0, "WordNet corpus is not available offline; expected synsets for 'car'."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_uri_env_set():
    assert os.environ.get("LANCEDB_URI") == LANCEDB_URI, (
        f"LANCEDB_URI environment variable must be set to {LANCEDB_URI}."
    )


def test_lancedb_table_env_set():
    assert os.environ.get("LANCEDB_TABLE") == LANCEDB_TABLE, (
        f"LANCEDB_TABLE environment variable must be set to {LANCEDB_TABLE}."
    )


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_URI), f"LanceDB data directory {LANCEDB_URI} does not exist."


def test_seeded_table_has_100_rows():
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    table_names = db.table_names()
    assert LANCEDB_TABLE in table_names, (
        f"Expected pre-seeded table '{LANCEDB_TABLE}' in {LANCEDB_URI}; found {table_names}."
    )
    table = db.open_table(LANCEDB_TABLE)
    assert table.count_rows() == 100, (
        f"Seeded table '{LANCEDB_TABLE}' must contain 100 rows, found {table.count_rows()}."
    )


def test_seeded_table_schema():
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table(LANCEDB_TABLE)
    field_names = {f.name for f in table.schema}
    assert {"id", "content"}.issubset(field_names), (
        f"Seeded table schema must include 'id' and 'content'; got {field_names}."
    )


def test_no_pre_built_fts_index():
    # The candidate must build the FTS index themselves as part of the task.
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table(LANCEDB_TABLE)
    indices = table.list_indices()
    fts_on_content = [idx for idx in indices if "content" in (idx.columns or [])]
    assert fts_on_content == [], (
        "FTS index on 'content' must NOT exist at initial state; the candidate must build it."
    )


def test_solution_file_not_pre_present():
    solution_path = os.path.join(PROJECT_DIR, "solution.py")
    assert not os.path.exists(solution_path), (
        f"solution.py must not exist at initial state; the candidate creates it. Found: {solution_path}"
    )
