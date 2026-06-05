import importlib
import os

import pytest


PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")


def test_lancedb_importable():
    """The target library (LanceDB) must be importable in the eval environment."""
    mod = importlib.import_module("lancedb")
    assert mod is not None, "Failed to import the `lancedb` package."


def test_tantivy_importable():
    """`tantivy` must be importable so the candidate has the option of using it."""
    mod = importlib.import_module("tantivy")
    assert mod is not None, "Failed to import the `tantivy` package."


def test_jieba_importable():
    """`jieba` must be available for Chinese pre-tokenization."""
    mod = importlib.import_module("jieba")
    assert mod is not None, "Failed to import the `jieba` package."


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_directory_exists():
    assert os.path.isdir(DB_DIR), (
        f"LanceDB data directory {DB_DIR} is missing; the seed step did not run."
    )


def test_zealt_run_id_present():
    assert os.environ.get("ZEALT_RUN_ID"), (
        "ZEALT_RUN_ID environment variable is not set in the task environment."
    )


def test_three_seed_tables_present():
    """All three multilingual tables must already be seeded with 40 rows each."""
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(DB_DIR)
    names = set(db.table_names())
    for lang in ("en", "de", "zh"):
        expected = f"docs_{lang}_{run_id}"
        assert expected in names, (
            f"Seed table {expected!r} is missing. Available tables: {sorted(names)}"
        )
        tbl = db.open_table(expected)
        assert tbl.count_rows() == 40, (
            f"Table {expected!r} should have 40 rows but has {tbl.count_rows()}."
        )
        schema_names = [f.name for f in tbl.schema]
        for col in ("id", "content"):
            assert col in schema_names, (
                f"Table {expected!r} is missing required column {col!r}; "
                f"schema columns are {schema_names}."
            )


def test_solution_module_not_yet_present():
    """The candidate is responsible for creating solution.py; it must not pre-exist."""
    solution_path = os.path.join(PROJECT_DIR, "solution.py")
    assert not os.path.exists(solution_path), (
        f"{solution_path} already exists before the candidate has run; "
        "the seed must not pre-create the solution module."
    )
