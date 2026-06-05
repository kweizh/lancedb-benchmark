import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb module is not importable."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow module is not importable."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy module is not importable."


def test_project_directory_exists():
    assert os.path.isdir(
        PROJECT_DIR
    ), f"Project directory {PROJECT_DIR} does not exist."


def test_env_lance_db_uri_set():
    assert os.environ.get(
        "LANCE_DB_URI"
    ), "Environment variable LANCE_DB_URI must be set."


def test_env_movies_table_set():
    assert os.environ.get(
        "MOVIES_TABLE"
    ), "Environment variable MOVIES_TABLE must be set."


def test_env_prefix_table_set():
    assert os.environ.get(
        "PREFIX_TABLE"
    ), "Environment variable PREFIX_TABLE must be set."


def test_lance_db_uri_exists_on_disk():
    uri = os.environ["LANCE_DB_URI"]
    assert os.path.isdir(uri), f"LANCE_DB_URI path {uri} does not exist on disk."


def test_seed_tables_present():
    import lancedb

    db = lancedb.connect(os.environ["LANCE_DB_URI"])
    names = set(db.table_names())
    movies_tbl = os.environ["MOVIES_TABLE"]
    prefix_tbl = os.environ["PREFIX_TABLE"]
    assert (
        movies_tbl in names
    ), f"Expected seeded table '{movies_tbl}' but found tables: {sorted(names)}"
    assert (
        prefix_tbl in names
    ), f"Expected seeded table '{prefix_tbl}' but found tables: {sorted(names)}"


def test_seed_tables_row_counts():
    import lancedb

    db = lancedb.connect(os.environ["LANCE_DB_URI"])
    movies = db.open_table(os.environ["MOVIES_TABLE"])
    prefix = db.open_table(os.environ["PREFIX_TABLE"])
    assert (
        movies.count_rows() == 500
    ), f"movies table should have 500 rows, got {movies.count_rows()}"
    assert (
        prefix.count_rows() == 50
    ), f"prefix_vectors table should have 50 rows, got {prefix.count_rows()}"
