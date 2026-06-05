import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb is not importable in the runtime environment."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow is not importable in the runtime environment."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy is not importable in the runtime environment."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB data directory {LANCEDB_DIR} does not exist."


def test_zealt_run_id_env_set():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable is not set."


def test_source_embeddings_table_seeded():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(LANCEDB_DIR)
    table_name = f"embeddings_{run_id}"
    assert table_name in db.table_names(), (
        f"Source table {table_name} was not seeded in {LANCEDB_DIR}."
    )
    tbl = db.open_table(table_name)
    assert tbl.count_rows() == 500, (
        f"Source table {table_name} must contain 500 pre-seeded rows."
    )
    schema = tbl.schema
    field_names = set(schema.names)
    assert {"id", "vector"}.issubset(field_names), (
        f"Source table {table_name} must expose at least an `id` and `vector` column; got {field_names}."
    )


def test_knn_edges_table_not_yet_present():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(LANCEDB_DIR)
    target = f"knn_edges_{run_id}"
    assert target not in db.table_names(), (
        f"Output table {target} should not exist before the executor builds the graph."
    )


def test_solution_module_not_yet_present():
    # The executor is expected to create solution.py; verify it is not already shipped.
    path = os.path.join(PROJECT_DIR, "solution.py")
    assert not os.path.exists(path), (
        "solution.py should not be pre-created; the executor is expected to author it."
    )
