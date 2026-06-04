import json
import os

import pytest


PROJECT_DIR = "/home/user/myproject"
WORKSPACE_DB = "/workspace/db"
EXPECTED_PATH = "/home/user/myproject/.expected.json"
SEED_SCRIPT = "/opt/seed_state.py"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_lancedb_version_pinned():
    import lancedb

    assert lancedb.__version__.startswith("0.25."), (
        f"Expected lancedb 0.25.x, got {lancedb.__version__}."
    )


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} does not exist."
    )


def test_workspace_db_dir_exists():
    assert os.path.isdir(WORKSPACE_DB), (
        f"LanceDB workspace directory {WORKSPACE_DB} does not exist."
    )


def test_seed_script_present():
    assert os.path.isfile(SEED_SCRIPT), (
        f"Seed script {SEED_SCRIPT} must be baked into the image."
    )


def test_expected_fixture_present():
    assert os.path.isfile(EXPECTED_PATH), (
        f"Expected fixture {EXPECTED_PATH} must be written by the seed step."
    )
    with open(EXPECTED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), (
        f"{EXPECTED_PATH} must contain a JSON object."
    )
    assert "sentinel_id" in data and isinstance(data["sentinel_id"], int), (
        f"{EXPECTED_PATH} must contain integer 'sentinel_id'."
    )
    assert (
        "query_vector" in data
        and isinstance(data["query_vector"], list)
        and len(data["query_vector"]) == 64
    ), f"{EXPECTED_PATH} must contain a 64-element list 'query_vector'."


def test_vectors_table_seeded_with_100_rows_and_no_index():
    import lancedb

    db = lancedb.connect(WORKSPACE_DB)
    names = db.table_names()
    assert "vectors" in names, (
        f"Expected table 'vectors' to be seeded in {WORKSPACE_DB}; found {names}."
    )
    table = db.open_table("vectors")
    count = table.count_rows()
    assert count == 100, (
        f"Expected the seeded 'vectors' table to start with exactly 100 rows, got {count}."
    )
    indices = table.list_indices()
    assert len(indices) == 0, (
        f"Expected the seeded 'vectors' table to have NO vector index, got {indices}."
    )


def test_index_build_log_not_yet_created():
    import lancedb

    db = lancedb.connect(WORKSPACE_DB)
    names = db.table_names()
    assert "index_build_log" not in names, (
        "The audit log table 'index_build_log' must NOT exist before the executor runs."
    )


def test_solution_module_not_yet_created():
    solution_path = os.path.join(PROJECT_DIR, "solution.py")
    assert not os.path.exists(solution_path), (
        f"Solution module {solution_path} must be created by the executor, not pre-seeded."
    )
