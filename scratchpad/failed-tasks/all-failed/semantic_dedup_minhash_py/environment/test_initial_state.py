import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = "/app/lancedb_data"


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb is not importable in the environment."


def test_datasketch_importable():
    mod = importlib.import_module("datasketch")
    assert hasattr(mod, "MinHashLSH"), "datasketch.MinHashLSH is not available."
    assert hasattr(mod, "MinHash"), "datasketch.MinHash is not available."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy is not importable in the environment."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB data directory {LANCEDB_DIR} does not exist."


def test_seeded_table_present():
    import lancedb

    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable must be set in the initial state."

    db = lancedb.connect(LANCEDB_DIR)
    table_name = f"documents_{run_id}"
    names = db.table_names()
    assert table_name in names, (
        f"Expected seeded table '{table_name}' to exist in {LANCEDB_DIR}; found tables: {names!r}."
    )

    tbl = db.open_table(table_name)
    assert tbl.count_rows() == 300, (
        f"Expected seeded table to contain 300 rows, found {tbl.count_rows()}."
    )

    schema_names = {f.name for f in tbl.schema}
    for required in ("id", "text", "vector"):
        assert required in schema_names, (
            f"Seeded table is missing required column '{required}'. Schema: {schema_names!r}."
        )


def test_ground_truth_artifact_present():
    # The seeded fixture records the ground-truth duplicate pairs so the
    # verifier (not the candidate) can confirm correctness without re-deriving
    # the corpus. This file is part of the initial state.
    gt_path = "/app/ground_truth_pairs.json"
    assert os.path.isfile(gt_path), (
        f"Expected ground-truth pairs file at {gt_path} to be present in the initial state."
    )
