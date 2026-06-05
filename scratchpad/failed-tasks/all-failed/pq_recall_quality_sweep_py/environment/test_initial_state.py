import json
import os

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
FIXTURE_DIR = "/app/fixtures"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_fixture_data_exists():
    path = os.path.join(FIXTURE_DIR, "data.npy")
    assert os.path.isfile(path), f"Fixture file {path} is missing."
    arr = np.load(path)
    assert arr.shape == (1024, 64), f"Fixture data has wrong shape: {arr.shape}"
    assert arr.dtype == np.float32, f"Fixture data has wrong dtype: {arr.dtype}"


def test_fixture_queries_exists():
    path = os.path.join(FIXTURE_DIR, "queries.npy")
    assert os.path.isfile(path), f"Fixture file {path} is missing."
    arr = np.load(path)
    assert arr.shape == (30, 64), f"Fixture queries have wrong shape: {arr.shape}"
    assert arr.dtype == np.float32, f"Fixture queries have wrong dtype: {arr.dtype}"


def test_fixture_metadata_exists():
    path = os.path.join(FIXTURE_DIR, "metadata.json")
    assert os.path.isfile(path), f"Fixture file {path} is missing."
    with open(path) as f:
        meta = json.load(f)
    assert meta.get("dim") == 64, "Fixture metadata.dim must be 64."
    assert meta.get("rows") == 1024, "Fixture metadata.rows must be 1024."
    assert meta.get("num_queries") == 30, "Fixture metadata.num_queries must be 30."


def test_zealt_run_id_present():
    assert os.environ.get("ZEALT_RUN_ID"), "ZEALT_RUN_ID environment variable must be set."
