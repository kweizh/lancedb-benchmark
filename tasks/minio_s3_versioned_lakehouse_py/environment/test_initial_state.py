import importlib
import json
import os
import shutil
import time
import urllib.request

import pytest

PROJECT_DIR = "/home/user/myproject"
FIXTURES_DIR = "/app/fixtures"
MINIO_ENDPOINT = "http://127.0.0.1:9000"


def test_lancedb_importable():
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"lancedb could not be imported: {exc}")


def test_supporting_libraries_importable():
    for mod in ("pyarrow", "numpy"):
        try:
            importlib.import_module(mod)
        except Exception as exc:  # pragma: no cover - diagnostic
            pytest.fail(f"Required library '{mod}' could not be imported: {exc}")


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_mc_binary_available():
    assert shutil.which("mc") is not None, "MinIO client 'mc' binary not found in PATH."


def test_minio_server_reachable():
    last_err = None
    for _ in range(30):
        try:
            with urllib.request.urlopen(
                f"{MINIO_ENDPOINT}/minio/health/live", timeout=3
            ) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - diagnostic
            last_err = exc
            time.sleep(1)
    pytest.fail(f"MinIO server not reachable at {MINIO_ENDPOINT}: {last_err}")


def test_fixture_files_exist():
    for name in ("base.json", "added.json", "queries.json"):
        path = os.path.join(FIXTURES_DIR, name)
        assert os.path.isfile(path), f"Expected fixture file {path} to exist."


def test_base_fixture_structure():
    with open(os.path.join(FIXTURES_DIR, "base.json")) as f:
        rows = json.load(f)
    assert isinstance(rows, list) and len(rows) > 0, "base.json must be a non-empty JSON array."
    row = rows[0]
    for key in ("id", "text", "category", "vector"):
        assert key in row, f"base.json rows must contain the key '{key}'."
    assert isinstance(row["vector"], list) and len(row["vector"]) == 8, \
        "base.json vectors must be lists of 8 floats."


def test_added_fixture_structure():
    with open(os.path.join(FIXTURES_DIR, "added.json")) as f:
        rows = json.load(f)
    assert isinstance(rows, list) and len(rows) > 0, "added.json must be a non-empty JSON array."
    for key in ("id", "text", "category", "vector"):
        assert key in rows[0], f"added.json rows must contain the key '{key}'."


def test_queries_fixture_structure():
    with open(os.path.join(FIXTURES_DIR, "queries.json")) as f:
        queries = json.load(f)
    assert isinstance(queries, dict), "queries.json must be a JSON object."
    for name in ("q_added", "q_deleted"):
        assert name in queries, f"queries.json must define the query '{name}'."
        assert isinstance(queries[name], list) and len(queries[name]) == 8, \
            f"query '{name}' must be a list of 8 floats."
