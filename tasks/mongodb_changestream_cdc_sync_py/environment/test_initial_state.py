import importlib
import os
import shutil

import pytest

PROJECT_DIR = "/home/user/project"
START_SCRIPT = "/usr/local/bin/start-mongo.sh"


def test_lancedb_importable():
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"lancedb is not importable: {exc}")


def test_pymongo_importable():
    try:
        importlib.import_module("pymongo")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"pymongo is not importable: {exc}")


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_mongod_binary_available():
    assert shutil.which("mongod") is not None, "mongod binary not found in PATH."


def test_mongosh_binary_available():
    assert shutil.which("mongosh") is not None, "mongosh binary not found in PATH."


def test_start_mongo_helper_present_and_executable():
    assert os.path.isfile(START_SCRIPT), (
        f"MongoDB startup helper {START_SCRIPT} does not exist."
    )
    assert os.access(START_SCRIPT, os.X_OK), (
        f"MongoDB startup helper {START_SCRIPT} is not executable."
    )
