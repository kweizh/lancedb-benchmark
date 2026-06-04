import importlib
import os

import pytest


WORKSPACE = "/workspace"


def test_lancedb_importable():
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"lancedb python package is not importable: {exc!r}")


def test_pyarrow_importable():
    try:
        importlib.import_module("pyarrow")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"pyarrow python package is not importable: {exc!r}")


def test_numpy_importable():
    try:
        importlib.import_module("numpy")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"numpy python package is not importable: {exc!r}")


def test_tantivy_not_installed():
    # The task explicitly requires the native (non-Tantivy) FTS backend.
    # The environment must not pre-install tantivy to keep the constraint honest.
    spec = importlib.util.find_spec("tantivy")
    assert spec is None, (
        "tantivy must NOT be installed in this environment; the task requires the "
        "native Lance FTS backend (use_tantivy=False)."
    )


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_lancedb_dir_is_clean_or_absent():
    # The candidate solution is expected to (re)create /workspace/db. We do not
    # pre-seed it, so the directory must either be absent or empty before the run.
    db_path = os.path.join(WORKSPACE, "db")
    if os.path.exists(db_path):
        assert os.path.isdir(db_path), f"{db_path} exists but is not a directory."
        assert os.listdir(db_path) == [], (
            f"{db_path} must be empty before the task runs; found pre-existing entries."
        )


def test_output_file_not_present_yet():
    # The candidate solution is responsible for creating the results file.
    results_path = os.path.join(WORKSPACE, "output", "fts_results.json")
    assert not os.path.exists(results_path), (
        f"{results_path} must not exist before the task runs."
    )
