import os
import shutil

import pytest

PROJECT_DIR = "/home/user/myproject"
DATA_DIR = "/home/user/myproject/data"


def test_python3_available():
    assert shutil.which("python3") is not None, "python3 binary not found in PATH."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(DATA_DIR), (
        f"LanceDB data directory {DATA_DIR} does not exist. The seeded `documents_${{ZEALT_RUN_ID}}` "
        "table must already be present here before the task starts."
    )


def test_lancedb_importable():
    try:
        import lancedb  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"`import lancedb` failed: {exc!r}")


def test_fastapi_importable():
    try:
        import fastapi  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"`import fastapi` failed: {exc!r}")


def test_uvicorn_importable():
    try:
        import uvicorn  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"`import uvicorn` failed: {exc!r}")


def test_strawberry_importable():
    try:
        import strawberry  # noqa: F401
        from strawberry.fastapi import GraphQLRouter  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"`import strawberry` (with FastAPI extras) failed: {exc!r}")


def test_openai_importable():
    try:
        import openai  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"`import openai` failed: {exc!r}")


def test_zealt_run_id_set():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable must be set before the task starts."


def test_openai_api_key_set():
    key = os.environ.get("OPENAI_API_KEY")
    assert key, "OPENAI_API_KEY environment variable must be set so embeddings can be computed at runtime."


def test_seeded_documents_table_present():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable must be set before the task starts."

    import lancedb

    db = lancedb.connect(DATA_DIR)
    table_name = f"documents_{run_id}"
    assert table_name in db.table_names(), (
        f"Expected seeded LanceDB table `{table_name}` to be present in {DATA_DIR}. "
        f"Found tables: {db.table_names()!r}"
    )
    tbl = db.open_table(table_name)
    nrows = tbl.count_rows()
    assert nrows >= 300, f"Expected the seeded table to have at least 300 rows, got {nrows}."
