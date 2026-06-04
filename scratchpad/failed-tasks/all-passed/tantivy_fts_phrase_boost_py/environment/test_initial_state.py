import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
SEED_PATH = os.path.join(PROJECT_DIR, "seed", "docs.json")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_tantivy_importable():
    import tantivy  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} does not exist."
    )


def test_seed_file_exists():
    assert os.path.isfile(SEED_PATH), (
        f"Seed corpus {SEED_PATH} does not exist."
    )


def test_seed_file_has_60_rows():
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    assert isinstance(docs, list), "Seed corpus must be a JSON list."
    assert len(docs) == 60, f"Seed corpus must contain 60 documents, got {len(docs)}."
    for row in docs:
        assert "id" in row and isinstance(row["id"], int), (
            "Each seeded doc must have an integer `id`."
        )
        assert "title" in row and isinstance(row["title"], str), (
            "Each seeded doc must have a string `title`."
        )
        assert "body" in row and isinstance(row["body"], str), (
            "Each seeded doc must have a string `body`."
        )


def test_zealt_run_id_env_var_set():
    assert os.environ.get("ZEALT_RUN_ID"), (
        "ZEALT_RUN_ID environment variable must be set before running this task."
    )
