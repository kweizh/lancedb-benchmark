import csv
import os
import shutil

import pytest

PROJECT_DIR = "/home/user/loader_project"
CSV_PATH = "/app/data/articles.csv"


def test_python3_available():
    assert shutil.which("python3") is not None, "python3 binary not found in PATH."


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_openai_importable():
    import openai  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} does not exist."
    )


def test_csv_dataset_exists():
    assert os.path.isfile(CSV_PATH), (
        f"Pre-baked CSV dataset {CSV_PATH} does not exist."
    )


def test_csv_dataset_has_expected_columns():
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    expected = ["id", "title", "body", "category", "published"]
    assert header == expected, (
        f"CSV header {header!r} does not match expected {expected!r}."
    )


def test_csv_dataset_has_5000_rows():
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        count = sum(1 for _ in reader)
    assert count == 5000, f"CSV expected to contain 5000 data rows, found {count}."


def test_openai_api_key_present():
    assert os.environ.get("OPENAI_API_KEY"), (
        "OPENAI_API_KEY environment variable must be set in the task environment."
    )


def test_lance_db_directory_clean():
    db_dir = os.path.join(PROJECT_DIR, "lance_db")
    # The LanceDB directory should NOT pre-exist; the executor creates it.
    if os.path.exists(db_dir):
        # If for some reason it exists, it must at least be empty.
        assert not os.listdir(db_dir), (
            f"LanceDB directory {db_dir} should be empty before the task starts."
        )
