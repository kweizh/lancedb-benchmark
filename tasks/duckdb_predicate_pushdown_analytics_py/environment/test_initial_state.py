import csv
import importlib.util
import json
import os

PROJECT_DIR = "/home/user/project"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.jsonl")
CATEGORIES_PATH = os.path.join(DATA_DIR, "categories.csv")

REQUIRED_DOC_KEYS = {"id", "title", "category", "price", "in_stock", "vector"}
VECTOR_DIM = 8


def test_lancedb_importable():
    assert importlib.util.find_spec("lancedb") is not None, \
        "The lancedb Python package is not importable in the environment."


def test_duckdb_importable():
    assert importlib.util.find_spec("duckdb") is not None, \
        "The duckdb Python package is not importable in the environment."


def test_pyarrow_importable():
    assert importlib.util.find_spec("pyarrow") is not None, \
        "The pyarrow Python package is not importable in the environment."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_data_dir_exists():
    assert os.path.isdir(DATA_DIR), f"Data directory {DATA_DIR} does not exist."


def test_documents_file_exists():
    assert os.path.isfile(DOCUMENTS_PATH), f"Documents file {DOCUMENTS_PATH} does not exist."


def test_categories_file_exists():
    assert os.path.isfile(CATEGORIES_PATH), f"Categories file {CATEGORIES_PATH} does not exist."


def test_documents_structure():
    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) > 0, f"{DOCUMENTS_PATH} contains no data rows."
    for ln in lines:
        obj = json.loads(ln)
        assert REQUIRED_DOC_KEYS.issubset(obj.keys()), (
            f"A document row is missing required keys; expected {sorted(REQUIRED_DOC_KEYS)}, "
            f"got {sorted(obj.keys())}."
        )
        assert isinstance(obj["vector"], list) and len(obj["vector"]) == VECTOR_DIM, (
            f"Document id={obj.get('id')} must have a vector of length {VECTOR_DIM}."
        )


def test_categories_structure():
    with open(CATEGORIES_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    assert {"category", "department", "tax_rate"}.issubset(set(fieldnames)), (
        f"categories.csv must contain columns 'category', 'department', 'tax_rate'; got {fieldnames}."
    )
    assert len(rows) > 0, "categories.csv contains no data rows."


def test_at_least_one_category_has_apostrophe():
    with open(CATEGORIES_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        categories = [row["category"] for row in reader]
    assert any("'" in c for c in categories), (
        "Expected at least one category name containing an apostrophe to exercise SQL escaping."
    )
