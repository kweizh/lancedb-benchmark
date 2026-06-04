import csv
import json
import os
import shutil
import subprocess

import pytest

PROJECT_DIR = "/home/user/loader_project"
DB_DIR = os.path.join(PROJECT_DIR, "lance_db")
CSV_PATH = "/app/data/articles.csv"
RUN_ID = os.environ.get("ZEALT_RUN_ID", "local")
TABLE_NAME = f"articles_{RUN_ID}"

QUERY_A = (
    "Photosynthesis is the biochemical process by which chloroplasts in green "
    "plants convert sunlight, carbon dioxide, and water into glucose and oxygen."
)
QUERY_A_EXPECTED_ID = 4242

QUERY_B = (
    "The Apollo 11 mission landed humans on the lunar surface in 1969, with "
    "astronauts Armstrong and Aldrin walking on the Moon."
)
QUERY_B_EXPECTED_ID = 1337

QUERY_C = "Photosynthesis chlorophyll plants sunlight glucose"
QUERY_C_EXPECTED_ID = 4242


def _count_csv_rows(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return sum(1 for _ in reader)


@pytest.fixture(scope="module")
def cleaned_db():
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)
    yield


@pytest.fixture(scope="module")
def ingested(cleaned_db):
    loader = os.path.join(PROJECT_DIR, "loader.py")
    assert os.path.isfile(loader), f"loader.py not found at {loader}"

    result = subprocess.run(
        [
            "python3",
            "loader.py",
            "ingest",
            "--csv",
            CSV_PATH,
            "--table",
            TABLE_NAME,
            "--text-col",
            "body",
            "--batch-size",
            "500",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"`loader.py ingest` failed with exit {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result


def _run_search(query, k):
    loader_dir = PROJECT_DIR
    result = subprocess.run(
        [
            "python3",
            "loader.py",
            "search",
            "--table",
            TABLE_NAME,
            "--query",
            query,
            "--k",
            str(k),
        ],
        capture_output=True,
        text=True,
        cwd=loader_dir,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"`loader.py search` failed with exit {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"search stdout is not valid JSON: {exc}\nRaw stdout:\n{result.stdout}"
        )
    return payload


def test_loader_script_exists():
    loader = os.path.join(PROJECT_DIR, "loader.py")
    assert os.path.isfile(loader), f"Expected loader.py at {loader}"


def test_ingest_runs_successfully(ingested):
    # The fixture itself asserts exit code 0; this test is a smoke check.
    assert ingested.returncode == 0


def test_lance_db_directory_created_after_ingest(ingested):
    assert os.path.isdir(DB_DIR), (
        f"LanceDB directory {DB_DIR} was not created by ingest."
    )


def test_table_row_count_matches_csv(ingested):
    import lancedb

    expected = _count_csv_rows(CSV_PATH)
    db = lancedb.connect(DB_DIR)
    assert TABLE_NAME in db.table_names(), (
        f"Table {TABLE_NAME!r} not found after ingest. "
        f"Existing tables: {db.table_names()}"
    )
    table = db.open_table(TABLE_NAME)
    actual = table.count_rows()
    assert actual == expected, (
        f"Table row count {actual} does not match CSV row count {expected}."
    )


def test_table_schema_has_required_columns(ingested):
    import lancedb

    db = lancedb.connect(DB_DIR)
    table = db.open_table(TABLE_NAME)
    names = set(table.schema.names)
    required = {"id", "title", "body", "category", "published"}
    missing = required - names
    assert not missing, (
        f"Table schema is missing required columns: {missing}. "
        f"Found columns: {sorted(names)}"
    )


def test_table_has_1536d_vector_column(ingested):
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(DB_DIR)
    table = db.open_table(TABLE_NAME)
    schema = table.schema

    found_dim = None
    for field in schema:
        t = field.type
        # FixedSizeList vector column
        if pa.types.is_fixed_size_list(t):
            value_type = t.value_type
            if pa.types.is_floating(value_type) and t.list_size == 1536:
                found_dim = 1536
                break
        # ListType vector column (variable but sample one row to confirm length)
        if pa.types.is_list(t) and pa.types.is_floating(t.value_type):
            row = table.to_pandas().iloc[0]
            vec = row[field.name]
            if vec is not None and len(vec) == 1536:
                found_dim = 1536
                break

    assert found_dim == 1536, (
        f"Could not find a 1536-d float vector column in table schema. "
        f"Schema: {schema}"
    )


def test_search_query_a_returns_expected_top1(ingested):
    payload = _run_search(QUERY_A, 5)
    assert payload.get("query") == QUERY_A, (
        f"search payload 'query' field mismatch: {payload.get('query')!r}"
    )
    assert payload.get("k") == 5, (
        f"search payload 'k' field mismatch: {payload.get('k')!r}"
    )
    results = payload.get("results")
    assert isinstance(results, list), "results must be a list"
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    top = results[0]
    assert int(top["id"]) == QUERY_A_EXPECTED_ID, (
        f"Top-1 id for query A expected {QUERY_A_EXPECTED_ID}, got {top.get('id')}. "
        f"Full results: {results}"
    )
    for item in results:
        for required_key in ("id", "title", "category", "published", "score"):
            assert required_key in item, (
                f"Result row missing required key {required_key!r}: {item}"
            )


def test_search_query_b_returns_expected_top1(ingested):
    payload = _run_search(QUERY_B, 3)
    assert payload.get("query") == QUERY_B
    assert payload.get("k") == 3
    results = payload.get("results")
    assert isinstance(results, list) and len(results) == 3, (
        f"Expected 3 results, got {len(results) if isinstance(results, list) else type(results)}"
    )
    top = results[0]
    assert int(top["id"]) == QUERY_B_EXPECTED_ID, (
        f"Top-1 id for query B expected {QUERY_B_EXPECTED_ID}, got {top.get('id')}. "
        f"Full results: {results}"
    )


def test_search_short_query_returns_expected_top1(ingested):
    payload = _run_search(QUERY_C, 5)
    results = payload.get("results")
    assert isinstance(results, list) and len(results) == 5
    top = results[0]
    assert int(top["id"]) == QUERY_C_EXPECTED_ID, (
        f"Top-1 id for short photosynthesis query expected {QUERY_C_EXPECTED_ID}, "
        f"got {top.get('id')}. Full results: {results}"
    )
