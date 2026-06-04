import json
import os
import shutil
import subprocess

import pytest

PROJECT_DIR = "/home/user/myproject"
SEARCH_SCRIPT = os.path.join(PROJECT_DIR, "search.py")
LANCEDB_DATA_DIR = os.path.join(PROJECT_DIR, "lancedb_data")


@pytest.fixture(scope="session", autouse=True)
def clean_lancedb_data():
    """Remove any cached LanceDB data so the first invocation in this verifier
    session exercises the index-creation code path."""
    if os.path.isdir(LANCEDB_DATA_DIR):
        shutil.rmtree(LANCEDB_DATA_DIR)
    yield


def _run_search(query: str, k: int):
    assert os.path.isfile(SEARCH_SCRIPT), (
        f"search.py not found at {SEARCH_SCRIPT}"
    )
    env = os.environ.copy()
    assert env.get("ZEALT_RUN_ID"), (
        "ZEALT_RUN_ID environment variable must be set for verification."
    )
    result = subprocess.run(
        ["python3", "search.py", "--query", query, "--k", str(k)],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"search.py failed for query {query!r} (exit {result.returncode}).\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    # Some libraries print warnings to stdout; locate the JSON payload.
    stdout = result.stdout.strip()
    # Try parsing whole stdout first; if that fails, try to find a trailing JSON array.
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.rfind("[")
        end = stdout.rfind("]")
        assert start != -1 and end != -1 and end > start, (
            f"Could not locate JSON list in stdout for query {query!r}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        parsed = json.loads(stdout[start : end + 1])
    assert isinstance(parsed, list), (
        f"search.py must print a JSON list to stdout. Got: {type(parsed).__name__}"
    )
    return parsed


def _assert_result_shape(results, query):
    for i, row in enumerate(results):
        assert isinstance(row, dict), (
            f"Result {i} for query {query!r} is not an object: {row!r}"
        )
        assert "id" in row, f"Result {i} for query {query!r} missing 'id'"
        assert isinstance(row["id"], int), (
            f"Result {i} for query {query!r} has non-integer id: {row['id']!r}"
        )
        assert "title" in row and isinstance(row["title"], str), (
            f"Result {i} for query {query!r} missing string 'title'"
        )
        assert "body" in row and isinstance(row["body"], str), (
            f"Result {i} for query {query!r} missing string 'body'"
        )


def test_phrase_query_returns_phrase_match():
    results = _run_search('"vector database"', 5)
    assert len(results) >= 1, (
        "Phrase query must return at least one result."
    )
    _assert_result_shape(results, '"vector database"')
    assert results[0]["id"] == 57, (
        "Top-1 for the phrase query '\"vector database\"' must be the rigged "
        "document with id=57 (the only seeded doc whose body contains the "
        f"contiguous phrase 'vector database'). Got id={results[0]['id']}."
    )


def test_field_boost_query_promotes_title_match():
    query = "title:lancedb^3 body:vector"
    results = _run_search(query, 5)
    assert len(results) >= 1, (
        "Field-boost query must return at least one result."
    )
    _assert_result_shape(results, query)
    assert results[0]["id"] == 58, (
        "Top-1 for the field-boost query 'title:lancedb^3 body:vector' must "
        "be the rigged document with id=58 (the only seeded doc with "
        f"'lancedb' in its title). Got id={results[0]['id']}."
    )


def test_boolean_and_query_returns_both_terms_doc():
    query = "+rust +tantivy"
    results = _run_search(query, 5)
    assert len(results) >= 1, (
        "Boolean AND query must return at least one result."
    )
    _assert_result_shape(results, query)
    assert results[0]["id"] == 59, (
        "Top-1 for the boolean AND query '+rust +tantivy' must be the rigged "
        "document with id=59 (the only seeded doc whose body contains both "
        f"'rust' and 'tantivy'). Got id={results[0]['id']}."
    )


def test_tantivy_fts_index_was_created():
    """Confirm a Tantivy-backed FTS index was built on the table.

    The Tantivy backend (`use_tantivy=True`) stores its index files under
    ``<table>.lance/_indices/fts/`` and does NOT register the index in
    LanceDB's metadata-level index list (`Table.list_indices()` returns an
    empty list for Tantivy FTS indices in lancedb 0.25.3). We therefore check
    for the presence of the on-disk ``_indices/fts`` directory and verify it
    is non-empty.
    """
    import lancedb

    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID env var must be set for verification."

    db = lancedb.connect(LANCEDB_DATA_DIR)
    table_name = f"docs_{run_id}"
    assert table_name in db.table_names(), (
        f"Expected LanceDB table '{table_name}' to exist after running search.py. "
        f"Available tables: {db.table_names()}"
    )
    fts_dir = os.path.join(
        LANCEDB_DATA_DIR, f"{table_name}.lance", "_indices", "fts"
    )
    assert os.path.isdir(fts_dir), (
        "Expected a Tantivy-backed FTS index directory at "
        f"'{fts_dir}'. The candidate must build the FTS index with "
        "`use_tantivy=True`."
    )
    contents = os.listdir(fts_dir)
    assert contents, (
        f"Tantivy FTS index directory '{fts_dir}' is empty; expected the "
        "Tantivy backend to have written index segment files."
    )


def test_row_count_after_first_run():
    """The first search.py invocation must have ingested all 60 seed documents."""
    import lancedb

    run_id = os.environ.get("ZEALT_RUN_ID")
    db = lancedb.connect(LANCEDB_DATA_DIR)
    table_name = f"docs_{run_id}"
    table = db.open_table(table_name)
    assert table.count_rows() == 60, (
        f"Expected 60 rows in table '{table_name}', got {table.count_rows()}."
    )
