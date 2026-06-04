import importlib
import os
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_PATH = "/home/user/myproject/lancedb"

RUN_ID = os.environ.get("ZEALT_RUN_ID", "")
TABLE_NAME = f"pdf_chunks_{RUN_ID}"

# Ensure the candidate's solution.py is importable.
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


@pytest.fixture(scope="module")
def db():
    import lancedb

    assert os.path.isdir(LANCEDB_PATH), (
        f"Expected LanceDB directory at {LANCEDB_PATH}; the candidate must persist "
        "the table there."
    )
    return lancedb.connect(LANCEDB_PATH)


@pytest.fixture(scope="module")
def table(db):
    names = db.table_names()
    assert TABLE_NAME in names, (
        f"Expected LanceDB table '{TABLE_NAME}' under {LANCEDB_PATH}, "
        f"but only found {names}. Make sure ingestion used the ZEALT_RUN_ID suffix."
    )
    return db.open_table(TABLE_NAME)


@pytest.fixture(scope="module")
def solution_module():
    sys.modules.pop("solution", None)
    solution_path = os.path.join(PROJECT_DIR, "solution.py")
    assert os.path.isfile(solution_path), (
        f"Expected candidate's solution module at {solution_path}."
    )
    module = importlib.import_module("solution")
    assert hasattr(module, "search") and callable(module.search), (
        "solution.py must expose a top-level callable named 'search(query, k)'."
    )
    return module


def _validate_result_shape(results, k):
    assert isinstance(results, list), f"Expected list, got {type(results).__name__}."
    assert len(results) == k, f"Expected {k} results, got {len(results)}."
    for i, r in enumerate(results):
        assert isinstance(r, dict), f"Result {i} is not a dict: {type(r).__name__}."
        for key in ("doc_id", "page", "snippet"):
            assert key in r, f"Result {i} is missing required key '{key}': {r!r}."
        assert isinstance(r["doc_id"], str), (
            f"Result {i} doc_id must be str, got {type(r['doc_id']).__name__}."
        )
        assert isinstance(r["page"], int) and not isinstance(r["page"], bool), (
            f"Result {i} page must be int, got {type(r['page']).__name__}."
        )
        assert isinstance(r["snippet"], str) and len(r["snippet"]) > 0, (
            f"Result {i} snippet must be a non-empty string."
        )


def test_run_id_present():
    assert RUN_ID, "ZEALT_RUN_ID must be set in the verifier environment."


def test_table_exists(db):
    names = db.table_names()
    assert TABLE_NAME in names, (
        f"Expected LanceDB table '{TABLE_NAME}' to exist under {LANCEDB_PATH}; "
        f"found tables: {names}."
    )


def test_schema_has_required_columns(table):
    schema = table.schema
    names = set(schema.names)
    required = {"doc_id", "page", "chunk_id", "text", "embedding"}
    missing = required - names
    assert not missing, (
        f"Table schema is missing required columns: {missing}. Found columns: {names}."
    )


def test_table_has_minimum_rows(table):
    n = table.count_rows()
    assert n >= 9, (
        f"Expected at least 9 chunks (3 PDFs * ~3+ pages), got {n}."
    )


def test_doc_ids_cover_corpus(table):
    df = table.to_pandas()
    docs = set(str(x) for x in df["doc_id"].unique().tolist())
    required = {"alpha", "bravo", "charlie"}
    missing = required - docs
    assert not missing, (
        f"Expected doc_id values {required} (from the corpus PDFs), "
        f"got {docs} (missing {missing})."
    )


def test_pages_are_positive_and_reasonable(table):
    df = table.to_pandas()
    pages = df["page"].tolist()
    assert all(int(p) > 0 for p in pages), (
        f"All page values must be positive integers; got min={min(pages)}."
    )
    for doc in ("alpha", "bravo", "charlie"):
        sub = df[df["doc_id"] == doc]
        assert len(sub) > 0, f"No rows found for doc_id={doc!r}."
        max_page = int(sub["page"].max())
        assert 3 <= max_page <= 12, (
            f"max page for doc_id={doc!r} is {max_page}; expected between 3 and 12 "
            "(PDFs have ~5 pages each)."
        )


def test_search_password_anchor(solution_module):
    query = "What is the secret password mentioned in the corpus?"
    results = solution_module.search(query, 5)
    _validate_result_shape(results, 5)
    top = results[0]
    assert top["doc_id"] == "bravo", (
        f"Expected top-1 doc_id 'bravo' for the password query, got {top['doc_id']!r}. "
        f"Full top-5: {[(r['doc_id'], r['page']) for r in results]}"
    )
    assert top["page"] == 3, (
        f"Expected top-1 page 3 for the password query, got {top['page']}. "
        f"Full top-5: {[(r['doc_id'], r['page']) for r in results]}"
    )
    assert "RAVENCLAW123" in top["snippet"], (
        f"Expected the secret password 'RAVENCLAW123' to appear in the top-1 snippet, "
        f"but it was not present in: {top['snippet']!r}"
    )


def test_search_butterflies_anchor(solution_module):
    query = "Which document discusses the cobalt blue migration of butterflies across the Sahara?"
    results = solution_module.search(query, 5)
    _validate_result_shape(results, 5)
    top = results[0]
    assert top["doc_id"] == "alpha", (
        f"Expected top-1 doc_id 'alpha' for the butterflies query, got {top['doc_id']!r}. "
        f"Full top-5: {[(r['doc_id'], r['page']) for r in results]}"
    )
    assert top["page"] == 2, (
        f"Expected top-1 page 2 for the butterflies query, got {top['page']}. "
        f"Full top-5: {[(r['doc_id'], r['page']) for r in results]}"
    )


def test_search_gyroscope_anchor(solution_module):
    query = "Where can I find the procedure to calibrate a quantum gyroscope using lithium niobate?"
    results = solution_module.search(query, 5)
    _validate_result_shape(results, 5)
    top = results[0]
    assert top["doc_id"] == "charlie", (
        f"Expected top-1 doc_id 'charlie' for the gyroscope query, got {top['doc_id']!r}. "
        f"Full top-5: {[(r['doc_id'], r['page']) for r in results]}"
    )
    assert top["page"] == 4, (
        f"Expected top-1 page 4 for the gyroscope query, got {top['page']}. "
        f"Full top-5: {[(r['doc_id'], r['page']) for r in results]}"
    )


def test_verifier_direct_openai_search(table):
    """End-to-end sanity check: verifier embeds the gyroscope query with OpenAI
    and runs a direct vector search against the candidate's table.

    Skipped when the candidate chose an embedding model whose dimensionality
    differs from text-embedding-3-small (1536). The candidate-side search()
    tests above already prove their pipeline works end-to-end.
    """
    from openai import OpenAI

    schema = table.schema
    embedding_field = schema.field("embedding")
    expected_dim = 1536
    field_type = embedding_field.type
    list_size = getattr(field_type, "list_size", None)
    if list_size is None or list_size != expected_dim:
        pytest.skip(
            f"Embedding column dim={list_size} != {expected_dim}; "
            "candidate chose a non-text-embedding-3-small model."
        )

    client = OpenAI()
    query = "Where can I find the procedure to calibrate a quantum gyroscope using lithium niobate?"
    resp = client.embeddings.create(model="text-embedding-3-small", input=query)
    qvec = resp.data[0].embedding
    results = table.search(qvec).limit(5).to_list()
    assert results, "Direct vector search returned no results."
    assert results[0]["doc_id"] == "charlie", (
        f"Direct OpenAI-driven search expected doc_id 'charlie', got {results[0]['doc_id']!r}."
    )
    assert int(results[0]["page"]) == 4, (
        f"Direct OpenAI-driven search expected page 4, got {results[0]['page']}."
    )
