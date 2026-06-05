import importlib
import os
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_PATH = "/home/user/myproject/lancedb"

RUN_ID = os.environ.get("ZEALT_RUN_ID", "")
TABLE_NAME = f"docs_sections_{RUN_ID}"

EXPECTED_REPO_PATHS = {
    "auth-guide.md",
    "api-reference.md",
    "deployment.md",
    "performance.md",
    "monitoring.md",
    "migrations.md",
}

# Ensure the candidate's indexer.py is importable.
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
        f"but only found {names}. Make sure indexing used the ZEALT_RUN_ID suffix."
    )
    return db.open_table(TABLE_NAME)


@pytest.fixture(scope="module")
def indexer_module():
    sys.modules.pop("indexer", None)
    indexer_path = os.path.join(PROJECT_DIR, "indexer.py")
    assert os.path.isfile(indexer_path), (
        f"Expected candidate's indexer module at {indexer_path}."
    )
    module = importlib.import_module("indexer")
    assert hasattr(module, "search") and callable(module.search), (
        "indexer.py must expose a top-level callable named 'search(query, k)'."
    )
    return module


def _validate_result_shape(results, k):
    assert isinstance(results, list), f"Expected list, got {type(results).__name__}."
    assert len(results) == k, f"Expected {k} results, got {len(results)}."
    for i, r in enumerate(results):
        assert isinstance(r, dict), f"Result {i} is not a dict: {type(r).__name__}."
        for key in ("repo_path", "doc_title", "section_title", "score"):
            assert key in r, f"Result {i} is missing required key '{key}': {r!r}."
        assert isinstance(r["repo_path"], str) and r["repo_path"], (
            f"Result {i} repo_path must be a non-empty str, got {type(r['repo_path']).__name__}."
        )
        assert isinstance(r["doc_title"], str) and r["doc_title"], (
            f"Result {i} doc_title must be a non-empty str, got {type(r['doc_title']).__name__}."
        )
        assert isinstance(r["section_title"], str) and r["section_title"], (
            f"Result {i} section_title must be a non-empty str, got {type(r['section_title']).__name__}."
        )
        assert isinstance(r["score"], (int, float)) and not isinstance(r["score"], bool), (
            f"Result {i} score must be a numeric (float-compatible), got {type(r['score']).__name__}."
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
    required = {"repo_path", "doc_title", "section_title", "content", "embedding"}
    missing = required - names
    assert not missing, (
        f"Table schema is missing required columns: {missing}. Found columns: {names}."
    )


def test_table_has_minimum_rows(table):
    n = table.count_rows()
    assert n >= 18, (
        f"Expected at least 18 sections (6 files * >=3 sections), got {n}."
    )


def test_repo_paths_cover_corpus(table):
    df = table.to_pandas()
    paths = set(str(x) for x in df["repo_path"].unique().tolist())
    missing = EXPECTED_REPO_PATHS - paths
    assert not missing, (
        f"Expected repo_path values {EXPECTED_REPO_PATHS}, "
        f"got {paths} (missing {missing})."
    )


def test_each_doc_has_at_least_three_sections(table):
    df = table.to_pandas()
    for repo_path in EXPECTED_REPO_PATHS:
        sub = df[df["repo_path"] == repo_path]
        assert len(sub) > 0, f"No rows found for repo_path={repo_path!r}."
        section_titles = set(str(s) for s in sub["section_title"].unique().tolist())
        assert len(section_titles) >= 3, (
            f"Expected at least 3 distinct section_title rows for {repo_path!r}, "
            f"got {len(section_titles)}: {section_titles}."
        )


def test_search_sso_anchor(indexer_module):
    query = "How do I configure single sign-on with Okta SAML for our application?"
    results = indexer_module.search(query, 3)
    _validate_result_shape(results, 3)
    top = results[0]
    assert top["repo_path"] == "auth-guide.md", (
        f"Expected top-1 repo_path 'auth-guide.md' for the SSO query, got {top['repo_path']!r}. "
        f"Full top-3: {[(r['repo_path'], r['section_title']) for r in results]}"
    )
    assert top["section_title"].strip() == "SSO Setup", (
        f"Expected top-1 section_title 'SSO Setup' for the SSO query, got {top['section_title']!r}. "
        f"Full top-3: {[(r['repo_path'], r['section_title']) for r in results]}"
    )


def test_search_rate_limit_anchor(indexer_module):
    query = "What HTTP status does the API return when a client exceeds the rate limit, and how should the client back off?"
    results = indexer_module.search(query, 3)
    _validate_result_shape(results, 3)
    top = results[0]
    assert top["repo_path"] == "api-reference.md", (
        f"Expected top-1 repo_path 'api-reference.md' for the rate-limit query, got {top['repo_path']!r}. "
        f"Full top-3: {[(r['repo_path'], r['section_title']) for r in results]}"
    )
    assert top["section_title"].strip() == "Rate Limiting", (
        f"Expected top-1 section_title 'Rate Limiting' for the rate-limit query, got {top['section_title']!r}. "
        f"Full top-3: {[(r['repo_path'], r['section_title']) for r in results]}"
    )


def test_search_rollback_anchor(indexer_module):
    query = "What is the procedure to roll back a failed database migration that left the schema in an inconsistent state?"
    results = indexer_module.search(query, 3)
    _validate_result_shape(results, 3)
    top = results[0]
    assert top["repo_path"] == "migrations.md", (
        f"Expected top-1 repo_path 'migrations.md' for the rollback query, got {top['repo_path']!r}. "
        f"Full top-3: {[(r['repo_path'], r['section_title']) for r in results]}"
    )
    assert top["section_title"].strip() == "Rolling Back Migrations", (
        f"Expected top-1 section_title 'Rolling Back Migrations' for the rollback query, got {top['section_title']!r}. "
        f"Full top-3: {[(r['repo_path'], r['section_title']) for r in results]}"
    )


def test_verifier_direct_openai_search(table):
    """End-to-end sanity check: verifier embeds the rollback query with OpenAI
    and runs a direct vector search against the candidate's table.

    Skipped automatically if the candidate chose an embedding model whose
    dimensionality differs from text-embedding-3-small (1536).
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
    query = "What is the procedure to roll back a failed database migration that left the schema in an inconsistent state?"
    resp = client.embeddings.create(model="text-embedding-3-small", input=query)
    qvec = resp.data[0].embedding
    results = table.search(qvec).limit(5).to_list()
    assert results, "Direct vector search returned no results."
    assert results[0]["repo_path"] == "migrations.md", (
        f"Direct OpenAI-driven search expected repo_path 'migrations.md', "
        f"got {results[0]['repo_path']!r}."
    )
    assert str(results[0]["section_title"]).strip() == "Rolling Back Migrations", (
        f"Direct OpenAI-driven search expected section_title 'Rolling Back Migrations', "
        f"got {results[0]['section_title']!r}."
    )
