import json
import os
import subprocess

import pytest

PROJECT_DIR = "/home/user/myproject"
RUN_ID = os.environ.get("ZEALT_RUN_ID", "")
TABLE_NAME = f"products_{RUN_ID}"
DATA_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
SOLUTION_PY = os.path.join(PROJECT_DIR, "solution.py")
RUN_SEARCH_PY = os.path.join(PROJECT_DIR, "run_search.py")


def _run_cli(query: str, k: int) -> list:
    """Invoke the candidate's run_search.py and parse stdout as JSON."""
    result = subprocess.run(
        ["python3", "run_search.py", "--query", query, "--k", str(k)],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"run_search.py exited with code {result.returncode}.\n"
        f"stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as e:
        raise AssertionError(
            f"run_search.py stdout must be valid JSON.\nstdout=\n{result.stdout}\nerr={e}"
        )


def test_candidate_files_exist():
    assert os.path.isfile(SOLUTION_PY), f"Missing {SOLUTION_PY}"
    assert os.path.isfile(RUN_SEARCH_PY), f"Missing {RUN_SEARCH_PY}"


def test_lancedb_table_exists_with_correct_shape():
    import lancedb

    db = lancedb.connect(DATA_DIR)
    names = db.table_names()
    assert TABLE_NAME in names, (
        f"Expected LanceDB table '{TABLE_NAME}' under {DATA_DIR}; found tables: {names}"
    )
    tbl = db.open_table(TABLE_NAME)
    assert len(tbl) == 60, f"Expected 60 rows in {TABLE_NAME}, got {len(tbl)}"

    schema = tbl.schema
    field_names = [f.name for f in schema]
    # Determine which column holds the vector.
    vec_col = None
    for fname in ("vector", "embedding", "embeddings"):
        if fname in field_names:
            vec_col = fname
            break
    if vec_col is None:
        # Fall back to whichever column has a fixed-size list type.
        import pyarrow as pa

        for f in schema:
            if pa.types.is_fixed_size_list(f.type):
                vec_col = f.name
                break
    assert vec_col is not None, (
        f"Could not locate a fixed-size-list vector column. Schema: {schema}"
    )
    import pyarrow as pa

    field = schema.field(vec_col)
    assert pa.types.is_fixed_size_list(field.type), (
        f"Vector column '{vec_col}' is not a fixed-size list. Type: {field.type}"
    )
    assert field.type.list_size == 1024, (
        f"Vector dimension must be 1024 (voyage-3). Got {field.type.list_size}."
    )


@pytest.mark.parametrize(
    "query,expected_id,k",
    [
        (
            "noise cancelling over-ear bluetooth headphones for travel",
            "prod-electronics-headphones-anc",
            5,
        ),
        (
            "high pressure espresso machine for home baristas",
            "prod-kitchen-espresso-machine",
            5,
        ),
        (
            "lightweight running shoes with carbon plate for marathon racing",
            "prod-fitness-carbon-marathon-shoes",
            3,
        ),
        (
            "bestselling fantasy novel about a young wizard and dragons",
            "prod-books-dragon-wizard-fantasy",
            3,
        ),
    ],
)
def test_cli_top1_matches_anchor(query, expected_id, k):
    results = _run_cli(query, k)
    assert isinstance(results, list) and len(results) >= 1, (
        f"Expected non-empty JSON array, got: {results!r}"
    )
    first = results[0]
    assert isinstance(first, dict), f"First result must be an object: {first!r}"
    assert "id" in first and "description" in first, (
        f"Result entries must contain 'id' and 'description'. Got keys: {list(first)}"
    )
    assert first["id"] == expected_id, (
        f"For query {query!r} expected rank-1 id={expected_id!r}, got {first['id']!r}."
    )
    assert len(results) <= k, f"CLI returned {len(results)} results, must be <= k={k}."


def test_solution_search_matches_cli_top1():
    import importlib.util

    spec = importlib.util.spec_from_file_location("candidate_solution", SOLUTION_PY)
    module = importlib.util.module_from_spec(spec)
    sys_path_backup = list(__import__("sys").path)
    try:
        __import__("sys").path.insert(0, PROJECT_DIR)
        spec.loader.exec_module(module)
    finally:
        __import__("sys").path[:] = sys_path_backup

    assert hasattr(module, "search"), "solution.py must define a callable `search`."
    query = "noise cancelling over-ear bluetooth headphones for travel"
    results = module.search(query, 5)
    assert isinstance(results, list) and len(results) >= 1, (
        f"solution.search must return a non-empty list, got: {results!r}"
    )
    first = results[0]
    assert isinstance(first, dict) and "id" in first, (
        f"solution.search results must be dicts with 'id', got: {first!r}"
    )
    assert first["id"] == "prod-electronics-headphones-anc", (
        f"solution.search top-1 should be 'prod-electronics-headphones-anc', got {first['id']!r}"
    )


def test_verifier_direct_voyage_search_matches_anchor():
    """Independent end-to-end check: the verifier embeds the query with voyageai
    using `input_type='query'`, runs a direct LanceDB vector search against the
    candidate's table, and confirms the rigged anchor wins. This proves the
    table holds genuine Voyage embeddings rather than random or precomputed data.
    """
    import lancedb
    import voyageai

    api_key = os.environ.get("VOYAGE_API_KEY")
    assert api_key, "VOYAGE_API_KEY must be set in the verifier env."

    client = voyageai.Client(api_key=api_key)
    query = "noise cancelling over-ear bluetooth headphones for travel"
    resp = client.embed([query], model="voyage-3", input_type="query")
    qvec = resp.embeddings[0]
    assert len(qvec) == 1024, f"voyage-3 query embedding dim must be 1024, got {len(qvec)}"

    db = lancedb.connect(DATA_DIR)
    tbl = db.open_table(TABLE_NAME)
    rows = tbl.search(qvec).limit(5).to_list()
    assert rows, "Direct LanceDB vector search returned no rows."
    assert rows[0]["id"] == "prod-electronics-headphones-anc", (
        f"Verifier direct search expected rank-1 'prod-electronics-headphones-anc', "
        f"got {rows[0]['id']!r}. Table may not contain real Voyage embeddings."
    )
