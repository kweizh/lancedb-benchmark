import json
import os
import subprocess
import sys
import time

import pytest

PROJECT_DIR = "/home/user/myproject"
HEADLINES_PATH = os.path.join(PROJECT_DIR, "headlines.json")
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
JINA_URL = "https://api.jina.ai/v1/embeddings"

ANCHORS = [
    {
        "query": "central bank interest rate hike",
        "expected_id": 1,
        "expected_topic": "finance",
        "expected_headline": "Federal Reserve Raises Benchmark Interest Rate by 25 Basis Points",
    },
    {
        "query": "Olympics gold medal swimming",
        "expected_id": 11,
        "expected_topic": "sports",
        "expected_headline": "American Swimmer Wins Olympic Gold Medal in 200m Freestyle",
    },
    {
        "query": "spacecraft Mars landing",
        "expected_id": 21,
        "expected_topic": "space",
        "expected_headline": "NASA Spacecraft Successfully Lands on Surface of Mars",
    },
]


def _run_id() -> str:
    val = os.environ.get("ZEALT_RUN_ID")
    assert val, "ZEALT_RUN_ID env var must be set."
    return val


def _table_name() -> str:
    return f"headlines_{_run_id()}"


def _jina_embed(texts, task):
    """Independent verifier-side call to the Jina API. Returns list of float vectors."""
    import httpx

    api_key = os.environ.get("JINA_API_KEY")
    assert api_key, "JINA_API_KEY env var must be set for verifier."
    payload = {"model": "jina-embeddings-v3", "task": task, "input": list(texts)}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_exc = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(JINA_URL, json=payload, headers=headers)
            assert resp.status_code == 200, (
                f"Jina API returned status {resp.status_code}: {resp.text[:500]}"
            )
            data = resp.json()
            assert "data" in data, f"Jina response missing 'data' field: {data}"
            ordered = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in ordered]
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(2)
    raise AssertionError(f"Jina API call failed after retries: {last_exc}")


@pytest.fixture(scope="session", autouse=True)
def _ensure_project_path_on_syspath():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)


@pytest.fixture(scope="session")
def candidate_solution():
    """Import the candidate solution module and ensure the index has been built."""
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    import importlib

    solution = importlib.import_module("solution")
    assert hasattr(solution, "build_index"), "solution.build_index must be defined."
    assert hasattr(solution, "search"), "solution.search must be defined."

    # Always (re)build to guarantee a fresh, deterministic state.
    solution.build_index()
    return solution


def test_lancedb_data_directory_exists(candidate_solution):
    assert os.path.isdir(LANCEDB_DIR), (
        f"Expected LanceDB data dir {LANCEDB_DIR} to exist after build_index()."
    )


def test_lancedb_table_exists_and_row_count(candidate_solution):
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    names = db.table_names()
    expected = _table_name()
    assert expected in names, (
        f"Expected LanceDB table '{expected}' in {LANCEDB_DIR}, got {names}."
    )
    tbl = db.open_table(expected)
    assert tbl.count_rows() == 50, (
        f"Expected 50 rows in table {expected}, got {tbl.count_rows()}."
    )


def test_lancedb_embedding_dim_matches_jina(candidate_solution):
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(_table_name())
    schema = tbl.schema

    # Find the embedding-like column (a fixed-size list of floats).
    vec_field = None
    for field in schema:
        ftype = field.type
        if pa.types.is_fixed_size_list(ftype):
            value_type = ftype.value_type
            if pa.types.is_floating(value_type):
                vec_field = field
                break
    assert vec_field is not None, (
        f"Could not locate a fixed_size_list<float, N> embedding column in schema: {schema}"
    )
    stored_dim = vec_field.type.list_size

    probe_vec = _jina_embed(["probe"], task="retrieval.query")[0]
    assert len(probe_vec) == stored_dim, (
        f"Embedding dim mismatch: LanceDB table has dim {stored_dim} but Jina returned dim {len(probe_vec)}."
    )


@pytest.mark.parametrize("anchor", ANCHORS, ids=[a["expected_topic"] for a in ANCHORS])
def test_anchor_query_rank1(candidate_solution, anchor):
    results = candidate_solution.search(anchor["query"], k=5)
    assert isinstance(results, list), f"search() must return a list, got {type(results)}."
    assert len(results) == 5, f"search(k=5) must return 5 results, got {len(results)}."
    for entry in results:
        assert isinstance(entry, dict), f"Each result must be a dict, got {type(entry)}."
        for key in ("id", "headline", "topic"):
            assert key in entry, f"Result entry missing key '{key}': {entry}"

    top = results[0]
    assert int(top["id"]) == anchor["expected_id"], (
        f"For query {anchor['query']!r}, expected rank-1 id={anchor['expected_id']} "
        f"(headline {anchor['expected_headline']!r}), got id={top['id']} headline={top.get('headline')!r}."
    )
    assert top["headline"] == anchor["expected_headline"], (
        f"For query {anchor['query']!r}, rank-1 headline mismatch: got {top['headline']!r}."
    )
    assert top["topic"] == anchor["expected_topic"], (
        f"For query {anchor['query']!r}, rank-1 topic mismatch: got {top['topic']!r}."
    )


def test_search_k_truncation(candidate_solution):
    results = candidate_solution.search("spacecraft Mars landing", k=3)
    assert isinstance(results, list)
    assert len(results) == 3, f"search(k=3) must return 3 results, got {len(results)}."


def test_cli_run_py(candidate_solution):
    cmd = [sys.executable, "run.py", "spacecraft Mars landing", "--k", "5"]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ},
    )
    assert proc.returncode == 0, (
        f"CLI run.py failed: rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # The stdout must contain a JSON array; tolerate extra logs before/after by locating the array.
    out = proc.stdout.strip()
    # Try direct parse first, then fall back to extracting the last JSON array.
    try:
        parsed = json.loads(out)
    except Exception:
        # Find the last '[' ... ']' that parses as JSON.
        last_open = out.rfind("[")
        last_close = out.rfind("]")
        assert last_open != -1 and last_close != -1 and last_close > last_open, (
            f"Could not locate JSON array in CLI stdout: {out!r}"
        )
        parsed = json.loads(out[last_open : last_close + 1])

    assert isinstance(parsed, list), f"CLI stdout must be a JSON array, got {type(parsed)}."
    assert len(parsed) == 5, f"CLI must print 5 results for k=5, got {len(parsed)}."
    assert int(parsed[0]["id"]) == 21, (
        f"CLI rank-1 must be id=21 for the Mars query, got id={parsed[0].get('id')}."
    )


def test_asymmetric_task_parameter_agrees(candidate_solution):
    """Independently embed the finance query and confirm LanceDB rank-1 matches candidate rank-1."""
    import lancedb

    query = "central bank interest rate hike"
    qvec = _jina_embed([query], task="retrieval.query")[0]
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(_table_name())
    rows = tbl.search(qvec).limit(1).to_list()
    assert rows, "Verifier-side vector search returned no rows."
    verifier_rank1_id = int(rows[0]["id"])
    candidate_rank1_id = int(candidate_solution.search(query, k=1)[0]["id"])
    assert verifier_rank1_id == candidate_rank1_id == 1, (
        f"Verifier rank-1 id={verifier_rank1_id}, candidate rank-1 id={candidate_rank1_id}, "
        f"expected both to be 1 (Federal Reserve headline)."
    )
