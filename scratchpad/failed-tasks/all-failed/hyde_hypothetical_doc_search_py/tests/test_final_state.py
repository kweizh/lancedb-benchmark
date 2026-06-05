import importlib
import importlib.util
import inspect
import json
import os
import sys

import lancedb
import numpy as np
import pytest
from openai import OpenAI

PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
LANCEDB_DIR = "/app/lancedb_data"
EXPECTED_FIXTURE = "/app/.expected.json"
QUERY = "GC differences between languages?"


def _load_solution():
    """Import /home/user/myproject/solution.py as a module."""
    assert os.path.isfile(SOLUTION_PATH), (
        f"Candidate solution file is missing at {SOLUTION_PATH}."
    )
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    spec = importlib.util.spec_from_file_location("solution", SOLUTION_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def solution_mod():
    return _load_solution()


@pytest.fixture(scope="session")
def expected():
    with open(EXPECTED_FIXTURE) as f:
        return json.load(f)


def test_required_callables_exposed(solution_mod):
    for name in ("generate_hypothetical", "hyde_search", "baseline_search"):
        assert hasattr(solution_mod, name), (
            f"solution.py must define a callable named '{name}'."
        )
        fn = getattr(solution_mod, name)
        assert callable(fn), f"solution.py attribute '{name}' must be callable."

    sig_hyde = inspect.signature(solution_mod.hyde_search)
    sig_base = inspect.signature(solution_mod.baseline_search)
    assert "query" in sig_hyde.parameters or len(sig_hyde.parameters) >= 1, (
        "hyde_search must accept a 'query' parameter as its first argument."
    )
    assert "query" in sig_base.parameters or len(sig_base.parameters) >= 1, (
        "baseline_search must accept a 'query' parameter as its first argument."
    )


def test_hyde_search_returns_rigged_top1(solution_mod, expected):
    rigged_id = expected["rigged_id"]
    result = solution_mod.hyde_search(QUERY, k=5)
    assert isinstance(result, list), (
        f"hyde_search must return a list, got {type(result).__name__}."
    )
    assert len(result) == 5, f"hyde_search must return exactly 5 ids, got {len(result)}."
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in result), (
        f"hyde_search must return Python ints; got types {[type(x).__name__ for x in result]}."
    )
    assert len(set(result)) == 5, f"hyde_search results must be unique; got {result}."
    assert result[0] == rigged_id, (
        f"hyde_search top-1 must be the rigged doc id={rigged_id} for the query "
        f"{QUERY!r}; got {result}."
    )


def test_baseline_search_misses_rigged(solution_mod, expected):
    rigged_id = expected["rigged_id"]
    result = solution_mod.baseline_search(QUERY, k=5)
    assert isinstance(result, list), (
        f"baseline_search must return a list, got {type(result).__name__}."
    )
    assert len(result) == 5, (
        f"baseline_search must return exactly 5 ids, got {len(result)}."
    )
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in result), (
        f"baseline_search must return Python ints; got types {[type(x).__name__ for x in result]}."
    )
    assert len(set(result)) == 5, f"baseline_search results must be unique; got {result}."
    assert result[0] != rigged_id, (
        f"baseline_search top-1 must NOT be the rigged doc id={rigged_id} (the whole "
        f"point of HyDE is to recover docs the baseline misses); got {result}."
    )


def test_generate_hypothetical_returns_long_string(solution_mod):
    text = solution_mod.generate_hypothetical(QUERY)
    assert isinstance(text, str), (
        f"generate_hypothetical must return a str, got {type(text).__name__}."
    )
    assert len(text) >= 100, (
        f"generate_hypothetical must return at least a 100-character paragraph; "
        f"got {len(text)} chars."
    )


def test_verifier_direct_baseline_matches(solution_mod):
    """Independent verifier-side cross-check: embed the raw query with text-embedding-3-small
    and search LanceDB directly. The candidate's baseline_search top-1 must match this."""
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=[QUERY])
    qvec = np.asarray(resp.data[0].embedding, dtype=np.float32)

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table("programming_qa")
    rows = tbl.search(qvec).limit(5).to_list()
    verifier_top1 = int(rows[0]["id"])

    candidate = solution_mod.baseline_search(QUERY, k=5)
    assert candidate[0] == verifier_top1, (
        f"Candidate baseline_search top-1 ({candidate[0]}) disagrees with the "
        f"verifier's direct text-embedding-3-small + LanceDB cosine top-1 "
        f"({verifier_top1}); baseline_search must embed the raw query directly."
    )


def test_hyde_search_stable_across_calls(solution_mod, expected):
    """temperature=0 and a corpus with one dominant GC-essay doc must make HyDE rank-1 stable."""
    rigged_id = expected["rigged_id"]
    r1 = solution_mod.hyde_search(QUERY, k=5)
    r2 = solution_mod.hyde_search(QUERY, k=5)
    assert r1[0] == rigged_id and r2[0] == rigged_id, (
        f"hyde_search rank-1 must be the rigged doc id={rigged_id} on consecutive calls; "
        f"got {r1[0]} then {r2[0]}."
    )
