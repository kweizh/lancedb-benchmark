import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
QUERY_PATH = os.path.join(PROJECT_DIR, "query.json")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "output.json")
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")


def _import_solution():
    spec = importlib.util.spec_from_file_location("solution", SOLUTION_PATH)
    assert spec is not None and spec.loader is not None, (
        f"Could not load module spec from {SOLUTION_PATH}"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def query_fixture():
    assert os.path.isfile(QUERY_PATH), f"{QUERY_PATH} must exist."
    with open(QUERY_PATH) as f:
        data = json.load(f)
    assert data["relevant_ids"] == [50, 51, 52, 53, 54], (
        "query.json's relevant_ids must be [50, 51, 52, 53, 54]."
    )
    return data


@pytest.fixture(scope="module")
def run_id():
    rid = os.environ.get("ZEALT_RUN_ID", "")
    assert rid, "ZEALT_RUN_ID must be set in the verifier environment."
    return rid


def test_run_py_executes_and_writes_output(query_fixture):
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    result = subprocess.run(
        ["python3", "run.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"`python3 run.py` failed (exit={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert os.path.isfile(OUTPUT_PATH), (
        f"{OUTPUT_PATH} must be created by run.py."
    )
    with open(OUTPUT_PATH) as f:
        out = json.load(f)
    assert isinstance(out, list) and len(out) == 10, (
        f"output.json must be a JSON list of 10 ints, got: {out!r}"
    )
    assert all(isinstance(x, int) for x in out), (
        f"output.json entries must all be ints, got: {out!r}"
    )


def test_rocchio_search_signature_and_top1(query_fixture):
    mod = _import_solution()
    assert hasattr(mod, "rocchio_search"), (
        "solution.py must expose a callable 'rocchio_search'."
    )
    q0 = query_fixture["q0"]
    relevant_ids = query_fixture["relevant_ids"]
    result = mod.rocchio_search(q0, relevant_ids)
    assert isinstance(result, list), f"rocchio_search must return a list, got {type(result)}."
    assert len(result) == 10, f"rocchio_search default k=10 must return 10 ids, got {len(result)}."
    assert all(isinstance(x, int) for x in result), (
        f"rocchio_search must return list[int]; got {result!r}"
    )
    assert result[0] == 55, (
        f"After Rocchio expansion, top-1 must be doc 55 (cluster-0 centroid). Got: {result}"
    )
    cluster0_hits = sum(1 for x in result[0:5] if 0 <= x < 100)
    assert cluster0_hits >= 4, (
        f"At least 4 of top-5 must be cluster 0 (ids 0..99). Got: {result[0:5]}"
    )


def test_rocchio_search_is_deterministic(query_fixture):
    mod = _import_solution()
    q0 = query_fixture["q0"]
    relevant_ids = query_fixture["relevant_ids"]
    a = mod.rocchio_search(q0, relevant_ids)
    b = mod.rocchio_search(q0, relevant_ids)
    assert a == b, (
        f"rocchio_search must be deterministic on identical inputs. Got {a!r} vs {b!r}"
    )


def test_top1_is_not_initial_top1(query_fixture, run_id):
    """Independent verifier-side computation: confirm the candidate's top-1
    matches the Rocchio-expanded query top-1, NOT the initial top-1."""
    import lancedb

    mod = _import_solution()
    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(f"documents_{run_id}")

    q0 = np.array(query_fixture["q0"], dtype=np.float32)
    relevant_ids = query_fixture["relevant_ids"]

    # Initial top-10 (n_rel + n_nrel)
    initial = tbl.search(q0).distance_type("cosine").limit(10).to_list()
    initial_ids = [int(r["id"]) for r in initial]
    initial_top1 = initial_ids[0]

    # Fetch rel/nrel vectors directly from the table
    rel_set = set(relevant_ids)
    nrel_ids = [i for i in initial_ids if i not in rel_set]
    assert len(nrel_ids) >= 1, "Need at least one non-relevant id."

    def fetch_vectors(ids):
        rows = tbl.search().where(f"id IN ({','.join(str(i) for i in ids)})").limit(len(ids)).to_list()
        # Sort to match the requested order is unnecessary; we average.
        vecs = np.array([r["vector"] for r in rows], dtype=np.float32)
        return vecs

    rel_vecs = fetch_vectors(relevant_ids)
    nrel_vecs = fetch_vectors(nrel_ids)
    alpha, beta, gamma = 1.0, 0.75, 0.15
    q_prime = alpha * q0 + beta * rel_vecs.mean(axis=0) - gamma * nrel_vecs.mean(axis=0)
    q_prime = q_prime / (np.linalg.norm(q_prime) + 1e-12)

    expanded = tbl.search(q_prime).distance_type("cosine").limit(10).to_list()
    expanded_ids = [int(r["id"]) for r in expanded]

    candidate = mod.rocchio_search(query_fixture["q0"], relevant_ids)

    assert candidate[0] == expanded_ids[0], (
        "Candidate's top-1 must match the verifier-recomputed Rocchio top-1. "
        f"Candidate={candidate}, verifier_expanded={expanded_ids}, initial_top1={initial_top1}"
    )
    assert candidate[0] != initial_top1, (
        "Candidate's top-1 must differ from the initial (un-expanded) top-1, "
        "confirming the Rocchio update was actually applied. "
        f"Candidate top-1 = initial top-1 = {initial_top1}"
    )


def test_table_schema_and_norms(run_id):
    import lancedb

    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(f"documents_{run_id}")
    assert tbl.count_rows() == 400, (
        f"Table must contain 400 rows, found {tbl.count_rows()}"
    )
    schema = tbl.schema
    field_names = {f.name for f in schema}
    for required in ("id", "cluster", "vector"):
        assert required in field_names, (
            f"Required column '{required}' missing from table schema. Found: {field_names}"
        )

    df = tbl.to_pandas()
    vecs = np.stack(df["vector"].to_numpy()).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-3), (
        "All stored vectors must be approximately L2-normalized (norm ~ 1.0)."
    )

    # Doc 55 lies (approximately) at the cluster-0 centroid (one of the
    # build-time invariants of the rigged fixture).
    row_55 = df[df["id"] == 55].iloc[0]
    assert int(row_55["cluster"]) == 0, (
        f"Doc 55 must belong to cluster 0 in the fixture, got {row_55['cluster']}"
    )
