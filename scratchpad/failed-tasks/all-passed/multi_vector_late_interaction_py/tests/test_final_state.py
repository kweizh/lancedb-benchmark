import importlib
import os
import sys

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
FIXTURE_PATH = os.path.join(PROJECT_DIR, ".fixture.npz")


@pytest.fixture(scope="module")
def solution_module():
    assert os.path.isfile(SOLUTION_PATH), f"Missing {SOLUTION_PATH}"
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    return importlib.import_module("solution")


@pytest.fixture(scope="module")
def fixture_data():
    with np.load(FIXTURE_PATH) as data:
        return {
            "Q1": np.asarray(data["Q1"], dtype=np.float32),
            "Q2": np.asarray(data["Q2"], dtype=np.float32),
            "rigged_doc1": int(data["rigged_doc1"]),
            "rigged_doc2": int(data["rigged_doc2"]),
            "decoy_doc_id": int(data["decoy_doc_id"]),
        }


def _load_full_table_as_arrays():
    import lancedb

    table_name = "colbert_tokens"
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(table_name)
    df = tbl.to_pandas()
    # Build per-doc array of shape (4, 32)
    docs = {}
    for _, row in df.iterrows():
        emb = np.asarray(row["embedding"], dtype=np.float32).reshape(-1)
        assert emb.shape == (32,), f"Unexpected embedding shape: {emb.shape}"
        doc_id = int(row["doc_id"])
        token_idx = int(row["token_idx"])
        docs.setdefault(doc_id, {})[token_idx] = emb
    # Convert to ordered 4x32 arrays
    matrix = {}
    for doc_id, tokens in docs.items():
        idxs = sorted(tokens.keys())
        assert idxs == [0, 1, 2, 3], f"Doc {doc_id} has token indices {idxs}, expected [0,1,2,3]"
        matrix[doc_id] = np.stack([tokens[i] for i in idxs], axis=0)  # (4, 32)
    return matrix  # dict[doc_id -> (4,32)]


def _cosine_sim(a, b):
    """a: (...d), b: (...d) -> cosine similarity (last dim)."""
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return np.sum(an * bn, axis=-1)


def _reference_late_interaction(query: np.ndarray, doc_matrices: dict) -> list[tuple[int, float]]:
    """Compute late-interaction score for every doc against query (M,32). Returns sorted list (doc_id, score) desc."""
    M = query.shape[0]
    scored = []
    for doc_id, tokens in doc_matrices.items():
        # tokens: (4, 32), query: (M, 32)
        # sims[i, j] = cos(q_i, t_j)
        qn = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-12)  # (M,32)
        tn = tokens / (np.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-12)  # (4,32)
        sims = qn @ tn.T  # (M, 4)
        score = float(sims.max(axis=1).sum())
        scored.append((doc_id, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def _reference_sum_pool(query: np.ndarray, doc_matrices: dict) -> list[tuple[int, float]]:
    """Sum-pool baseline: sum query tokens to 1 vec, sum doc tokens to 1 vec, cosine."""
    q_sum = query.sum(axis=0)
    scored = []
    for doc_id, tokens in doc_matrices.items():
        d_sum = tokens.sum(axis=0)
        scored.append((doc_id, float(_cosine_sim(q_sum, d_sum))))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def test_solution_module_loads_and_callable(solution_module):
    assert callable(getattr(solution_module, "colbert_search", None)), (
        "solution.py must expose a callable named colbert_search."
    )


def test_colbert_search_top1_for_q1(solution_module, fixture_data):
    result = solution_module.colbert_search(fixture_data["Q1"], k=5)
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 5, f"Expected 5 results, got {len(result)}"
    assert all(isinstance(x, int) for x in result), (
        f"Expected list[int]; got types {[type(x).__name__ for x in result]}"
    )
    assert result[0] == fixture_data["rigged_doc1"], (
        f"Expected top-1 = rigged_doc1 ({fixture_data['rigged_doc1']}), got {result[0]} (full result: {result})"
    )


def test_colbert_search_top1_for_q2(solution_module, fixture_data):
    result = solution_module.colbert_search(fixture_data["Q2"], k=5)
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 5, f"Expected 5 results, got {len(result)}"
    assert result[0] == fixture_data["rigged_doc2"], (
        f"Expected top-1 = rigged_doc2 ({fixture_data['rigged_doc2']}), got {result[0]} (full result: {result})"
    )


def test_sum_pool_baseline_strictly_worse_than_late_interaction_on_q1(fixture_data):
    """Verifier independently computes sum-pool baseline directly on LanceDB; rigged decoy must win."""
    docs = _load_full_table_as_arrays()
    sum_pool_ranking = _reference_sum_pool(fixture_data["Q1"], docs)
    sum_pool_top1 = sum_pool_ranking[0][0]
    assert sum_pool_top1 != fixture_data["rigged_doc1"], (
        f"Sum-pool baseline must not beat late-interaction. "
        f"Got sum-pool top-1 = {sum_pool_top1}, late-interaction top-1 = {fixture_data['rigged_doc1']}. "
        f"Top-5 sum-pool ranking: {sum_pool_ranking[:5]}"
    )
    assert sum_pool_top1 == fixture_data["decoy_doc_id"], (
        f"Sum-pool baseline top-1 should be the rigged decoy ({fixture_data['decoy_doc_id']}); got {sum_pool_top1}. "
        f"Top-5 sum-pool ranking: {sum_pool_ranking[:5]}"
    )


def test_reference_late_interaction_matches_candidate(solution_module, fixture_data):
    """Verifier's own late-interaction reference (over the FULL table) must agree on top-1 for both queries."""
    docs = _load_full_table_as_arrays()
    ref_q1 = _reference_late_interaction(fixture_data["Q1"], docs)
    ref_q2 = _reference_late_interaction(fixture_data["Q2"], docs)
    assert ref_q1[0][0] == fixture_data["rigged_doc1"], (
        f"Reference late-interaction top-1 for Q1 should be {fixture_data['rigged_doc1']}; got {ref_q1[:3]}"
    )
    assert ref_q2[0][0] == fixture_data["rigged_doc2"], (
        f"Reference late-interaction top-1 for Q2 should be {fixture_data['rigged_doc2']}; got {ref_q2[:3]}"
    )
    cand_q1 = solution_module.colbert_search(fixture_data["Q1"], k=5)
    cand_q2 = solution_module.colbert_search(fixture_data["Q2"], k=5)
    assert cand_q1[0] == ref_q1[0][0], (
        f"Candidate top-1 Q1 = {cand_q1[0]} mismatches verifier reference top-1 {ref_q1[0][0]}"
    )
    assert cand_q2[0] == ref_q2[0][0], (
        f"Candidate top-1 Q2 = {cand_q2[0]} mismatches verifier reference top-1 {ref_q2[0][0]}"
    )


def test_colbert_search_respects_k_parameter(solution_module, fixture_data):
    r3 = solution_module.colbert_search(fixture_data["Q1"], k=3)
    r5 = solution_module.colbert_search(fixture_data["Q1"], k=5)
    assert len(r3) == 3, f"k=3 must return 3 results; got {len(r3)} ({r3})"
    assert len(r5) == 5, f"k=5 must return 5 results; got {len(r5)} ({r5})"
    assert r3[0] == fixture_data["rigged_doc1"], f"k=3 top-1 must equal rigged doc; got {r3[0]}"
    assert r5[0] == fixture_data["rigged_doc1"], f"k=5 top-1 must equal rigged doc; got {r5[0]}"
    assert len(set(r5)) == 5, f"Returned doc_ids must be distinct; got {r5}"
