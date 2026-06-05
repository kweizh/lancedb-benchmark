import json
import os
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
GT_PATH = "/app/ground_truth.json"


@pytest.fixture(scope="module")
def gt():
    assert os.path.isfile(GT_PATH), f"Ground-truth fixture missing at {GT_PATH}"
    with open(GT_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def solution():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    # Drop a possibly cached stale copy.
    sys.modules.pop("solution", None)
    import solution as sol  # type: ignore
    return sol


def test_solution_exports_callables(solution):
    assert hasattr(solution, "phrase_search"), "solution must expose phrase_search()"
    assert hasattr(solution, "boolean_must_search"), "solution must expose boolean_must_search()"
    assert callable(solution.phrase_search), "phrase_search must be callable"
    assert callable(solution.boolean_must_search), "boolean_must_search must be callable"


def test_phrase_strict_adjacency_slop0(solution, gt):
    res = solution.phrase_search(["machine", "learning"], slop=0, k=50)
    assert isinstance(res, list), f"phrase_search must return list, got {type(res)}"
    assert all(isinstance(x, int) for x in res), f"phrase_search must return list[int]: {res}"
    expected = set(gt["adjacent_ids"])
    assert set(res) == expected, (
        f"slop=0 must return only adjacent ground-truth ids. "
        f"Expected {sorted(expected)}, got {sorted(res)}"
    )
    assert len(res) <= 50


def test_phrase_within_slop3(solution, gt):
    res = solution.phrase_search(["machine", "learning"], slop=3, k=50)
    expected = set(gt["adjacent_ids"]) | set(gt["within3_ids"])
    assert set(res) == expected, (
        f"slop=3 must return adjacent + within-3 ids. "
        f"Expected {sorted(expected)}, got {sorted(res)}. "
        f"Missing: {sorted(expected - set(res))}, Extra: {sorted(set(res) - expected)}"
    )
    # Should not include any far id
    far_set = set(gt["far_ids"])
    assert set(res).isdisjoint(far_set), (
        f"slop=3 leaked far-gap ids: {sorted(set(res) & far_set)}"
    )


def test_phrase_k_truncates_to_first_k(solution):
    full = solution.phrase_search(["machine", "learning"], slop=3, k=50)
    five = solution.phrase_search(["machine", "learning"], slop=3, k=5)
    assert isinstance(five, list)
    assert len(five) == 5, f"k=5 must return exactly 5 results, got {len(five)}"
    # BM25 scores tie for many docs so the within-tie ordering returned by the
    # native engine is not guaranteed to be stable; we only assert that the
    # k=5 result is a subset of the k=50 result.
    assert set(five).issubset(set(full)), (
        f"k=5 results must be a subset of k=50 results. five={five} vs full={full}"
    )


def test_boolean_must_not_excludes_deep(solution, gt):
    res = solution.boolean_must_search(
        must_terms=["machine", "learning"], must_not_terms=["deep"], k=50
    )
    assert isinstance(res, list)
    assert all(isinstance(x, int) for x in res)
    deep_ids = set(gt["deep_ids"])
    assert deep_ids, "Fixture must declare at least one deep_id"
    leaked = set(res) & deep_ids
    assert not leaked, f"boolean MUST_NOT failed; deep-tagged ids leaked: {sorted(leaked)}"
    # Should still return all the non-deep docs (since all 50 docs contain machine AND learning).
    expected = set(range(50)) - deep_ids
    assert set(res) == expected, (
        f"boolean must include every doc containing machine AND learning and NOT deep. "
        f"Expected {sorted(expected)}, got {sorted(res)}"
    )
    assert len(res) >= 5, "Fixture should yield at least 5 matches"


def test_determinism(solution):
    # Native FTS may return tied-score docs in a non-stable order across calls,
    # so we only assert that the SET of ids is invariant across repeated calls.
    a1 = solution.phrase_search(["machine", "learning"], slop=0, k=50)
    a2 = solution.phrase_search(["machine", "learning"], slop=0, k=50)
    assert set(a1) == set(a2), "phrase_search result set must be deterministic"
    b1 = solution.boolean_must_search(["machine", "learning"], ["deep"], k=50)
    b2 = solution.boolean_must_search(["machine", "learning"], ["deep"], k=50)
    assert set(b1) == set(b2), "boolean_must_search result set must be deterministic"
