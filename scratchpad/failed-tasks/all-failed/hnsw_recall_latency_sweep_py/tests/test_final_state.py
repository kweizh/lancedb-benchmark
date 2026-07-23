import math
import os
import sys

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"

K = 10
DIM = 64
NUM_QUERIES = 30
QUERY_SEED = 12345
PARAM_GRID = {"nprobes": [1, 4, 16, 64], "refine_factor": [1, 10, 50]}
RECALL_MATCH_TOL = 1e-6
MONO_TOL = 1e-6
HIGH_EFFORT = (64, 50)
LOW_EFFORT = (1, 1)
HIGH_RECALL_THRESHOLD = 0.9
LOW_EFFORT_MARGIN = 0.05

EXPECTED_KEYS = {"nprobes", "refine_factor", "recall", "mean_latency_ms"}


def _uri():
    return os.environ.get("LANCEDB_URI", "/app/lancedb")


def _table_name():
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    assert run_id, "ZEALT_RUN_ID environment variable is not set."
    return f"vectors_{run_id}"


@pytest.fixture(scope="session")
def query_set():
    rng = np.random.default_rng(QUERY_SEED)
    return rng.standard_normal((NUM_QUERIES, DIM)).astype("float32")


@pytest.fixture(scope="session")
def solution_module():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    import solution  # type: ignore

    assert hasattr(solution, "build_index"), "solution.build_index is missing."
    assert hasattr(solution, "sweep"), "solution.sweep is missing."
    # Build the ANN index (idempotent per the contract).
    solution.build_index()
    return solution


@pytest.fixture(scope="session")
def sweep_results(solution_module, query_set):
    return solution_module.sweep(query_set, PARAM_GRID, k=K)


@pytest.fixture(scope="session")
def ground_truth(query_set):
    """Exact top-K neighbor id sets per query, computed by brute-force L2."""
    import lancedb

    tbl = lancedb.connect(_uri()).open_table(_table_name())
    arrow = tbl.to_arrow()
    ids = np.asarray(arrow["id"].to_pylist(), dtype=np.int64)
    vecs = np.asarray(arrow["vector"].to_pylist(), dtype=np.float32)

    gt = []
    for q in query_set:
        dist = np.sum((vecs - q) ** 2, axis=1)
        # tie-break by ascending id: lexsort with id as secondary key
        order = np.lexsort((ids, dist))
        gt.append(set(int(x) for x in ids[order][:K]))
    return gt


def _reference_recall(nprobes, refine_factor, query_set, gt):
    """Reproduce the ANN search on the same built index and compute recall@K."""
    import lancedb

    tbl = lancedb.connect(_uri()).open_table(_table_name())
    total = 0.0
    for q, gt_ids in zip(query_set, gt):
        rows = (
            tbl.search(q)
            .nprobes(nprobes)
            .refine_factor(refine_factor)
            .limit(K)
            .to_list()
        )
        ann_ids = set(int(r["id"]) for r in rows[:K])
        total += len(ann_ids & gt_ids) / float(K)
    return total / len(query_set)


def _by_config(results):
    return {(int(r["nprobes"]), int(r["refine_factor"])): r for r in results}


def test_return_shape_and_schema(sweep_results):
    assert isinstance(sweep_results, list), "sweep must return a list."
    expected_configs = [
        (n, r) for n in PARAM_GRID["nprobes"] for r in PARAM_GRID["refine_factor"]
    ]
    assert len(sweep_results) == len(expected_configs), (
        f"sweep must return {len(expected_configs)} configurations, "
        f"got {len(sweep_results)}."
    )

    for entry in sweep_results:
        assert isinstance(entry, dict), "Each sweep entry must be a dict."
        assert set(entry.keys()) == EXPECTED_KEYS, (
            f"Each entry must have exactly keys {EXPECTED_KEYS}, got {set(entry.keys())}."
        )
        assert isinstance(entry["nprobes"], int), "nprobes must be an int."
        assert isinstance(entry["refine_factor"], int), "refine_factor must be an int."
        assert isinstance(entry["recall"], float), "recall must be a float."
        assert isinstance(entry["mean_latency_ms"], float), (
            "mean_latency_ms must be a float."
        )
        assert 0.0 <= entry["recall"] <= 1.0, (
            f"recall must be in [0,1], got {entry['recall']}."
        )
        assert math.isfinite(entry["mean_latency_ms"]) and entry["mean_latency_ms"] > 0, (
            f"mean_latency_ms must be finite and > 0, got {entry['mean_latency_ms']}."
        )

    # Cartesian product coverage (each config exactly once) and sorted ordering.
    seen = [(int(e["nprobes"]), int(e["refine_factor"])) for e in sweep_results]
    assert sorted(set(seen)) == sorted(expected_configs), (
        "sweep must cover each (nprobes, refine_factor) pair exactly once."
    )
    assert len(seen) == len(set(seen)), "Duplicate configurations in results."
    assert seen == sorted(seen), (
        "Results must be sorted ascending by (nprobes, refine_factor)."
    )


def test_ground_truth_recall_correctness(sweep_results, query_set, ground_truth):
    by_cfg = _by_config(sweep_results)
    for (nprobes, refine_factor), entry in by_cfg.items():
        expected = _reference_recall(nprobes, refine_factor, query_set, ground_truth)
        assert abs(entry["recall"] - expected) <= RECALL_MATCH_TOL, (
            f"Reported recall for (nprobes={nprobes}, refine_factor={refine_factor}) "
            f"is {entry['recall']}, but independent brute-force computation gives "
            f"{expected}."
        )


def test_monotonic_in_nprobes(sweep_results):
    by_cfg = _by_config(sweep_results)
    for refine in PARAM_GRID["refine_factor"]:
        recalls = [by_cfg[(n, refine)]["recall"] for n in PARAM_GRID["nprobes"]]
        for prev, cur in zip(recalls, recalls[1:]):
            assert cur >= prev - MONO_TOL, (
                f"Recall must be non-decreasing in nprobes (refine_factor={refine}): "
                f"{recalls}."
            )


def test_monotonic_in_refine_factor(sweep_results):
    by_cfg = _by_config(sweep_results)
    for nprobes in PARAM_GRID["nprobes"]:
        recalls = [by_cfg[(nprobes, r)]["recall"] for r in PARAM_GRID["refine_factor"]]
        for prev, cur in zip(recalls, recalls[1:]):
            assert cur >= prev - MONO_TOL, (
                f"Recall must be non-decreasing in refine_factor (nprobes={nprobes}): "
                f"{recalls}."
            )


def test_high_effort_recall_threshold(sweep_results):
    by_cfg = _by_config(sweep_results)
    high = by_cfg[HIGH_EFFORT]["recall"]
    assert high >= HIGH_RECALL_THRESHOLD, (
        f"Highest-effort config {HIGH_EFFORT} must reach recall >= "
        f"{HIGH_RECALL_THRESHOLD}, got {high}."
    )


def test_low_effort_strictly_worse(sweep_results):
    by_cfg = _by_config(sweep_results)
    low = by_cfg[LOW_EFFORT]["recall"]
    high = by_cfg[HIGH_EFFORT]["recall"]
    assert low < high - LOW_EFFORT_MARGIN, (
        f"Lowest-effort config {LOW_EFFORT} recall ({low}) must be strictly lower than "
        f"highest-effort config {HIGH_EFFORT} recall ({high}) by at least "
        f"{LOW_EFFORT_MARGIN}."
    )


def test_determinism(solution_module, sweep_results, query_set):
    second = solution_module.sweep(query_set, PARAM_GRID, k=K)
    first_by = _by_config(sweep_results)
    second_by = _by_config(second)
    assert set(first_by.keys()) == set(second_by.keys()), (
        "Repeated sweep returned a different set of configurations."
    )
    for cfg, entry in first_by.items():
        assert abs(entry["recall"] - second_by[cfg]["recall"]) <= 1e-9, (
            f"Recall for config {cfg} is not deterministic across calls: "
            f"{entry['recall']} vs {second_by[cfg]['recall']}."
        )
