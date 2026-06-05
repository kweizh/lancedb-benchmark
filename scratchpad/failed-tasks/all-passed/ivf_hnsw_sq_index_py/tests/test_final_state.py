import importlib
import os
import subprocess
import sys

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"


def _seeds():
    return [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]


def _query_vec(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(128).astype("float32")


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine distance between query vector `a` (1D) and matrix `b` (2D)."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    b_unit = b / (b_norms + 1e-12)
    cos_sim = b_unit @ a_norm
    return 1.0 - cos_sim


@pytest.fixture(scope="module")
def solution_module():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    try:
        mod = importlib.import_module("solution")
    except Exception as exc:  # pragma: no cover - surface import errors
        pytest.fail(
            f"Failed to import candidate's solution.py from {PROJECT_DIR}: {exc!r}"
        )
    return mod


@pytest.fixture(scope="module")
def lancedb_table():
    import lancedb

    db_path = os.environ.get("LANCE_DB_PATH")
    assert db_path, "LANCE_DB_PATH must be set for verification."
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID must be set for verification."
    table_name = f"vectors_{run_id}"
    db = lancedb.connect(db_path)
    names = db.table_names()
    assert table_name in names, (
        f"Expected table '{table_name}' to exist in {db_path}; found {names}."
    )
    return db.open_table(table_name)


@pytest.fixture(scope="module")
def brute_force_vectors(lancedb_table):
    df = lancedb_table.to_pandas()
    assert "id" in df.columns and "vector" in df.columns, (
        f"Table must contain 'id' and 'vector' columns; got {list(df.columns)}."
    )
    df = df.sort_values("id").reset_index(drop=True)
    ids = df["id"].to_numpy().astype(np.int64)
    vecs = np.array([np.asarray(v, dtype=np.float32) for v in df["vector"]])
    assert vecs.shape == (1024, 128), (
        f"Expected 1024 x 128 vectors, got {vecs.shape}."
    )
    return ids, vecs


def test_solution_search_callable(solution_module):
    assert hasattr(solution_module, "search"), (
        "solution.py must expose a top-level callable `search(query_vec, k, nprobes)`."
    )
    assert callable(solution_module.search), (
        "solution.search must be a callable function."
    )


def test_index_exists_with_correct_type_and_column(lancedb_table):
    indices = lancedb_table.list_indices()
    matching = []
    for idx in indices:
        index_type = getattr(idx, "index_type", None) or (
            idx.get("index_type") if isinstance(idx, dict) else None
        )
        columns = getattr(idx, "columns", None) or (
            idx.get("columns") if isinstance(idx, dict) else None
        )
        if index_type is None or columns is None:
            continue
        normalized = str(index_type).upper().replace("_", "")
        if normalized == "IVFHNSWSQ" and list(columns) == ["vector"]:
            matching.append((idx, getattr(idx, "name", None)))
    assert len(matching) >= 1, (
        f"Expected at least one IVF_HNSW_SQ index on column 'vector'; "
        f"got indices: {indices!r}"
    )
    # Cross-check via index_stats(name).index_type which canonically returns 'IVF_HNSW_SQ'.
    _, name = matching[0]
    if name is not None:
        stats = lancedb_table.index_stats(name)
        stats_type = getattr(stats, "index_type", "")
        assert str(stats_type).upper().replace("_", "") == "IVFHNSWSQ", (
            f"index_stats({name!r}).index_type must canonically equal 'IVF_HNSW_SQ'; "
            f"got {stats_type!r}."
        )


def test_search_nprobes_1_truncates(solution_module):
    q = _query_vec(11)
    results = solution_module.search(q, 10, 1)
    assert isinstance(results, list), (
        f"search(...) must return a list; got {type(results).__name__}."
    )
    assert len(results) <= 10, (
        f"With nprobes=1, search must return at most 10 results; got {len(results)}."
    )


def test_search_nprobes_8_returns_full_k(solution_module):
    q = _query_vec(11)
    results = solution_module.search(q, 10, 8)
    assert isinstance(results, list), (
        f"search(...) must return a list; got {type(results).__name__}."
    )
    assert len(results) == 10, (
        f"With nprobes=8, search must return exactly 10 results; got {len(results)}."
    )
    for row in results:
        assert isinstance(row, dict), (
            f"Each search result must be a dict; got {type(row).__name__}."
        )
        assert "id" in row, f"Each result row must contain an 'id' field; got {row!r}."


def test_recall_at_10_nprobes_8(solution_module, brute_force_vectors):
    ids, vecs = brute_force_vectors
    recalls = []
    for seed in _seeds():
        q = _query_vec(seed)
        dists = _cosine_distance(q, vecs)
        gt_top10 = set(ids[np.argsort(dists)[:10]].tolist())
        cand_rows = solution_module.search(q, 10, 8)
        cand_ids = {int(r["id"]) for r in cand_rows}
        recalls.append(len(gt_top10 & cand_ids) / 10.0)
    avg_recall = sum(recalls) / len(recalls)
    assert avg_recall >= 0.90, (
        f"Expected average recall@10 with nprobes=8 to be >= 0.90; "
        f"got {avg_recall:.4f} (per-seed: {recalls})."
    )


def test_higher_nprobes_improves_recall(solution_module, brute_force_vectors):
    ids, vecs = brute_force_vectors
    recalls_low = []
    recalls_high = []
    for seed in _seeds():
        q = _query_vec(seed)
        dists = _cosine_distance(q, vecs)
        gt_top10 = set(ids[np.argsort(dists)[:10]].tolist())

        cand_low = {int(r["id"]) for r in solution_module.search(q, 10, 1)}
        cand_high = {int(r["id"]) for r in solution_module.search(q, 10, 8)}

        recalls_low.append(len(gt_top10 & cand_low) / 10.0)
        recalls_high.append(len(gt_top10 & cand_high) / 10.0)

    avg_low = sum(recalls_low) / len(recalls_low)
    avg_high = sum(recalls_high) / len(recalls_high)
    assert avg_high > avg_low, (
        f"Expected average recall@10 with nprobes=8 ({avg_high:.4f}) to be strictly "
        f"greater than with nprobes=1 ({avg_low:.4f})."
    )


def test_cli_smoke():
    # NOTE: lancedb 0.25.3 occasionally raises SIGABRT during Python interpreter
    # shutdown after a write-heavy session (the candidate's import may seed the
    # table). We tolerate any non-zero exit as long as the program produced the
    # expected stdout line before tearing down.
    result = subprocess.run(
        [
            "python3",
            "-c",
            "import solution; print(len(solution.search([0.0]*128, 10, 8)))",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
    )
    assert "10" in result.stdout.split(), (
        f"Expected stdout to contain '10' on its own line; "
        f"got stdout={result.stdout!r} stderr={result.stderr!r} "
        f"returncode={result.returncode}."
    )
