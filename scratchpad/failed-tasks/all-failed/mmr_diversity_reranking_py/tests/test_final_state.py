import importlib.util
import os
import sys

import numpy as np
import pytest


PROJECT_DIR = "/home/user/myproject"
DB_DIR = "/app/db"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")


def _table_name():
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    assert run_id, "ZEALT_RUN_ID must be set in the verifier environment."
    return f"mmr_docs_{run_id}"


def _open_table():
    import lancedb

    db = lancedb.connect(DB_DIR)
    name = _table_name()
    list_fn = getattr(db, "table_names", None) or db.list_tables
    names = list(list_fn())
    assert name in names, (
        f"LanceDB table {name!r} not found in {DB_DIR}. Found tables: {names}."
    )
    return db.open_table(name)


def _load_solution_module():
    assert os.path.isfile(SOLUTION_PATH), (
        f"Candidate solution module not found at {SOLUTION_PATH}."
    )
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    spec = importlib.util.spec_from_file_location("solution", SOLUTION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = module
    spec.loader.exec_module(module)
    return module


def _load_rows():
    """Return (ids np.ndarray int64, cluster_ids np.ndarray int64, vectors np.ndarray float64 [N,32])."""
    table = _open_table()
    df = table.to_pandas()
    assert len(df) == 120, f"Expected 120 rows in {_table_name()!r}, got {len(df)}."
    assert set(df.columns) >= {"id", "cluster_id", "vector"}, (
        f"Table must expose columns id, cluster_id, vector; got {list(df.columns)}."
    )
    ids = np.asarray(df["id"].tolist(), dtype=np.int64)
    cluster_ids = np.asarray(df["cluster_id"].tolist(), dtype=np.int64)
    vectors = np.stack([np.asarray(v, dtype=np.float64) for v in df["vector"].tolist()], axis=0)
    assert vectors.shape == (120, 32), (
        f"Vector column must be 32-d for all 120 rows; got shape {vectors.shape}."
    )
    return ids, cluster_ids, vectors


def _cosine_sim_matrix(query, mat):
    """query: (32,), mat: (N,32). Returns (N,) cosine similarities in float64."""
    q = np.asarray(query, dtype=np.float64)
    qn = np.linalg.norm(q)
    assert qn > 0, "Query vector must be non-zero."
    mn = np.linalg.norm(mat, axis=1)
    safe = np.where(mn > 0, mn, 1.0)
    return (mat @ q) / (safe * qn)


def _build_query(ids, cluster_ids, vectors):
    """Return the deterministic verifier query: unit-normalised sum of centroids of clusters 0..4."""
    mix = np.zeros(32, dtype=np.float64)
    for c in range(5):
        mask = cluster_ids == c
        assert mask.sum() == 12, (
            f"Cluster {c} must have exactly 12 rows; got {int(mask.sum())}."
        )
        mix += vectors[mask].mean(axis=0)
    norm = np.linalg.norm(mix)
    assert norm > 0, "Verifier query mixture is unexpectedly zero."
    return mix / norm


# -------- Fixture / schema checks --------


def test_table_exists_with_120_rows():
    table = _open_table()
    assert table.count_rows() == 120, (
        f"Expected 120 rows in the fixture table, got {table.count_rows()}."
    )


def test_table_schema_id_cluster_and_fixed_size_vector():
    import pyarrow as pa

    table = _open_table()
    schema = table.schema
    names = schema.names
    for required in ("id", "cluster_id", "vector"):
        assert required in names, (
            f"Schema must contain column {required!r}; got fields {names}."
        )
    assert pa.types.is_integer(schema.field("id").type), (
        f"id column must be an integer type; got {schema.field('id').type}."
    )
    assert pa.types.is_integer(schema.field("cluster_id").type), (
        f"cluster_id column must be an integer type; got {schema.field('cluster_id').type}."
    )
    vector_type = schema.field("vector").type
    assert pa.types.is_fixed_size_list(vector_type), (
        f"vector column must be a fixed_size_list type; got {vector_type}."
    )
    assert vector_type.list_size == 32, (
        f"vector column must have list size 32; got {vector_type.list_size}."
    )
    assert pa.types.is_floating(vector_type.value_type), (
        f"vector column inner type must be floating-point; got {vector_type.value_type}."
    )


def test_ids_and_cluster_counts():
    ids, cluster_ids, _ = _load_rows()
    assert sorted(ids.tolist()) == list(range(120)), (
        f"id column must cover 0..119 exactly once; got first 5={sorted(ids.tolist())[:5]}."
    )
    unique, counts = np.unique(cluster_ids, return_counts=True)
    assert unique.tolist() == list(range(10)), (
        f"cluster_id must take values 0..9; got {unique.tolist()}."
    )
    assert counts.tolist() == [12] * 10, (
        f"Each cluster_id must have exactly 12 rows; got counts {counts.tolist()}."
    )


# -------- Behavioural MMR checks --------


def _call_candidate(query, k, lambda_):
    solution = _load_solution_module()
    assert hasattr(solution, "mmr_search"), "solution.py must expose a top-level `mmr_search` function."
    result = solution.mmr_search(query.tolist(), k=k, lambda_=lambda_)
    assert isinstance(result, list), (
        f"mmr_search(..., lambda_={lambda_}) must return a list; got {type(result).__name__}."
    )
    assert len(result) == k, (
        f"mmr_search(..., k={k}, lambda_={lambda_}) must return exactly {k} ids; got {len(result)}."
    )
    coerced = [int(x) for x in result]
    assert len(set(coerced)) == k, (
        f"mmr_search(..., lambda_={lambda_}) must return {k} distinct ids; got {coerced}."
    )
    for value in coerced:
        assert 0 <= value < 120, (
            f"mmr_search(..., lambda_={lambda_}) returned id {value} outside 0..119."
        )
    return coerced


def test_lambda_one_matches_pure_vector_search():
    ids, cluster_ids, vectors = _load_rows()
    query = _build_query(ids, cluster_ids, vectors)
    sims = _cosine_sim_matrix(query, vectors)
    # Stable order: highest sim first; ties broken by id ascending. Candidates must produce the same
    # order because LanceDB's cosine search is deterministic on identical inputs and ties are
    # extremely unlikely with 32-d float32 random data.
    pure_order = np.argsort(-sims, kind="stable")
    pure_top10 = [int(ids[i]) for i in pure_order[:10]]

    result = _call_candidate(query, k=10, lambda_=1.0)
    assert result == pure_top10, (
        "mmr_search with lambda_=1.0 must collapse to pure cosine top-10. "
        f"Expected {pure_top10}, got {result}."
    )


def test_lambda_low_drives_high_diversity():
    ids, cluster_ids, vectors = _load_rows()
    query = _build_query(ids, cluster_ids, vectors)
    result = _call_candidate(query, k=10, lambda_=0.3)
    cluster_by_id = {int(i): int(c) for i, c in zip(ids, cluster_ids)}
    distinct_clusters = {cluster_by_id[i] for i in result}
    assert len(distinct_clusters) >= 7, (
        "mmr_search with lambda_=0.3 must span at least 7 distinct clusters (diversity dominates). "
        f"Got clusters {sorted(distinct_clusters)} from ids {result}."
    )


def test_lambda_balanced_preserves_top1_and_some_diversity():
    ids, cluster_ids, vectors = _load_rows()
    query = _build_query(ids, cluster_ids, vectors)
    sims = _cosine_sim_matrix(query, vectors)
    pure_top1 = int(ids[int(np.argmax(sims))])

    result = _call_candidate(query, k=10, lambda_=0.7)
    cluster_by_id = {int(i): int(c) for i, c in zip(ids, cluster_ids)}
    distinct_clusters = {cluster_by_id[i] for i in result}
    assert len(distinct_clusters) >= 5, (
        "mmr_search with lambda_=0.7 must still span at least 5 distinct clusters. "
        f"Got clusters {sorted(distinct_clusters)} from ids {result}."
    )
    assert pure_top1 in result, (
        f"mmr_search with lambda_=0.7 must include the pure-search top-1 id {pure_top1}; "
        f"got {result}."
    )
