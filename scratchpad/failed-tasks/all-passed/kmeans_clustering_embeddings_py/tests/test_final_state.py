import importlib
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")


@pytest.fixture(scope="session", autouse=True)
def run_candidate_pipeline():
    """Execute the candidate's run.py once before any verification runs."""
    result = subprocess.run(
        ["python3", "run.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"`python3 run.py` exited with status {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    yield


@pytest.fixture(scope="session")
def lancedb_conn():
    import lancedb

    return lancedb.connect(LANCEDB_DIR)


@pytest.fixture(scope="session")
def embeddings_matrix_and_ids(lancedb_conn):
    tbl = lancedb_conn.open_table("embeddings")
    df = tbl.to_pandas()
    df = df.sort_values("id").reset_index(drop=True)
    ids = df["id"].to_numpy().astype(np.int64)
    X = np.vstack(df["vector"].values).astype(np.float32)
    return X, ids


@pytest.fixture(scope="session")
def fitted_kmeans(embeddings_matrix_and_ids):
    from sklearn.cluster import KMeans

    X, _ = embeddings_matrix_and_ids
    km = KMeans(n_clusters=8, random_state=2026, n_init=10)
    km.fit(X)
    return km


@pytest.fixture(scope="session")
def solution_module():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    # Drop any cached version so a fresh import is forced after run.py.
    if "solution" in sys.modules:
        del sys.modules["solution"]
    mod = importlib.import_module("solution")
    return mod


# ------------------------- clusters table schema & content -------------------------


def test_clusters_table_exists(lancedb_conn):
    assert "clusters" in lancedb_conn.table_names(), (
        "Candidate must create a LanceDB table named 'clusters'."
    )


def test_clusters_table_row_count(lancedb_conn):
    tbl = lancedb_conn.open_table("clusters")
    assert tbl.count_rows() == 800, (
        f"'clusters' table must have 800 rows; got {tbl.count_rows()}."
    )


def test_clusters_table_schema(lancedb_conn):
    tbl = lancedb_conn.open_table("clusters")
    schema = tbl.schema
    field_names = [f.name for f in schema]
    assert set(field_names) == {"id", "cluster_id"}, (
        f"'clusters' schema must have exactly fields 'id' and 'cluster_id', got {field_names}."
    )
    id_field = schema.field("id")
    cid_field = schema.field("cluster_id")
    assert pa.types.is_int64(id_field.type), (
        f"'clusters.id' must be Int64, got {id_field.type}."
    )
    assert pa.types.is_int32(cid_field.type), (
        f"'clusters.cluster_id' must be Int32, got {cid_field.type}."
    )


def test_clusters_ids_match_embeddings(lancedb_conn, embeddings_matrix_and_ids):
    _, ids = embeddings_matrix_and_ids
    df = lancedb_conn.open_table("clusters").to_pandas()
    assert set(df["id"].astype(np.int64).tolist()) == set(ids.tolist()), (
        "'clusters.id' must equal the set of ids in 'embeddings'."
    )


def test_clusters_distinct_cluster_ids(lancedb_conn):
    df = lancedb_conn.open_table("clusters").to_pandas()
    distinct = set(int(x) for x in df["cluster_id"].tolist())
    assert distinct == {0, 1, 2, 3, 4, 5, 6, 7}, (
        f"'clusters.cluster_id' must contain exactly {{0..7}}; got {distinct}."
    )


def test_clusters_balance(lancedb_conn):
    df = lancedb_conn.open_table("clusters").to_pandas()
    counts = df.groupby("cluster_id").size().to_dict()
    for cid in range(8):
        n = int(counts.get(cid, 0))
        assert 80 <= n <= 120, (
            f"Cluster {cid} must have between 80 and 120 members; got {n}."
        )


# ------------------------- centroids table schema & content -------------------------


def test_centroids_table_exists(lancedb_conn):
    assert "centroids" in lancedb_conn.table_names(), (
        "Candidate must create a LanceDB table named 'centroids'."
    )


def test_centroids_table_row_count(lancedb_conn):
    tbl = lancedb_conn.open_table("centroids")
    assert tbl.count_rows() == 8, (
        f"'centroids' must have 8 rows; got {tbl.count_rows()}."
    )


def test_centroids_table_schema(lancedb_conn):
    tbl = lancedb_conn.open_table("centroids")
    schema = tbl.schema
    field_names = {f.name for f in schema}
    assert "cluster_id" in field_names, "'centroids' must have a 'cluster_id' column."
    assert "vector" in field_names, "'centroids' must have a 'vector' column."
    assert pa.types.is_int32(schema.field("cluster_id").type), (
        f"'centroids.cluster_id' must be Int32; got {schema.field('cluster_id').type}."
    )
    vec_type = schema.field("vector").type
    assert pa.types.is_fixed_size_list(vec_type), (
        f"'centroids.vector' must be a fixed-size list; got {vec_type}."
    )
    assert vec_type.list_size == 32, (
        f"'centroids.vector' must have list_size=32; got {vec_type.list_size}."
    )
    assert pa.types.is_floating(vec_type.value_type), (
        f"'centroids.vector' element type must be float; got {vec_type.value_type}."
    )


def test_centroids_cluster_ids(lancedb_conn):
    df = lancedb_conn.open_table("centroids").to_pandas()
    assert set(int(x) for x in df["cluster_id"].tolist()) == set(range(8)), (
        "'centroids.cluster_id' values must be {0..7}."
    )


# ------------------------- solution module API -------------------------


def test_solution_module_exposes_callables(solution_module):
    assert callable(getattr(solution_module, "cluster_centroids", None)), (
        "solution.cluster_centroids must be callable."
    )
    assert callable(getattr(solution_module, "nearest_cluster", None)), (
        "solution.nearest_cluster must be callable."
    )


def test_cluster_centroids_shape_and_values(solution_module, fitted_kmeans):
    centroids = solution_module.cluster_centroids()
    assert isinstance(centroids, np.ndarray), (
        "cluster_centroids() must return a numpy ndarray."
    )
    assert centroids.shape == (8, 32), (
        f"cluster_centroids() must return shape (8, 32); got {centroids.shape}."
    )
    assert centroids.dtype == np.float32, (
        f"cluster_centroids() must return float32 array; got {centroids.dtype}."
    )
    expected = fitted_kmeans.cluster_centers_.astype(np.float32)
    assert np.allclose(centroids, expected, atol=1e-4), (
        "cluster_centroids() must equal KMeans.cluster_centers_ row-by-row "
        "(row i = centroid for cluster_id i)."
    )


def test_adjusted_rand_index(lancedb_conn, embeddings_matrix_and_ids):
    from sklearn.metrics import adjusted_rand_score

    _, ids = embeddings_matrix_and_ids
    gt = np.load(os.path.join(LANCEDB_DIR, "ground_truth.npy"))
    assert gt.shape == (800,), f"ground_truth shape mismatch: {gt.shape}"

    clusters_df = lancedb_conn.open_table("clusters").to_pandas()
    embeddings_df = lancedb_conn.open_table("embeddings").to_pandas()
    # Align both by id then compare in the order of `ids` (sorted).
    clusters_df = clusters_df.set_index("id").reindex(ids).reset_index()
    embeddings_df = embeddings_df.set_index("id").reindex(ids).reset_index()
    # Map embeddings_df row order -> ground truth order assumes seed wrote ground_truth
    # in id-sorted order. The seed script guarantees this.
    predicted = clusters_df["cluster_id"].to_numpy().astype(np.int64)
    score = adjusted_rand_score(gt, predicted)
    assert score >= 0.90, f"Adjusted Rand Index must be >= 0.90; got {score:.4f}."


def test_nearest_cluster_matches_kmeans(solution_module, fitted_kmeans):
    rng = np.random.default_rng(7)
    queries = rng.standard_normal((5, 32)).astype(np.float32)
    for i, q in enumerate(queries):
        expected = int(fitted_kmeans.predict(q.reshape(1, -1))[0])
        actual = solution_module.nearest_cluster(q)
        assert isinstance(actual, int), (
            f"nearest_cluster must return a Python int; got {type(actual)} for query {i}."
        )
        assert actual == expected, (
            f"nearest_cluster(query {i}) returned {actual}; KMeans predicts {expected}."
        )


# ------------------------- idempotency -------------------------


def test_rerun_pipeline_is_idempotent(lancedb_conn):
    """Re-running run.py must not duplicate rows or corrupt schemas."""
    result = subprocess.run(
        ["python3", "run.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Second run of `python3 run.py` failed: {result.stderr}"
    )
    # Re-open via fresh connection — fixture-cached conn may have stale handles.
    import lancedb

    db2 = lancedb.connect(LANCEDB_DIR)
    assert db2.open_table("clusters").count_rows() == 800, (
        "After rerun, 'clusters' must still have exactly 800 rows."
    )
    assert db2.open_table("centroids").count_rows() == 8, (
        "After rerun, 'centroids' must still have exactly 8 rows."
    )
