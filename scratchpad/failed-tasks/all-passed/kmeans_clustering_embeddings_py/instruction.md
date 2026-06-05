# K-means Clustering of LanceDB Embeddings

## Background
You are given a pre-seeded LanceDB table named `embeddings` that holds 800 rows of 32-dimensional float vectors. Your task is to cluster these embeddings with K-means, persist the resulting labels back into a NEW LanceDB table, and expose a small Python API that lets callers query the centroids and look up the nearest cluster for an arbitrary vector. The clustering must be fully deterministic so the verifier can reproduce results.

## Requirements
- Load all rows of the `embeddings` table into memory as a pandas DataFrame.
- Fit a `sklearn.cluster.KMeans` model with exactly 8 clusters, `random_state=2026`, and `n_init=10`.
- Persist the cluster assignment for every row of `embeddings` into a NEW LanceDB table called `clusters` containing two columns only: `id` (Int64, matching the `id` column of `embeddings`) and `cluster_id` (Int32, in the range 0..7).
- Persist the 8 centroids into a NEW LanceDB table called `centroids` containing two columns: `cluster_id` (Int32) and `vector` (fixed-size list of float32, dimension 32). The centroids in this table must equal `kmeans.cluster_centers_` in the order produced by sklearn (row i = centroid for cluster_id i).
- Expose a Python module that the verifier can import and call.

## Implementation Hints
- Use `table.to_pandas()` to materialize the embeddings table; `pandas==2.2.3` is pre-installed.
- The vector column comes back as a numpy array per row; stack them with `numpy.vstack` before fitting.
- For the `nearest_cluster` API, run a cosine vector search on the `centroids` table and return the `cluster_id` of the top-1 result.
- All three tables (`embeddings`, `clusters`, `centroids`) live in the same LanceDB directory `/home/user/myproject/lancedb_data`.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 run.py`
  - When invoked with no arguments, `run.py` must read `embeddings` from `/home/user/myproject/lancedb_data`, fit the K-means model, and (over)write the `clusters` and `centroids` LanceDB tables. It must be idempotent (safe to re-run).
- After `run.py` exits successfully, a module named `solution` in `/home/user/myproject/solution.py` must be importable from `/home/user/myproject` and expose the following top-level callables:
  - `cluster_centroids() -> numpy.ndarray` returning an array of shape `(8, 32)` and dtype `float32` containing the centroids in cluster-id order (row i = cluster i). The values must equal `kmeans.cluster_centers_` to within `atol=1e-4`.
  - `nearest_cluster(query_vec) -> int` accepting a length-32 1-D vector (Python list or numpy array) and returning the cluster id (0..7) of the nearest centroid, computed via a LanceDB cosine vector search against the `centroids` table.
- LanceDB tables after `run.py` runs:
  - `clusters`: exactly 800 rows; schema `{id: int64, cluster_id: int32}`; the set of `id` values equals the set of `id` values in `embeddings`.
  - `centroids`: exactly 8 rows; schema includes `cluster_id: int32` and a 32-d float32 vector column.
- Cluster quality (verifier asserts against the ground-truth labels baked into the fixture):
  - The 8 distinct cluster ids `{0..7}` must all be present in the `clusters` table.
  - sklearn Adjusted Rand Index (`sklearn.metrics.adjusted_rand_score`) between the predicted labels and the ground-truth labels must be >= 0.90.
  - Cluster balance: each of the 8 predicted clusters must contain between 80 and 120 rows (inclusive).
  - For 5 deterministic query vectors, `nearest_cluster(query)` must equal `KMeans(n_clusters=8, random_state=2026, n_init=10).fit(X).predict([query])[0]` where `X` is the verifier-reloaded embeddings matrix.

