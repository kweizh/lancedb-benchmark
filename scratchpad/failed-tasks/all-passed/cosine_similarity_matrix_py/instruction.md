# Cosine Similarity Matrix from LanceDB Self-Search

## Background
You are profiling the cluster structure of a small embedding dataset that lives inside a LanceDB table. Instead of computing pairwise cosine similarities directly with `numpy`, you must derive the full (N, N) similarity matrix purely from the results of LanceDB cosine searches. This validates that LanceDB's cosine ranking faithfully reflects the geometry of the stored vectors.

A pre-seeded LanceDB table is provisioned by the environment entrypoint at container start. The table has exactly 200 rows organized as 5 well-separated clusters of 40 rows each, with a 48-dimensional `vector` column. The table location and name are exposed through environment variables.

## Requirements
- Implement `/home/user/myproject/solution.py` exposing the following module-level functions:
  - `similarity_matrix() -> numpy.ndarray`: returns a `(200, 200)` float matrix `S` such that `S[i, j]` is the cosine similarity between the stored vectors of the rows with `id == i` and `id == j`.
  - `intra_class_mean(label: int) -> float`: returns the mean off-diagonal cosine similarity restricted to the rows whose `label` column equals `label`.
- The similarity values **MUST** be derived from LanceDB cosine search results. For every row `i`, issue a LanceDB query of the form `table.search(stored_vector_i).distance_type("cosine").limit(200)` (or equivalent through the search API) and use `similarity = 1 - distance` to populate row `i` of the matrix.
- The matrix indexing **MUST** be by the row's `id` column (the table is seeded so that ids are exactly `0..199`).

## Implementation Hints
- Connect to the LanceDB directory and open the seeded table using the environment variables `LANCEDB_URI` and `LANCEDB_TABLE` set by the entrypoint.
- For each row, read its stored vector back from the table and run a single cosine self-search with `limit` equal to the table size; map the returned `_distance` to similarity with `1 - distance`.
- Diagonal entries should be exactly `1.0` because each row is its own nearest neighbour under cosine distance.
- `intra_class_mean` should average the entries of the submatrix indexed by the ids of the given label, excluding the diagonal.
- `similarity_matrix()` may be called many times by the verifier; cache the result if you want to avoid re-querying LanceDB.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 -c "import solution, numpy as np; S=solution.similarity_matrix(); print(S.shape, S.dtype)"`
- The module `solution` exposes `similarity_matrix()` returning a `numpy.ndarray` with shape `(200, 200)`.
- The module `solution` exposes `intra_class_mean(label)` accepting an integer label in `[0, 4]` and returning a `float`.
- The returned matrix must satisfy:
  - `np.allclose(S, S.T, atol=1e-3)` (symmetric within tolerance).
  - All diagonal entries equal `1.0` within `atol=1e-3`.
  - All entries lie in `[-1.0, 1.0]` within `atol=1e-3`.
- For at least 4 of the 5 labels, `intra_class_mean(label) > global_off_diagonal_mean(S)`, demonstrating that LanceDB's cosine ranking preserves the cluster structure of the seeded data.
- The solution must NOT compute cosine similarity directly with `numpy` on the raw vector matrix; the values must come from LanceDB cosine search results (the verifier inspects the call into LanceDB by replacing the seeded vectors with a permutation and re-running the candidate, expecting the matrix to follow the permutation).

