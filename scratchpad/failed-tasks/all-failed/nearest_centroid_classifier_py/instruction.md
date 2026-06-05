# Nearest-Centroid Classifier with LanceDB

## Background
You are building a classical nearest-centroid classifier on top of LanceDB. The training data is already loaded into a LanceDB table inside the container, and you must use LanceDB itself (vector search and `where` filters) both to compute the per-class centroids and to classify new queries. There is no neural network and no external model service in this task — only the LanceDB API plus numpy.

## Environment
- A LanceDB database is initialized at `/app/lancedb_data`.
- One pre-seeded table holds the training data. Its base name is `train_data` and the parallel-run-safe full name is `train_data_${ZEALT_RUN_ID}` (you must read `ZEALT_RUN_ID` from the environment and append it).
- Schema of the training table:
  - `id` Int64
  - `label` Int32 — values are exactly 0..5 (six well-separated classes, 100 rows per class)
  - `vector` fixed_size_list<float32, 40>
- A 120-row labelled test fixture (20 rows per class) is provided as a JSON file at `/app/test_set.json`. It is a JSON list whose elements look like:
  ```json
  {"vector": [..40 floats..], "label": 0}
  ```

## Requirements
Implement `/home/user/myproject/solution.py` that exposes the following public API:

1. `build_centroids() -> None`
   - For each class `c` in 0..5, query the LanceDB training table for rows where `label == c`, retrieve their vectors, and compute the per-class centroid (component-wise mean over the 100 vectors of that class).
   - Write the 6 centroids into a NEW LanceDB table whose base name is `centroids` and parallel-run-safe full name is `centroids_${ZEALT_RUN_ID}`. The new table MUST use this schema:
     - `label` Int32
     - `vector` fixed_size_list<float32, 40>
   - The table MUST contain exactly 6 rows, one per class 0..5.
   - If the centroids table already exists it MUST be overwritten so that the function is idempotent across repeated calls.

2. `classify(query_vec) -> int`
   - Takes a length-40 vector (Python list or numpy array of floats).
   - Runs a LanceDB COSINE search against the `centroids_${ZEALT_RUN_ID}` table and returns the integer `label` of the nearest centroid.

3. `evaluate(test_set) -> float`
   - `test_set` is a list of dicts of shape `{"vector": list[float] (len 40), "label": int}`.
   - Classify each entry using `classify(...)` and return the fraction of correct predictions as a Python `float` in `[0.0, 1.0]`.

Provide a small driver script `/home/user/myproject/run.py` that:
- Calls `build_centroids()`.
- Loads `/app/test_set.json`.
- Prints exactly one line: `accuracy=<float>` (using the value returned by `evaluate`).

## Implementation Hints
- Read `ZEALT_RUN_ID` from the environment before constructing any table name.
- Open the existing training table via `db.open_table("train_data_" + run_id)`.
- For each label, you can pull the rows with `tbl.search().where("label = <c>").limit(<n>).to_pandas()` or via `tbl.to_pandas()` followed by an in-memory filter — either is fine as long as you use LanceDB to load the data.
- Use `db.create_table(name, data, mode="overwrite")` so that `build_centroids` is idempotent.
- Cosine search: `centroids_tbl.search(query_vec).distance_type("cosine").limit(1).to_pandas()` and read the `label` column from the single returned row.
- Vectors stored in LanceDB must be `float32` 40-d lists/arrays. Convert with `np.asarray(v, dtype=np.float32)` before inserting.
- The clusters in the training fixture are well separated, so a classifier with correctly computed centroids and a correct cosine search will easily exceed 0.90 accuracy. If you do not clear 0.90, something is wrong with how you are aggregating or storing the centroids.

## Acceptance Criteria
- Project path: `/home/user/myproject`
- Command: `python3 run.py`
  - Reads `ZEALT_RUN_ID` from the environment.
  - Calls `build_centroids()`, then `evaluate(test_set)` on the 120-row fixture at `/app/test_set.json`.
  - Prints a single line of the form `accuracy=<float>` to stdout (the float can be printed with any standard repr like `0.95` or `0.9583333333`).
- After running `python3 run.py`, the LanceDB database at `/app/lancedb_data` MUST contain a table named `centroids_${ZEALT_RUN_ID}` with exactly 6 rows and the schema `{label: Int32, vector: fixed_size_list<float32, 40>}`.
- Each centroid stored by the candidate MUST match the verifier's independent recomputation (mean of all training vectors per class) component-wise within `atol=1e-4`.
- The accuracy returned by `evaluate` and printed by `run.py` MUST be ≥ 0.90 on the provided test fixture.
- `classify(vec)` MUST be invokable from an independent Python process after `build_centroids()` has been called, and MUST return an integer in `{0,1,2,3,4,5}`.

