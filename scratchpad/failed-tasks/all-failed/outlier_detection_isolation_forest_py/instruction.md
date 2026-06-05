# Outlier Detection on a LanceDB Embeddings Table with Isolation Forest

## Background
You are given a pre-seeded LanceDB table called `events_${run-id}` that stores 1000 rows of low-dimensional behaviour embeddings. Most rows describe normal activity, but a small fraction were generated from a clearly different distribution and are considered outliers. Your job is to read those vectors out of LanceDB, fit a classical anomaly detector with `sklearn.ensemble.IsolationForest`, persist the per-row decision back into the same LanceDB table as a new boolean column, and expose a small Python API that the verifier can call to enumerate the most anomalous rows.

The LanceDB database lives at `/home/user/myproject/lancedb_data/`. The table name is `events_${run-id}` where `run-id` is the value of the `ZEALT_RUN_ID` environment variable. The table schema is `{id: int64, ts: int64, vector: fixed_size_list<float32, 20>}`. The vector contains 950 "normal" points drawn from `N(0, 0.5)` and 50 "outlier" points drawn from `N(5, 0.5)` (numpy seed 2026, contamination 0.05). The fixture is created automatically at container start; you do not need to seed it yourself.

## Requirements
- Implement a Python module `solution.py` at `/home/user/myproject/solution.py` that:
  1. Connects to the LanceDB database at `/home/user/myproject/lancedb_data/` and opens the `events_${run-id}` table (the table name MUST be read from the `ZEALT_RUN_ID` environment variable).
  2. Loads every vector from the table into memory and fits `sklearn.ensemble.IsolationForest(contamination=0.05, random_state=2026, n_estimators=200)` on them.
  3. Persists the model's per-row outlier decision back into the same LanceDB table as a new boolean column named `is_outlier`. The column MUST be present after the script runs and MUST have Arrow type `bool`.
  4. Exposes a callable `top_outliers(k: int = 20) -> list[int]` that returns the `id` values (as plain Python ints) of the `k` rows with the most-negative `decision_function` score (i.e. the strongest anomalies), sorted from most anomalous to least anomalous.
- Implement a `run.py` script at `/home/user/myproject/run.py` that, when executed with `python3 run.py`, performs the full pipeline end-to-end: fits the model, writes the `is_outlier` column, then prints `top_outliers(20)` as a JSON list to stdout on a single line prefixed by `TOP20=`.
- Both scripts must be re-runnable: running `python3 run.py` a second time must NOT crash (the `is_outlier` column already exists in the table after the first run; the candidate must handle that case gracefully — for example by detecting the existing column and using update semantics instead of `add_columns`).

## Implementation Hints
- Read the run id with `os.environ["ZEALT_RUN_ID"]` and build the table name as `events_${run-id}`.
- LanceDB 0.25.3 is installed system-wide; use `lancedb.connect(...)` and `db.open_table(...)`.
- `IsolationForest.predict(...)` returns `+1` for inliers and `-1` for outliers; `IsolationForest.decision_function(...)` returns a float that is **lower** for more anomalous points.
- LanceDB's `table.add_columns({"col": "<sql>"})` only accepts a SQL expression and cannot inject a different per-row value. To persist a per-row boolean you can: add a default `is_outlier` column first (e.g. `add_columns({"is_outlier": "false"})`) and then run a single `table.update(where="id IN (...)", values={"is_outlier": True})` for the predicted outlier ids; or drop-and-recreate the table including the new column. Either approach is acceptable as long as the final column ends up as Arrow type `bool`.
- The verifier will invoke `top_outliers(20)` directly by importing `solution.py`; make sure the function works even when called without first re-running the full pipeline (it must lazily fit / load whatever state it needs, or use a cached model on disk).

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 run.py`
- Stdout must contain a single line of the form `TOP20=[<id1>, <id2>, ..., <id20>]` where the bracketed list is a JSON array of 20 integer ids drawn from the rows of the `events_${run-id}` table.
- After `python3 run.py` finishes, the LanceDB table `events_${run-id}` must contain a column named `is_outlier` whose Arrow type is `bool`.
- The total number of rows where `is_outlier IS TRUE` must be 50 ± 10 (i.e. between 40 and 60 inclusive).
- Precision-at-50 against the ground-truth outlier ids must be ≥ 0.90 (i.e. at least 45 of the 50 flagged rows must be actual ground-truth outliers).
- Importing the `solution` module and calling `solution.top_outliers(20)` must return a list of exactly 20 integers, each one belonging to the ground-truth outlier id set.
- Running `python3 run.py` twice in a row must not crash on the second invocation.

