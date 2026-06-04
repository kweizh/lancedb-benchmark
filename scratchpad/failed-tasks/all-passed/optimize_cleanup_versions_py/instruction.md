# LanceDB Compaction and Old-Version Cleanup

## Background
Long-lived LanceDB tables accumulate two kinds of overhead: many small fragments from incremental writes, and a growing trail of historical versions retained for time travel. The `Table.optimize` method bundles both compaction and pruning of old versions into a single call, and the `cleanup_older_than` parameter (a `datetime.timedelta`) controls how aggressively past versions are reclaimed.

In this task you will write a Python script that exercises this maintenance path end-to-end against a real local LanceDB store.

## Requirements
- Connect to a LanceDB store at the URI given by the `LANCEDB_URI` environment variable (default `/workspace/db`).
- Create a table named `metrics` with the following Arrow schema:
  - `id: int64`
  - `value: float32`
  - `vector: fixed_size_list<float32>[8]`
- Seed the table with 100 rows in the initial `create_table` call.
- After creation, perform 8 small `table.add(...)` calls of 10 rows each. Each call must be its own write operation so that many fragments and many versions are produced.
- Capture `len(table.list_versions())` BEFORE running any optimize / cleanup.
- Call `table.optimize(cleanup_older_than=timedelta(seconds=0))` to compact fragments and prune every version older than the current one.
- Capture `len(table.list_versions())` AFTER the optimize call.
- Capture `table.count_rows()` AFTER the optimize call.
- Write the result JSON to `/workspace/output/optimize_state.json` with exactly these keys:
  - `pre_optimize_versions` (int)
  - `post_optimize_versions` (int)
  - `post_optimize_row_count` (int)

## Implementation Hints
- Use `lancedb.connect(uri)` from the synchronous Python API.
- Construct the Arrow schema explicitly with `pyarrow` so the fixed-size-list dimensionality is preserved.
- Generate deterministic vectors with `numpy` (no external models, no GPU).
- Remember that `cleanup_older_than` accepts a `datetime.timedelta`, not an integer count of seconds.
- The current version of the table is always retained, so post-optimize version count will be at least 1.
- Make sure the `/workspace/output/` directory exists before writing the result file.

## Acceptance Criteria
- Project path: /home/user/myproject
- Ensure the script is executed and the artifacts exist.
- LanceDB store path: read from the `LANCEDB_URI` environment variable (default `/workspace/db`).
- Table name: `metrics`.
- Output file: `/workspace/output/optimize_state.json` containing a single JSON object with keys `pre_optimize_versions`, `post_optimize_versions`, `post_optimize_row_count` (all integers).
- After the script runs:
  - `post_optimize_row_count` MUST equal `180` (100 seed rows + 8 batches of 10).
  - `pre_optimize_versions` MUST be strictly greater than `post_optimize_versions` (cleanup pruned older versions).
  - `post_optimize_versions` MUST be at least `1` (current version is always retained).
- No network access is required; do not call any embedding model or remote service.

