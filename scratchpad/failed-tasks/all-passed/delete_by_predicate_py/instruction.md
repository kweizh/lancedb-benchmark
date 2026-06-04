# Delete LanceDB Rows by SQL Predicate

## Background
You are operating on a LanceDB OSS database. A `logs` table needs to be cleaned up by removing rows that match several SQL predicates. You will use the Python SDK's `Table.delete(<predicate>)` API to perform the soft deletes, then export a summary of the surviving rows so the cleanup can be audited.

## Requirements
- Connect to LanceDB using the URI from the `LANCEDB_URI` environment variable (default: `/workspace/db`).
- Create a table named `logs` with the following Arrow schema (in this exact column order):
  - `id`: `int64`
  - `level`: `string` (one of `"info"`, `"warn"`, `"error"`)
  - `seq`: `int32`
  - `vector`: fixed-size list of `float32` with list size `4`
- Seed the table with exactly 50 deterministic rows where for each `id` in `1..=50`:
  - `level` cycles through `info`, `warn`, `error` (so `id=1`→`info`, `id=2`→`warn`, `id=3`→`error`, `id=4`→`info`, ...).
  - `seq = id * 2`.
  - `vector` may be any fixed deterministic 4-float vector (e.g., derived from `id` with a fixed seed). The verifier does not inspect vector values.
- After seeding, run these three deletes IN ORDER using `Table.delete(<sql_predicate>)`:
  1. Delete every row whose `level` equals `'warn'`.
  2. Delete every row whose `level` equals `'info'` AND whose `seq` is strictly greater than `60`.
  3. Delete every row whose `id` is in the set `{5, 9, 13}`.
- After all deletes complete, write a JSON file to `/workspace/output/delete_state.json` whose top-level object has exactly these keys:
  - `total_rows`: integer count of rows remaining in the `logs` table (from `Table.count_rows()` after deletes).
  - `remaining_ids_sorted`: a JSON array of integers — every surviving `id` value, sorted in ascending order.

## Implementation Hints
- Use `lancedb.connect(...)` against the `LANCEDB_URI` path and `db.create_table("logs", schema=...)` to materialize an empty table with a PyArrow schema, then append rows with `tbl.add(...)`. You can also create the table directly from a PyArrow Table that already contains the seed rows.
- The four columns must be created in the order listed above (`id`, `level`, `seq`, `vector`).
- `Table.delete(predicate)` takes a SQL `WHERE`-style string; quote string literals with single quotes (e.g., `level = 'warn'`).
- After deletes, derive `remaining_ids_sorted` from a query such as `tbl.to_pandas()` / `tbl.to_arrow()` / `tbl.search().select(["id"]).limit(<n>).to_list()`; make sure the output list is sorted in ascending numeric order.
- Ensure the parent directory `/workspace/output/` exists before writing the JSON file.

## Acceptance Criteria
- Project path: /home/user/lance_delete
- Ensure the real LanceDB writes and deletes are executed against the database at `LANCEDB_URI` (default `/workspace/db`) and the artifact is produced.
- Output file: /workspace/output/delete_state.json
- The table `logs` exists at `LANCEDB_URI` after the script runs.
- The output JSON file has exactly the top-level keys `total_rows` (integer) and `remaining_ids_sorted` (array of integers in ascending order). No other top-level keys are required.
- `total_rows` MUST equal the actual `Table.count_rows()` value reported by LanceDB after the deletes.
- `remaining_ids_sorted` MUST equal the sorted list of surviving `id` values from the `logs` table.

