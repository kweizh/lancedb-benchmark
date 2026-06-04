# LanceDB Versioning and Time Travel

## Background
LanceDB tracks every mutation as a new immutable version of the table. You can list every version that has ever existed with `list_versions()`, jump to any prior version with `checkout(version)`, and return to the most recent state with `checkout_latest()`. In this task you will exercise these APIs end-to-end on a small `docs` table, walking it through a known sequence of mutations and then time-travelling between two of the versions you produced.

## Requirements
- Connect to LanceDB at the URI given by the `LANCEDB_URI` environment variable (default `/workspace/db`).
- Create a table named `docs` with the schema below and seed it with rows `id = 1..5`:
  - `id`: `int64`
  - `title`: `string`
  - `vector`: `fixed_size_list<float32>[4]`
- After the initial seed, perform this exact mutation sequence on the table, in order:
  1. Add 3 more rows with `id` 6, 7, 8.
  2. `update` the row where `id = 3`, setting `title` to `"v3-updated"`.
  3. `delete` the row where `id = 1`.
- Call `table.list_versions()` and capture the total number of versions.
- Time-travel back to the early version that exists immediately after the initial seed (before any of the three mutation steps above), read all rows in that snapshot, then call `checkout_latest()` and read all rows in the latest version.
- Write the final result as JSON to `/workspace/output/version_state.json`.

## Implementation Hints
- LanceDB's versioning APIs are documented at https://docs.lancedb.com/tables/versioning. The relevant methods on a `Table` are `list_versions()`, `checkout(version)`, and `checkout_latest()`.
- Every write operation (create, add, update, delete) produces a new monotonically increasing integer version. Inspect what `list_versions()` returns to identify the version that corresponds to "right after the initial seed".
- After `checkout(early_version)` the table object reads from the historical snapshot; remember to call `checkout_latest()` before reading the latest state.
- Use deterministic 4-dimensional vectors (any fixed numpy values are fine). No model inference is required.
- The output directory `/workspace/output/` already exists in the environment; just create the JSON file inside it.

## Acceptance Criteria
- Project path: /workspace
- Ensure the real LanceDB mutations are executed against the table at `LANCEDB_URI` and the output artifact exists.
- Output file: /workspace/output/version_state.json
- The output file MUST be valid JSON containing exactly these top-level keys:
  - `num_versions`: integer, the value of `len(table.list_versions())` after all mutations are applied.
  - `early_version_ids_sorted`: JSON array of integers, the sorted list of `id` values visible in the early snapshot (immediately after the initial seed).
  - `latest_version_ids_sorted`: JSON array of integers, the sorted list of `id` values visible in the latest version.
  - `latest_id3_title`: string, the `title` of the row with `id = 3` in the latest version.

