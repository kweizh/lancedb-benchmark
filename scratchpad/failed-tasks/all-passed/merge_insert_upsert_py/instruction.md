# LanceDB Merge Insert Upsert Workflow

## Background
You are working with LanceDB, an embedded vector database that supports the `merge_insert` API for upserting rows (update existing keys and insert new ones in a single atomic operation). Your job is to write a Python script that implements a deterministic upsert workflow against a local LanceDB database and exports the final state of selected rows.

## Requirements
- Connect to a local LanceDB database. Read the connection URI from the `LANCEDB_URI` environment variable, with default `/workspace/db`.
- Create a table named `users` with the following Arrow schema (use `pyarrow` types):
  - `id`: `int64`
  - `email`: `string`
  - `score`: `float32`
  - `vector`: fixed-size list of `float32` with length `8`
  The table must be created fresh on every run (overwrite mode or drop-then-create) so the workflow is idempotent.
- Seed the table with exactly 10 rows whose `id` values are `1..10` (inclusive). Use these exact seed values so the workflow is reproducible:
  - `email = f"user_{id}@example.com"` for `id` in `1..10`.
  - `score`: cast the first 10 floats of `numpy.random.default_rng(0).random(10)` to `float32`. The value at index `i` is the score for row `id == i + 1`.
  - `vector`: a length-8 `float32` array, generated as `numpy.random.default_rng(100 + id).random(8).astype("float32")` for each row.
- Perform TWO upsert batches against the SAME table using `table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)`:
  1. UPDATE batch — incoming rows for `id` in `{2, 5, 7}` with brand-new values:
     - `email = f"updated_{id}@example.com"`
     - `score = float32(0.5 + 0.1 * id)` (so id 2 → 0.7, id 5 → 1.0, id 7 → 1.2)
     - `vector = numpy.random.default_rng(200 + id).random(8).astype("float32")`
  2. INSERT batch — incoming rows for `id` in `{11, 12}` (these are NEW keys, not present in the table):
     - `email = f"new_{id}@example.com"`
     - `score = float32(0.5 + 0.1 * id)` (so id 11 → 1.6, id 12 → 1.7)
     - `vector = numpy.random.default_rng(200 + id).random(8).astype("float32")`
- After both upserts, call `table.count_rows()` and write the final state to `/workspace/output/upsert_state.json`.

## Implementation Hints
- Use `lancedb.connect(uri)` to open the database, then `db.create_table("users", data=..., mode="overwrite")` (or `db.drop_table("users", ignore_missing=True)` followed by `db.create_table(...)`).
- Build each incoming batch as a `pyarrow.Table` whose schema matches the table schema exactly, including the fixed-size list vector column. A convenient way to build the vector column is `pa.FixedSizeListArray.from_arrays(pa.array(flat_floats, type=pa.float32()), 8)`.
- Apply the same `merge_insert("id")` pipeline twice — one call per batch.
- Make sure `/workspace/output/` exists before writing JSON (e.g. `os.makedirs("/workspace/output", exist_ok=True)`).
- The output JSON file must be a JSON array of objects sorted by `id` ascending, filtered to ids `{1, 2, 5, 7, 10, 11, 12}`. Each object has the keys `id` (int), `email` (string), and `score` (number, rounded to a JSON-serializable float). Do NOT include the vector column in the output.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 /home/user/myproject/upsert.py`
- Database URI: `/workspace/db` (read from env `LANCEDB_URI`).
- Table name: `users`.
- Final table row count (via `table.count_rows()`): exactly `12`.
- Output file: `/workspace/output/upsert_state.json` exists and contains a JSON array of exactly 7 objects sorted by `id` ascending.
- Each output element must have keys `id` (int), `email` (string), and `score` (number).
- The 3 updated rows (ids 2, 5, 7) reflect the upsert batch values (`updated_<id>@example.com` and `score = 0.5 + 0.1 * id`).
- The 2 inserted rows (ids 11, 12) reflect the insert batch values (`new_<id>@example.com` and `score = 0.5 + 0.1 * id`).
- The unmodified rows (ids 1, 10) retain their original seed values (`user_<id>@example.com` and the deterministic RNG-derived score).

