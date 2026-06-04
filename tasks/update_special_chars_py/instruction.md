# LanceDB: Update Rows with Special Characters

## Background
LanceDB tables expose two flavors of in-place updates: `table.update(where=..., values={...})` accepts plain Python values, while `table.update(where=..., values_sql={...})` accepts SQL expressions. When the new string value contains an apostrophe (e.g., `I'm good`, `It's a test`, `O'Brien`), naive use of `values_sql` produces an `Unterminated string literal` SQL error (LanceDB issue #1429). The Python `values=` form sidesteps this footgun entirely because the value is bound, not interpolated, into SQL.

Your job is to write a Python script that creates a small `notes` table, seeds it with 8 rows, performs three updates that each set a column to a string containing a single quote, and writes the post-update state out to disk for verification.

## Requirements
- Connect to LanceDB at the path provided by the `LANCEDB_URI` env var (default `/workspace/db`).
- Create a table named `notes` with columns:
  - `id: int64`
  - `author: string`
  - `body: string`
  - `vector: fixed_size_list<float32>[4]`
- Seed the table with exactly 8 rows whose `id` values are 1..8, populated with simple ASCII `author` and `body` strings and deterministic 4-d float32 vectors.
- Perform the following updates IN ORDER using `table.update(where=..., values={...})` (the Python dict form, NOT `values_sql`):
  1. Set the `body` of the row where `id = 2` to the string `I'm good`.
  2. Set the `body` of the row where `id = 4` to the string `It's a test`.
  3. Set the `author` of the row where `id = 6` to the string `O'Brien`.
- After all updates, read the rows with `id` in 1..8, sort them by `id` ascending, and write them to `/workspace/output/notes_after.json` as a JSON array of objects with keys `id`, `author`, `body` (no vector).

## Implementation Hints
- Use `lancedb.connect(os.environ.get("LANCEDB_URI", "/workspace/db"))`.
- Define the schema with `pyarrow` (`pa.schema([...])`) and `pa.list_(pa.float32(), 4)` for the vector column, then call `db.create_table("notes", data=..., schema=..., mode="overwrite")`.
- Generate deterministic vectors with `numpy.random.default_rng(seed=...)`.
- For each update, pass a Python dict to `values=`, e.g. `tbl.update(where="id = 2", values={"body": "I'm good"})`. Do **not** use `values_sql` — it requires SQL string escaping and is the cause of the apostrophe bug.
- For the final read, you can use `tbl.search().where("id >= 1 AND id <= 8").limit(100).to_list()` or `tbl.to_pandas()` and then sort by `id` before serializing.
- Make sure the output directory exists before writing (`os.makedirs("/workspace/output", exist_ok=True)`).

## Acceptance Criteria
- Project path: /workspace
- Ensure the script is executed and the output artifact exists.
- Log/output file: /workspace/output/notes_after.json
- The output file MUST be a JSON array of exactly 8 objects, each with keys `id` (int), `author` (string), and `body` (string), sorted by `id` ascending.
- The row with `id = 2` MUST have `body` equal exactly to the 8-character string `I'm good`.
- The row with `id = 4` MUST have `body` equal exactly to the 11-character string `It's a test`.
- The row with `id = 6` MUST have `author` equal exactly to the 7-character string `O'Brien`.
- The other rows (`id` in {1, 3, 5, 7, 8}) MUST retain their original seeded `author` and `body` values.
- The LanceDB table `notes` at `LANCEDB_URI` MUST also reflect these same three updates (the JSON output and the table must agree).

