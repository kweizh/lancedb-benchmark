# LanceDB Table Lifecycle: Create, Overwrite, Open, and Summarize

## Background
LanceDB is an embedded multimodal vector database built on the Lance columnar format. In this task you will exercise the core table lifecycle primitives of the Python SDK: connecting to a local database, creating a table from an explicit Apache Arrow schema with seeded deterministic data, demonstrating the `mode="overwrite"` behavior of `create_table`, reopening the table, and capturing a JSON summary of the resulting database state.

## Requirements
- Connect to a local LanceDB database located at the path given by the `LANCEDB_URI` environment variable. If the variable is unset, default to `/workspace/db`.
- Create a table named `products` using an **explicit PyArrow schema** with the following columns and exact Arrow types:
  - `id`: `int32`
  - `name`: `string`
  - `price`: `float64`
  - `tags`: `list<string>`
  - `vector`: `fixed_size_list<float32>[4]`
- Seed the table with at least 6 rows of deterministic data. Generate the 4-dimensional `vector` values from `numpy.random.default_rng(7)`.
- Demonstrate `mode="overwrite"` semantics by recreating the same `products` table once with a different (but schema-compatible) set of rows, then restore the original 6-row dataset (again via `mode="overwrite"`) so the final state has exactly the original 6 rows.
- Reopen the table with `db.open_table("products")` and use the resulting `Table` object to read the final row count.
- Write a JSON summary to `/workspace/output/table_state.json` with these top-level keys:
  - `tables_in_db`: a JSON array of table names returned by `db.table_names()`, sorted lexicographically.
  - `row_count`: the integer result of `table.count_rows()`.
  - `schema_field_names`: a sorted JSON array of the table's Arrow schema field names.

## Implementation Hints
- The Python LanceDB SDK is available as `lancedb`. Use `lancedb.connect(uri)` to obtain a database handle.
- Build the schema with `pyarrow.schema([...])`. `pa.list_(pa.string())` and `pa.list_(pa.float32(), 4)` are useful here.
- Pass the explicit schema to `db.create_table(..., schema=...)` to avoid type-inference surprises.
- See https://docs.lancedb.com/quickstart and https://docs.lancedb.com/tables/create for canonical create/open patterns.
- Do NOT run any local embedding model or download model weights. Vectors must come from `numpy.random.default_rng(7)`.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: python3 /home/user/myproject/solution.py
- Ensure the script is executed and the artifacts exist.
- After running the command:
  - The LanceDB database at the path given by `LANCEDB_URI` (default `/workspace/db`) contains a table named `products`.
  - `products` has exactly 6 rows.
  - The schema of `products` contains at least the fields: `id`, `name`, `price`, `tags`, `vector`.
  - The file `/workspace/output/table_state.json` exists and is valid JSON.
  - The JSON object contains keys `tables_in_db`, `row_count`, and `schema_field_names`.
  - `tables_in_db` is a sorted JSON array that includes `"products"`.
  - `row_count` is exactly `6`.
  - `schema_field_names` is a sorted JSON array that includes all of: `id`, `name`, `price`, `tags`, `vector`.

