# LanceDB Nested `list<struct>` and `struct` Schema

## Background
LanceDB is an open-source vector database built on top of the Lance columnar format. Beyond plain scalar columns, it supports rich Arrow types such as nested `struct` columns and `list<struct>` columns, and lets you filter on struct subfields using SQL dot-notation. This task exercises that capability end-to-end: build a non-trivial Arrow schema, seed deterministic data, run a vector search, and run a SQL filter that reaches into a nested struct.

## Requirements
- Connect to a local LanceDB database whose URI comes from the `LANCEDB_URI` environment variable (default `/workspace/db`).
- Create a table named `papers` with the following **exact** Arrow schema (use `pyarrow` types):
  - `id`: `int64`
  - `title`: `string`
  - `authors`: `list<struct<name: string, affiliation: string>>`
  - `metrics`: `struct<citations: int32, year: int32>`
  - `vector`: `fixed_size_list<float32>[8]`
- Seed the table with exactly 10 deterministic rows. Vary the leading author name (e.g., `alice`, `bob`, `carol`, ...), the citation count, and the year so that some `metrics.year` values are `>= 2022` and others are not.
- After seeding:
  1. Run a vector search against the `vector` column with a fixed query vector and `limit=3`.
  2. Run a metadata query that filters rows where `metrics.year >= 2022` (using dot-notation on the struct subfield) and returns up to 50 rows.
- Write the combined results to `/workspace/output/nested_results.json` as a JSON object with two keys:
  - `topk_titles`: list[str] — titles of the top-3 nearest neighbours, in rank order.
  - `recent_ids_sorted`: list[int] — ids of all rows with `metrics.year >= 2022`, sorted ascending.

## Implementation Hints
- Use `pyarrow` to build the schema. `pa.list_(pa.struct([...]))` and `pa.struct([...])` are the relevant constructors; the vector column should be `pa.list_(pa.float32(), 8)` (fixed-size list).
- Build the data as a `pyarrow.Table` (or a list of dicts that matches the schema) and pass it to `db.create_table("papers", data=..., schema=..., mode="overwrite")`.
- For determinism, seed `numpy.random.default_rng` with a fixed seed and use that to generate the 8-d float32 vectors.
- For the nested filter, LanceDB's SQL parser supports dot-notation on struct subfields, so a `where="metrics.year >= 2022"` clause on the search builder is sufficient. If the search builder requires a vector for `to_list()`, supply a placeholder zero vector and a generous `limit` — the filter is what matters for the second query.
- Create `/workspace/output/` if it does not already exist before writing the result file.

## Acceptance Criteria
- Project path: /workspace/solution
- Command: `python3 /workspace/solution/solution.py`
- After the command exits successfully:
  - A LanceDB table named `papers` exists at `${LANCEDB_URI:-/workspace/db}` with the exact Arrow schema described above (column names, types, struct field names, and the fixed-size-list width of 8 must all match).
  - The table contains exactly 10 rows.
  - The file `/workspace/output/nested_results.json` exists and parses as a JSON object with keys `topk_titles` (list of 3 strings) and `recent_ids_sorted` (sorted ascending list of integers, each present in the table and corresponding to a row whose `metrics.year >= 2022`).
  - `topk_titles` matches the top-3 results returned by a vector search on the `vector` column with the agreed fixed query vector and `limit=3`.

