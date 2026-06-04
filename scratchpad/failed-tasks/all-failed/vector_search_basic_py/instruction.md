# Basic LanceDB Vector Search

## Background
You are integrating LanceDB (the open-source embedded multimodal lakehouse) into a small Python service. The service indexes a tiny in-memory corpus with precomputed 8-dimensional vectors and runs a single nearest-neighbor query. Because the runtime has no GPU and no model weights, **all vectors are precomputed deterministically with `numpy.random.default_rng(42)`** — no embedding model is ever downloaded or invoked.

## Requirements
- Write a Python script `solution.py` that:
  - Connects to a local LanceDB database at the path given by the environment variable `LANCEDB_URI` (default to `/workspace/db` when the variable is not set).
  - Creates a table named `documents` with the following Arrow schema (in this exact column order):
    - `id`: `int64`
    - `text`: `string`
    - `vector`: `fixed_size_list<float32>[8]` (8-dimensional float32 vector)
  - Inserts 12 rows generated deterministically as follows, using a single `numpy.random.default_rng(42)` instance, draws in this exact order:
    1. `vectors = rng.standard_normal((12, 8)).astype(numpy.float32)`
    2. `query = rng.standard_normal(8).astype(numpy.float32)`
  - Row `i` (for `i` in `0..11`) must have `id = i`, `text = f"document_{i}"`, and `vector = vectors[i]`.
  - Runs a vector similarity search using `query` as the query vector and returns the top 5 nearest rows using LanceDB's default L2 distance metric.
  - Serializes the top 5 results as a JSON array to `/workspace/output/results.json`. Each list element must be a JSON object with exactly these keys:
    - `id` (integer)
    - `text` (string)
    - `_distance` (float)
  - The list must be ordered from nearest to farthest (smallest `_distance` first).

## Implementation Hints
- Connect with `lancedb.connect(...)` and create the table from a PyArrow Table or a list of dicts. Either pass a PyArrow schema or let LanceDB infer it — just make sure the vector column is a `pa.list_(pa.float32(), 8)` (fixed-size list of length 8).
- Use `table.search(query_vector).limit(5)` and convert results with `.to_list()` or `.to_pandas()`. The result already contains the `_distance` column.
- The script must be **idempotent**: running it twice in the same database should not fail. Either drop the table before creating it, or use a create-with-mode pattern.
- Make sure the output directory exists before writing (`os.makedirs("/workspace/output", exist_ok=True)`).
- Cast all JSON values to native Python types (e.g., `int(row["id"])`, `float(row["_distance"])`).

## Acceptance Criteria
- Project path: /workspace/myproject
- Command: `python3 /workspace/myproject/solution.py`
- The command must complete successfully (exit code 0).
- After execution, the file `/workspace/output/results.json` must exist and contain a JSON array of exactly 5 objects.
- Each object must contain the keys `id` (int), `text` (string), and `_distance` (float), and only those keys.
- Results must be ordered ascending by `_distance` (nearest first).
- The `id` values must match the true top-5 nearest neighbors computed against the same seeded `numpy.random.default_rng(42)` data using L2 distance.
- The LanceDB table at `${LANCEDB_URI:-/workspace/db}/documents.lance` (or equivalent path) must exist and contain exactly 12 rows with the documented schema.

