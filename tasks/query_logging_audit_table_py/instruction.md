# LanceDB Query Logging Audit Table

## Background
You are building an observability layer on top of a LanceDB knowledge base. A LanceDB database at `/home/user/myproject/data.lancedb` already contains a table named `articles` with 200 rows (columns: `id: int64`, `title: string`, `embedding: fixed_size_list<float, 64>`). Every retrieval performed against `articles` must be audited so operators can later compute latency percentiles, replay queries, and trace which user issued which search.

Implement a `LoggedSearcher` class that wraps `table.search(...)` and persists one audit row to a `query_logs` LanceDB table for every search it performs.

## Requirements
- Implement a Python module that exposes a class named `LoggedSearcher`.
- `LoggedSearcher` MUST be constructible with the LanceDB database URI, the name of the articles table (containing the precomputed 64-d vectors), and the name of the query-logs table.
- Expose an instance method `search(query_vector, top_k, query_id, user_id, query_text="")` that:
  - Runs a vector similarity search against the articles table for the given query vector, returning the top `top_k` hits.
  - Returns the same Python list of hits that a direct `table.search(query_vector).limit(top_k).to_list()` call would return (each hit must include at minimum the `id` and `title` columns).
  - Writes exactly one audit row to the configured `query_logs` table for every call, capturing wall-clock latency, the ordered list of returned ids, the count of returned hits, the supplied identifiers, and a timestamp taken at logging time.
- The candidate chooses the exact schema of `query_logs`, but it MUST contain at least these logical fields (you may name them as you like, but the verifier-facing names below are REQUIRED so the audit can be queried programmatically):
  - `query_id` (string)
  - `user_id` (string)
  - `query_text` (string)
  - `ts` (timestamp64; recommended `pyarrow.timestamp("us")` or `timestamp("ns")`)
  - `latency_ms` (floating-point, strictly positive)
  - `hit_count` (integer; equal to the length of the returned hits)
  - `top_ids` (Arrow list of int64; the `id` column values from the returned hits, in rank order)
- The `query_logs` table MUST be created lazily on first `search()` if it does not yet exist.
- The class MUST NOT mock LanceDB. Use the real `lancedb` Python package.

## Implementation Hints
- `lancedb.connect(db_uri)` opens (or creates) a database directory.
- `table.search(vector).limit(k).to_list()` returns the top-k rows including any selected columns.
- For lazy table creation, check `db.table_names()` and either `create_table` from a PyArrow schema or write the first row with an explicit schema.
- `time.perf_counter()` gives a high-resolution monotonic clock; convert the delta to milliseconds.
- `datetime.datetime.now(datetime.timezone.utc)` produces a timezone-aware timestamp that pyarrow's `timestamp` accepts.
- The returned `top_ids` list must contain plain Python `int` values matching the `id` column of the hits in rank order so that downstream consumers can store them as `list<int64>`.

## Acceptance Criteria
- Project path: /home/user/myproject
- Solution file: /home/user/myproject/solution.py exporting `LoggedSearcher`.
- Command (import-only; verification imports your class): `python3 -c "from solution import LoggedSearcher"` must succeed without raising.
- Class contract:
  - `LoggedSearcher(db_uri: str, articles_table: str, logs_table: str)` constructor.
  - `searcher.search(query_vector, top_k, query_id, user_id, query_text="")` returns a Python `list` whose entries each contain at least the keys `id` and `title`.
- Audit-table contract: after N successive `.search(...)` calls against a fresh `query_logs` table:
  - The configured `query_logs` LanceDB table exists.
  - It contains exactly N rows.
  - `ts` values are monotonically non-decreasing in insertion order.
  - For each logged row, `latency_ms` is strictly greater than 0.
  - For each logged row, `top_ids` equals the ordered list of `id` values of the hits returned by the corresponding `search()` call.
  - `hit_count` equals the length of `top_ids` for that row.
  - `query_id`, `user_id`, and `query_text` round-trip the values passed to `.search()`.
- Search-result fidelity: for a given query vector, the ids returned by `LoggedSearcher.search(vec, k, ...)` MUST equal, in the same order, the ids returned by `articles_table.search(vec).limit(k).to_list()` against the underlying table.

