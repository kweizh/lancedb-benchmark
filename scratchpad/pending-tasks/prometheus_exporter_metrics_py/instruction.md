# Prometheus Exporter for LanceDB Query Observability

## Background
You are instrumenting a LanceDB-backed retrieval service. Your job is to ship a Python module that wraps the LanceDB query surface, records standard Prometheus metrics for every query, and exposes them over HTTP in the Prometheus text exposition format so they can be scraped by a Prometheus server.

A `documents_${ZEALT_RUN_ID}` table has already been created for you under `/home/user/myproject/data/` by the container entrypoint. It contains 200 rows with the schema `{id: int64, content: string, vector: fixed_size_list<float32, 32>}` and has a native (non-Tantivy) FTS index built on `content`.

## Requirements
- Implement `/home/user/myproject/solution.py` exposing a `Search` class and a `start_metrics_server(port: int = 9100)` function.
- `Search(table, table_name: str)` MUST provide three query methods:
  - `vector_search(query_vector, k)` — pure vector search.
  - `fts_search(query_text, k)` — full-text search using the native Lance FTS index.
  - `hybrid_search(query_vector, query_text, k)` — hybrid vector + FTS search.
- Every call MUST update three Prometheus metrics from `prometheus_client`:
  - Counter `lancedb_query_total` with labels `query_type` and `table`, incremented by 1 per call.
  - Histogram `lancedb_query_duration_seconds` with labels `query_type` and `table`, observing the wall-clock duration of the underlying LanceDB call, with explicit buckets `[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]`.
  - Gauge `lancedb_table_rows` with label `table`, set to the current `count_rows()` of the wrapped table.
- The `query_type` label MUST be one of `"vector"`, `"fts"`, or `"hybrid"`.
- `start_metrics_server(port=9100)` MUST start a Prometheus HTTP exposition server (via `prometheus_client.start_http_server`) bound to `127.0.0.1` on the given port. Subsequent calls to the same port MUST NOT crash the process.
- A `GET /metrics` request to `http://127.0.0.1:9100/metrics` MUST return Prometheus text exposition format that `prometheus_client.parser.text_string_to_metric_families` can parse.

## Implementation Hints
- Read the LanceDB connection path from the standard location used by the seed step; the table name is `documents_${ZEALT_RUN_ID}` where `ZEALT_RUN_ID` is read from the environment.
- Use `prometheus_client.Counter`, `Histogram`, and `Gauge`. Pass the `buckets=(...)` keyword to `Histogram` to set the histogram bucket edges to the exact list above.
- `prometheus_client.start_http_server(port, addr="127.0.0.1")` is the simplest way to expose `/metrics`.
- Time the LanceDB call (not the metric work) using `time.perf_counter()` or `Histogram.time()`.
- Hybrid queries in lancedb 0.25.3 use `table.search(query_type="hybrid").vector(qvec).text(qtext).limit(k)`. Pure vector and pure FTS use `table.search(qvec)` and `table.search(qtext, query_type="fts")` respectively.
- Refresh the `lancedb_table_rows` gauge on every query so it always reflects the live row count.

## Acceptance Criteria
- Project path: /home/user/myproject
- The module `/home/user/myproject/solution.py` MUST be importable as `import solution` from `/home/user/myproject`.
- Public surface of `solution.py`:
  - `class Search:` with `__init__(self, table, table_name: str)` and instance methods `vector_search(self, query_vector, k)`, `fts_search(self, query_text, k)`, `hybrid_search(self, query_vector, query_text, k)`.
  - `def start_metrics_server(port: int = 9100) -> None:`.
- Metrics exposition (via `GET http://127.0.0.1:9100/metrics`, parsed by `prometheus_client.parser.text_string_to_metric_families`):
  - A metric family named `lancedb_query_total` of type `counter` exists with the labels `query_type` and `table`. Across all 20 verifier-issued queries, the sum of its `_total` samples equals 20.
  - A metric family named `lancedb_query_duration_seconds` of type `histogram` exists with the labels `query_type` and `table`. Its bucket edges include all of `[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]` (the `+Inf` bucket added by the client is allowed), and per-label `_bucket{le="+Inf"}` equals the corresponding `_count`.
  - A metric family named `lancedb_table_rows` of type `gauge` exists with the label `table`, and its sample value for `table="documents_${ZEALT_RUN_ID}"` equals 200.
  - For every metric family above, the `table` label value equals `documents_${ZEALT_RUN_ID}` where `${ZEALT_RUN_ID}` is read from the environment.
  - For `lancedb_query_total` and `lancedb_query_duration_seconds`, the `query_type` label takes at least the values used by the verifier: `"vector"`, `"fts"`, and `"hybrid"`.

