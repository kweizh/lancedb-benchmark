# Instrument LanceDB Searches with OpenTelemetry Tracing

## Background
The operations team needs distributed-tracing observability over a LanceDB-backed retrieval service. They want every `connect`, `search`, and result-materialization call to emit OpenTelemetry spans with LanceDB-specific attributes, exported as JSON Lines so that an existing log-shipping pipeline can pick them up. Your job is to write the instrumented Python service and a CLI that drives it.

The table to query was already seeded by the container entrypoint with 50 rows of 32-d vectors plus a `content` text column with a native FTS index.

## Requirements
- Implement a `LanceDBSearchService` class in `/home/user/myproject/solution.py` whose constructor opens a LanceDB connection and exposes a `search(query, k, query_type="vector")` method.
  - `query_type` accepts `"vector"`, `"fts"`, or `"hybrid"`.
  - For `"vector"`, `query` is a `list[float]` of length 32.
  - For `"fts"`, `query` is a `str`.
  - For `"hybrid"`, `query` is a `dict` with keys `text` (str) and `vector` (list[float]).
  - The return value is a `list[dict]` (each row materialized from LanceDB).
- Instrument the service with OpenTelemetry tracing using the real `opentelemetry-api` and `opentelemetry-sdk` packages.
  - Emit a span for the initial `connect` call.
  - Emit a span for every `search(...)` invocation.
  - Emit a separate child span for the result materialization (`to_pandas` or `to_list`).
  - The `search` span must carry attributes `lancedb.table`, `lancedb.query_type`, and `lancedb.k`.
  - The materialization span must carry `lancedb.result_count` and `lancedb.latency_ms`, and its parent must be the corresponding `search` span (same trace id, child `parent_id` == search span id).
- Export spans as JSON Lines to `/tmp/otel_spans.jsonl`. Either:
  - Run the official `otel/opentelemetry-collector-contrib` binary inside the container with a config that writes spans to that JSONL path, and use the OTLP HTTP exporter (`opentelemetry-exporter-otlp-proto-http`) from your Python process, OR
  - Use an in-process file exporter that writes one JSON object per span per line to the same path.
  - Either way, each line in the file must be a JSON object with at least these keys: `name`, `trace_id`, `span_id`, `parent_id` (null for roots), `attributes` (dict), `start_time`, `end_time`.
- Provide a CLI at `/home/user/myproject/run_queries.py` that loads the service and runs N queries, where N is provided via the `--n` flag. The CLI must run a mix of vector / fts / hybrid queries (at least one of each) and exit 0 when all queries succeed.
- The table name is `tracing_docs_${ZEALT_RUN_ID}` and the LanceDB data directory is `/app/lancedb_data/`. Both are populated by the entrypoint before your code runs.

## Implementation Hints
- The OpenTelemetry SDK ships `TracerProvider`, `SimpleSpanProcessor`/`BatchSpanProcessor`, and `SpanExporter`. A minimal custom `SpanExporter` that serializes each `ReadableSpan` to one JSON line is the simplest path; the OTLP-collector route works too but requires running and configuring the collector binary.
- When you implement a custom exporter, remember to call `force_flush()` (or shut the provider down) before the process exits so spans on a `BatchSpanProcessor` reach disk.
- LanceDB hybrid search uses `.search(query_type="hybrid").vector(vec).text(text)`. FTS uses `.search(text, query_type="fts")` against a column with an FTS index. Vector search is `.search(vec).limit(k).to_list()`.
- The materialization span is a child of the search span — open it inside the `with tracer.start_as_current_span("lancedb.search")` block, around the `to_list()` / `to_pandas()` call.
- Read `ZEALT_RUN_ID` from the environment to build the table name.

## Acceptance Criteria
- Project path: /home/user/myproject
- Solution module: /home/user/myproject/solution.py exposing `LanceDBSearchService` with the `search(query, k, query_type)` method described above.
- CLI: `python3 /home/user/myproject/run_queries.py --n <int>` runs N queries against the seeded table and exits 0.
- Trace log file: /tmp/otel_spans.jsonl exists after the CLI completes, with one JSON object per line.
- Each line includes the fields: `name`, `trace_id`, `span_id`, `parent_id`, `attributes`, `start_time`, `end_time`.
- The file contains at least one span named for connect (e.g. `lancedb.connect`).
- The file contains spans whose attributes include `lancedb.table = "tracing_docs_${ZEALT_RUN_ID}"` and `lancedb.query_type` taking each of the values `vector`, `fts`, and `hybrid` at least once across the run.
- Each search span has `lancedb.k` (int) as an attribute; its child materialization span has `lancedb.result_count` (int) and `lancedb.latency_ms` (number).
- For every materialization span, its `parent_id` equals the `span_id` of a search span emitted in the same `trace_id`.

