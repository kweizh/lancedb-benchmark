# Two-Tier Retrieval + OLAP Bridge (LanceDB + ClickHouse)

## Background
You are building the query layer of a product-analytics search system. Recall is served by **LanceDB** (a local, embedded vector + full-text-search database) while high-cardinality behavioral events live in a local **ClickHouse** server. A query first recalls candidate documents from LanceDB, then those candidates are enriched and aggregated in ClickHouse (time-bucketed counts, a dimension JOIN, and an exact quantile), and finally fused into a single correctly-ordered result set.

The environment is already prepared for you:
- A **ClickHouse server** is running locally (HTTP on `localhost:8123`, native protocol on `localhost:9000`, user `default`, empty password, database `default`).
- A **LanceDB** database is seeded on disk.

## The seeded data

### LanceDB — table `documents` at `/home/user/hybrid_bridge/data/lancedb`
Columns: `id` (int64), `text` (string), `category` (string), `vector` (float32, dimension 32). A full-text-search (FTS) index has been built on the `text` column.

### ClickHouse — database `default`
- `events`: `event_id UInt64`, `doc_id Int64`, `user_id UInt32`, `event_type String`, `ts DateTime`, `value Float64`. One document (`doc_id`) has many events.
- `users`: `user_id UInt32`, `tier String` (either `free` or `premium`), `region String`.

## The deterministic query embedding (you MUST reproduce it exactly)
Document vectors were produced with a fixed hashing bag-of-words embedding. To score a query against document vectors you must embed the query text with the **identical** function:
1. Lowercase the text and extract tokens with the regex `[a-z0-9]+`.
2. Start from a 32-dimensional zero vector.
3. For each token, compute `idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 32` and add `1.0` to `vector[idx]`.
4. L2-normalize the vector (leave it all-zeros if the norm is 0).
5. The vector dtype MUST be `float32`.

## Requirements
Implement a rerunnable CLI that reads a JSON query file, executes the two-tier pipeline, and writes a JSON result file.

The query file has the shape:
```json
{
  "text": "wireless noise cancelling headphones",
  "window_start": "2024-01-01 00:00:00",
  "window_end": "2024-01-08 00:00:00",
  "top": 5
}
```

Pipeline semantics (must be followed exactly):
1. **Recall (LanceDB FTS)**: The candidate set is the set of every document whose `text` produces a positive full-text-search match for the query `text` (default terms/OR matching). Retrieve them with a limit large enough to return all matches; the candidate set is a set of `id`s (ordering at this stage does not matter).
2. **Vector distance (LanceDB)**: For every candidate document compute `vector_distance`, the L2 `_distance` reported by LanceDB between the embedded query vector and that document's stored `vector`.
3. **Enrichment (ClickHouse)**: For the candidate `doc_id`s, using a single batched lookup (`doc_id IN (...)`) restricted to the half-open time window `ts >= window_start AND ts < window_end`, compute per document:
   - `events_in_window`: total number of events (any tier).
   - `premium_value_sum`: sum of `value` over events whose user is `premium` (JOIN `users` on `user_id`).
   - `p95_value`: the 95th percentile of `value` over all events in the window, computed with ClickHouse `quantileExact(0.95)`.
   - `peak_hour_count`: the largest per-hour event count when events are bucketed by `toStartOfHour(ts)` (any tier).
   - A candidate that has no events inside the window MUST still appear with all four numeric fields set to `0` / `0.0`.
4. **Fusion & ordering**: `score = premium_value_sum / (1.0 + vector_distance)`, rounded to 6 decimal places. Output the top `top` documents ordered by `score` descending, breaking ties by `doc_id` ascending.

## Implementation Hints
- Use the `lancedb` Python package for recall and vector distance, and either `clickhouse-connect` or `clickhouse-driver` for ClickHouse (both are installed).
- Correct type mapping matters: `doc_id` is a signed 64-bit integer on both sides; pass native Python ints in the `IN (...)` list, not numpy scalars or strings.
- Documents in the recall set that are missing from the ClickHouse aggregation result must be back-filled with zeros; do not silently drop them.
- Project path: `/home/user/hybrid_bridge`
- Command: `python3 run.py --query-file <path/to/query.json> --output <path/to/result.json>`
- The command MUST write a JSON array to the `--output` path. Each element MUST have exactly these keys: `doc_id` (int), `events_in_window` (int), `premium_value_sum` (float), `p95_value` (float), `peak_hour_count` (int), `vector_distance` (float), `score` (float). The array MUST contain at most `top` elements, ordered by `score` descending then `doc_id` ascending.

