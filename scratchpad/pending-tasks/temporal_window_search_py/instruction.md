# Time-Windowed Semantic Search over a LanceDB Event Stream

## Background
You are building the retrieval layer of an analytics pipeline that needs to answer time-windowed semantic queries over an [LanceDB](https://docs.lancedb.com/) `events` table. A pre-seeded LanceDB database lives at `/home/user/myproject/lancedb_data` and contains a single table `events_${ZEALT_RUN_ID}` (`ZEALT_RUN_ID` is provided in the environment) populated with 1,000 rows. Each row has:

- `id` (Int64) — primary key, strictly increasing from `0`
- `timestamp` (Int64) — unix seconds, uniformly sampled across `[2024-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`
- `event_type` (Utf8) — one of 8 discrete event types
- `payload` (Utf8) — short opaque label
- `vector` (FixedSizeList<float32, 32>) — 32-dimensional float32 vector

You will expose three retrieval functions that combine LanceDB's L2 vector search with SQL `where` predicates on `timestamp` and `event_type` so that filtering happens **server-side** (inside the LanceDB query plan), not in Python after the fact.

## Requirements
Implement a single Python module at `/home/user/myproject/solution.py` that the verifier will import. The module MUST expose three functions:

1. `window_search(query_vec, start_ts, end_ts, k)`
   - Return the top-`k` events whose `timestamp` is strictly within `[start_ts, end_ts)`, ordered by ascending L2 distance to `query_vec`, with ties broken by `id ASC`.
2. `bucketed_search(query_vec, bucket_seconds, num_buckets, k_per_bucket)`
   - Anchored at the constant base timestamp `BASE_TS = 1704067200` (i.e. `2024-01-01T00:00:00Z`), produce `num_buckets` half-open windows `[BASE_TS + i*bucket_seconds, BASE_TS + (i+1)*bucket_seconds)` for `i` in `0..num_buckets-1`.
   - Return a Python `dict` keyed by the integer bucket-start timestamp; each value is the top-`k_per_bucket` events within that bucket, ordered by ascending L2 distance with ties broken by `id ASC`.
   - You MUST issue a **separate LanceDB query** for each bucket with a SQL `where` clause that pushes the timestamp range into the LanceDB query plan. Post-filtering the full table in Python is forbidden.
3. `top_k_per_event_type(query_vec, k_per_type)`
   - For every distinct `event_type` present in the table, return the top-`k_per_type` events of that type, ordered by ascending L2 distance with ties broken by `id ASC`.
   - Returns a Python `dict` keyed by the event type string; each value is the ordered list.
   - Filtering by `event_type` MUST be expressed in the LanceDB SQL `where` clause.

Each event returned by the three functions MUST be a `dict` containing AT LEAST the keys `id` (int), `timestamp` (int), `event_type` (str), and `payload` (str). The exact L2 distance value is not asserted; only the id order is checked.

## Implementation Hints
- Read `ZEALT_RUN_ID` from the environment to construct the table name `events_${ZEALT_RUN_ID}` and open the table from `/home/user/myproject/lancedb_data`.
- Use `table.search(query_vec).where("...").limit(...)` — LanceDB pre-filters by default. See https://docs.lancedb.com/search/filtering and https://docs.lancedb.com/search/vector-search.
- `query_vec` is a Python list / numpy array of 32 floats. The vector column uses the default L2 metric.
- Tie-break by `id ASC` is required even though LanceDB itself does not guarantee a stable order among rows with equal distance. After collecting the top-`k`, sort by `(distance, id)` yourself before returning.
- You may cache the opened `Table` handle at module load time, but do NOT cache results across calls.

## Acceptance Criteria
- Project path: /home/user/myproject
- Solution module: /home/user/myproject/solution.py
- The module MUST be importable via `python3 -c 'import sys; sys.path.insert(0, "/home/user/myproject"); import solution'` without errors.
- The module MUST expose three top-level callables: `window_search`, `bucketed_search`, `top_k_per_event_type` with the signatures described above.
- For every test call the verifier performs, the returned list of events MUST be in the exact id order produced by a brute-force numpy ground-truth computation (L2 distance ASC, id ASC tie-break) using only rows that satisfy the SQL predicate(s) for that call.
- `bucketed_search` MUST return a `dict` with exactly `num_buckets` keys; each key is the integer bucket-start timestamp `BASE_TS + i*bucket_seconds`.
- `top_k_per_event_type` MUST return a `dict` whose key-set equals the set of distinct event_type strings present in the table.
- The candidate MUST NOT modify the seeded `events_${ZEALT_RUN_ID}` table (no inserts, deletes, or updates).

