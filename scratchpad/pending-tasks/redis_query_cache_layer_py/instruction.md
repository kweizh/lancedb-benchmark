# Redis-Backed Query Cache for LanceDB

## Background
LanceDB vector and full-text searches can be expensive when repeated for the same query. A common production pattern is to layer a Redis cache in front of LanceDB so that identical queries return immediately from memory.

## Requirements
Implement a Python module `solution.py` inside `/home/user/myproject` that exposes a class `CachedSearch` with the following surface:

- `CachedSearch(table, redis_url, ttl_seconds=60)` — `table` is an open LanceDB `Table` object, `redis_url` is a standard Redis connection URL.
- `search(query, k) -> dict` — `query` is either a numeric vector (list/tuple/ndarray of floats) or a plain text string. Return a dict with exact keys `results` (list of dicts), `cache_hit` (bool), and `latency_ms` (float).
- `invalidate_table()` — delete every cache entry that belongs to this table.

## Implementation Hints
- Build a deterministic cache key by SHA256-hashing the tuple `(table_name, normalized_query_bytes, k)`. Normalize numeric vectors to float32 little-endian bytes; normalize text by encoding it as UTF-8.
- Prefix every key with the table name so `invalidate_table()` can target only this table's entries (Redis `SCAN` + `DELETE`).
- Store the cache value as a serialized list of result dicts (pickle or JSON). Each result dict should be at minimum `{"id": ..., "_distance": ...}` (other LanceDB columns are fine to include).
- Use the standard `redis` Python client (synchronous API is sufficient).
- Apply `EXPIRE`/`SETEX` with the configured `ttl_seconds`.
- Measure latency with `time.perf_counter()` in milliseconds.
- Redis is already installed in the container and started automatically at entrypoint on `localhost:6379`. Use `redis_url="redis://localhost:6379/0"` when constructing the client.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 -c "from solution import CachedSearch"` must import cleanly.
- The `CachedSearch` constructor signature is `CachedSearch(table, redis_url, ttl_seconds=60)`.
- `search(query, k)` returns `{"results": list, "cache_hit": bool, "latency_ms": float}`.
- First call with a fresh `(query, k)` returns `cache_hit=False`.
- Second identical call returns `cache_hit=True` and is at least 5× faster than the first call.
- A different value of `k` for the same query is treated as a separate cache entry (independent miss/hit).
- `invalidate_table()` removes all entries for this table; subsequent identical searches miss again.
- After sleeping `ttl_seconds+1` seconds, the entry has expired and the next search misses again.
- Redis daemon listens on `localhost:6379` inside the container.
- The table used by the verifier is `docs_${ZEALT_RUN_ID}`, pre-seeded at container start.

