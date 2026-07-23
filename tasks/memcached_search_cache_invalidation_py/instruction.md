# Read-Through Search Cache for LanceDB backed by Memcached

## Background
Production vector-search services put a cache in front of the database so that repeated queries do not re-run the (relatively expensive) nearest-neighbor search. The hard part is *correctness under mutation*: once rows are inserted or updated, any cached result that no longer reflects the data must become unreachable, otherwise clients silently receive stale hits.

You must build a read-through search cache in front of a LanceDB table, backed by a **local memcached daemon** (already running on `127.0.0.1:11211`). No external network or hosted API is available or allowed.

## Environment (provided)
- A memcached daemon is running on `127.0.0.1:11211`.
- A LanceDB table has already been seeded. Read `/home/user/myproject/fixture.json` to learn its location. It contains the keys `db_path` (the LanceDB directory), `table_name`, `dim` (vector dimensionality), and `num_rows`. The table schema is `id: int64`, `category: string` (one of `A`,`B`,`C`,`D`,`E`), and `vector: fixed_size_list<float32, dim>`.
- A Python memcached client (`pymemcache`) is installed.

## Requirements
Implement a class `CachedSearcher` in `/home/user/myproject/solution.py` that provides a read-through cache with correct invalidation:
- `__init__(self, db_path, table_name, memcached_host="127.0.0.1", memcached_port=11211, ttl_seconds=300)` — connect to the LanceDB table and the memcached server.
- `search(self, query_vector, k=10, filter=None) -> list[dict]` — on a **cache hit**, return the cached result without touching LanceDB; on a **cache miss**, run the LanceDB vector search (L2), store the serialized result in memcached with the configured TTL, and return it. When `filter` is provided it is a LanceDB SQL predicate applied to the search (e.g. `"category = 'A'"`).
- `add(self, rows) -> None` — append the given rows (list of dicts matching the schema) to the LanceDB table, then bump the dataset version.
- `update(self, where, values) -> None` — update matching rows in the LanceDB table via a SQL `where` predicate and a `values` dict, then bump the dataset version.
- `current_version(self) -> int` — return the current dataset version.

## Implementation Hints
- Project path: `/home/user/myproject`
- Build a **stable cache key** from a canonical encoding of the query vector contents (for example its float32 byte representation), together with `k`, the `filter` string, the `table_name`, and the current dataset version. Identical `(query_vector, k, filter)` inputs against the same data must map to the same key; different inputs must map to different keys.
- Maintain a **per-table `dataset_version` counter inside memcached** so it is shared across processes and `CachedSearcher` instances. Every `add`/`update` must atomically increment it. Because the current version is part of the cache key, bumping it makes all previously cached entries for that table unreachable — this is how invalidation works. A writer created in one `CachedSearcher` instance must invalidate reads served by a *different* instance pointing at the same table.
- A cache miss must reflect the **latest committed state** of the table, including rows written by other `CachedSearcher` instances/connections — make sure your read path observes freshly committed data rather than a stale snapshot taken when the table was first opened.
- Store cached values with the configured TTL so that stale-but-unmutated entries also expire on their own.
- Each result returned by `search` must be a dict with exactly the keys `id` (int), `category` (str), and `_distance` (float), ordered by ascending `_distance`. What you return on a hit must be byte-for-byte identical to what you returned on the originating miss.
- All vectors are plain numeric lists; cast to float32 for the LanceDB query.

