# Zero-Downtime Vector Index Swap with LanceDB

## Background
You are upgrading the vector index of a production LanceDB table without dropping in-flight search traffic. The current index works but its tuning parameters need to be changed. You must build the new index on a SHADOW copy of the table while serving live searches, then atomically flip an alias pointer so future searches use the new index.

A base table already exists with a built `IVF_PQ` index. A JSON pointer file `/app/index_pointer.json` tracks which table is currently active. The base table follows the naming pattern `vectors_${ZEALT_RUN_ID}` and is the value the verifier passes into the manager constructor.

## Requirements
- Implement an `IndexSwapManager` class in `/home/user/myproject/solution.py` whose constructor takes a single positional argument `table_name` (the active alias, not the physical table — read the physical name from `/app/index_pointer.json`).
- The manager must expose three methods:
  - `build_shadow_index(index_type: str, params: dict) -> str`: creates a brand-new LanceDB table whose name is `f"{table_name}_shadow_{ZEALT_RUN_ID}"`, copies all rows from the currently active table into it, and builds a vector index of the requested type with the supplied params. Returns the shadow table name. Must NOT mutate `/app/index_pointer.json`.
  - `search(query_vec, k: int) -> list[dict]`: runs a vector search against whatever table is currently named in `/app/index_pointer.json` and returns a list of result dicts. This method must be safe to call concurrently from many threads while `build_shadow_index` and `promote_shadow` are running on another thread.
  - `promote_shadow() -> None`: atomically updates `/app/index_pointer.json` so the active table is the shadow built by the last successful `build_shadow_index` call. After this call, future `search()` invocations must read from the new table.
- The connection root is `/home/user/myproject/lancedb_data` (the same path used by the bootstrap entrypoint).
- All tables created by the solution must be suffixed with `_${ZEALT_RUN_ID}` to stay isolated across concurrent test runs.

## Implementation Hints
- LanceDB tables are independent on disk; copying data is just `db.create_table(new_name, data=source_tbl.to_arrow())` or similar.
- The pointer file is the single source of truth for the active table. Read it on every `search()` call so a concurrent promotion is visible to subsequent searches.
- `promote_shadow` should be atomic — write to a temp file in the same directory and `os.rename` over the pointer to avoid partial writes.
- `wait_for_index` requires a `datetime.timedelta`, never a raw integer.
- IVF_PQ training needs at least 256 rows; the seeded base table contains ≥300.
- Use `nprobes` or default search settings — there is no need to tune them.
- Read `ZEALT_RUN_ID` from the environment.

## Acceptance Criteria
- Project path: /home/user/myproject
- Solution file: /home/user/myproject/solution.py exposing class `IndexSwapManager`.
- The verifier will:
  - Instantiate `IndexSwapManager(f"vectors_{os.environ['ZEALT_RUN_ID']}")`.
  - Launch a background thread that fires 100 random `search(query_vec, k=5)` calls spread evenly over ~10 seconds.
  - Meanwhile call `build_shadow_index("IVF_PQ", {"num_partitions": 4, "num_sub_vectors": 4})` and then `promote_shadow()` from the main thread.
  - After joining the background thread, the verifier asserts:
    1. Every one of the 100 search calls returned a non-empty list with no exceptions raised.
    2. The p99 latency of those 100 calls is strictly less than 500 ms.
    3. `/app/index_pointer.json` now resolves to a physical table whose name ends with `_shadow_${ZEALT_RUN_ID}`.
    4. Opening that shadow table directly and calling `list_indices()` shows exactly one vector index, and `index_stats()` on it reports `num_indexed_rows >= 300`.
    5. After promotion, calling `search()` returns rows that exist in the shadow table (sanity check by id set membership).

