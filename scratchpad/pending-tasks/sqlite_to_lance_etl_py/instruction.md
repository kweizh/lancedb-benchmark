# Incremental SQLite → LanceDB ETL Sync

## Background
A news service stores article metadata and bodies in a SQLite database that is updated continuously by editors. Downstream consumers query a LanceDB table for semantic search over these articles. You will implement an **incremental ETL pipeline** that brings the LanceDB table into sync with new, updated, and deleted SQLite rows since the last sync watermark, using real OpenAI `text-embedding-3-small` embeddings.

## Requirements
- Implement a Python module at `/home/user/myproject/solution.py` that exposes a single function:
  ```python
  def sync(sqlite_path: str, table_name: str, since_ts: int) -> dict:
      ...
  ```
- The source `articles` table in SQLite has schema `id INTEGER, title TEXT, body TEXT, category TEXT, updated_at INTEGER, deleted INTEGER`.
- Only rows with `updated_at > since_ts` should be considered; among these:
  - Rows with `deleted = 0` are **upserted** into LanceDB via `merge_insert("id").when_matched_update_all().when_not_matched_insert_all()`.
  - Rows with `deleted = 1` are **removed** from LanceDB via `tbl.delete("id IN (...)")`.
- Compute per-row vector embeddings with **real OpenAI `text-embedding-3-small`** for upserts. Embed `title + "\n\n" + body`.
- Connect LanceDB at `/home/user/myproject/lancedb_data`. The LanceDB table name passed in is the full table name including the `${ZEALT_RUN_ID}` suffix that the caller appends; just call `db.open_table(table_name)` / `db.create_table(table_name, ...)` as needed.
- The LanceDB schema must contain (at minimum): `id (int64)`, `title (string)`, `body (string)`, `category (string)`, `updated_at (int64)`, and a 1536-dim float32 `vector` column.
- Return a dict with exact keys: `{"inserted": int, "updated": int, "deleted": int, "high_water_ts": int}` where:
  - `inserted` = upsert-batch rows whose `id` was not previously in the LanceDB table
  - `updated` = upsert-batch rows whose `id` was already present
  - `deleted` = number of ids removed via `tbl.delete`
  - `high_water_ts` = max `updated_at` across all considered rows (upserts + deletes), or `since_ts` if no rows were considered.

## Implementation Hints
- Read `ZEALT_RUN_ID` from the environment; the caller will pass `articles_${ZEALT_RUN_ID}` as `table_name`.
- Use the standard library `sqlite3` module to read the source table.
- Use the official `openai` Python SDK (≥1.50) with the `OPENAI_API_KEY` environment variable. Batch the embedding API call (e.g., 64 texts per request) for efficiency.
- Use `lancedb.connect` and `db.open_table` / `db.create_table`. Hold a `pyarrow` schema so that initial creation produces a `fixed_size_list<float32, 1536>` vector column.
- To compute `inserted` vs `updated` counts, look up which of the candidate ids already exist before calling `merge_insert`. A `count_rows(filter="id IN (...)")` or a small `where`-filtered scan is sufficient.
- Embedding caching by content hash is permitted but not required.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: python3 -c "import solution; print(solution.sync(...))" (the verifier imports `solution` directly)
- The `sync` function:
  - Accepts `sqlite_path: str`, `table_name: str`, `since_ts: int` and returns a dict with keys `inserted`, `updated`, `deleted`, `high_water_ts` (all `int`).
  - Reads only SQLite rows where `updated_at > since_ts`.
  - Upserts non-deleted rows into LanceDB via `merge_insert("id").when_matched_update_all().when_not_matched_insert_all()`.
  - Removes deleted rows via `tbl.delete("id IN (...)")`.
  - Embeds upsert texts with real OpenAI `text-embedding-3-small` (1536-dim float32 vectors).
- LanceDB connection: `/home/user/myproject/lancedb_data`.
- LanceDB table name: provided by the caller; it will include the `${ZEALT_RUN_ID}` suffix (e.g., `articles_${ZEALT_RUN_ID}`).
- LanceDB schema must include `id, title, body, category, updated_at, vector(1536-d float32)`.
- After a full initial sync followed by an incremental sync with 5 inserts, 10 updates, and 7 deletes (rows with `updated_at` strictly greater than the watermark), the final LanceDB table contents and the returned counts must match exactly.

