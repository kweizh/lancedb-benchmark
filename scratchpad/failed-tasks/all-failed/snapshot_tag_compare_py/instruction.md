# LanceDB Snapshot Tags & Diff Tool

## Background
LanceDB uses immutable manifests so every write (insert, update, delete) creates a new numeric version. The `table.tags` API lets you attach human-readable labels (similar to git tags) to specific versions, and the tagged versions are protected from `cleanup_old_versions()` pruning. In this task you will build a small tool that captures three named snapshots of an evolving `documents` table and provides a diff utility that compares two snapshots by their tag names using time-travel (`table.checkout(...)`).

## Requirements
- Build a Python module at `/home/user/myproject/solution.py` that uses the **real** LanceDB Python API (no mocking) and exposes two callables:
  - `build_snapshots(db_path: str, table_name: str) -> None` — creates the table, performs the three mutations, and writes three named tags.
  - `diff(db_path: str, table_name: str, tag_a: str, tag_b: str) -> dict` — returns the difference between two snapshots.
- Connect to a local LanceDB database at `/app/db`. The table name MUST be `documents_${run-id}` where `run-id` is read from the `ZEALT_RUN_ID` environment variable so concurrent trials do not collide.
- The table schema is `id: int64`, `text: string`, `vector: fixed_size_list<float32, 16>`.
- Seed phase (write version v1):
  - Insert 50 rows with `id` in `[0, 50)`. Use `numpy.random.default_rng(seed=2026)` to generate the 16-d float32 vectors deterministically; set `text = f"doc-{id}"`.
  - Attach the tag `v1_baseline` to the resulting version.
- Extend phase (write version v2):
  - Append 20 more rows with `id` in `[50, 70)`, vectors drawn from the same RNG sequence (so the seed is reproducible), and `text = f"doc-{id}"`.
  - Attach the tag `v2_extended` to the resulting version.
- Prune phase (write version v3):
  - Delete the rows whose `id < 5` via a SQL predicate (this removes exactly 5 rows).
  - Attach the tag `v3_pruned` to the resulting version.
- The `diff(tag_a, tag_b)` function MUST use `table.checkout(...)` (or equivalent tag-based time-travel) to read the `id` column from each snapshot and return a dict shaped exactly as:
  ```python
  {
      "added_ids":   sorted list of ints present in B but not in A,
      "removed_ids": sorted list of ints present in A but not in B,
      "common_count": int (number of ids in both snapshots)
  }
  ```
  After running, the function MUST call `table.checkout_latest()` so subsequent callers see the live version.
- Provide a CLI entry point so running `python3 /home/user/myproject/solution.py` executes `build_snapshots(...)` end-to-end against `/app/db` and the `documents_${run-id}` table.

## Implementation Hints
- The tags API lives on the table object: `table.tags.create(name, version)`, `table.tags.list()`, `table.checkout(tag_name)`. Verify by browsing https://docs.lancedb.com/tables/versioning.
- `table.version` returns the current numeric version after each write; use it when calling `table.tags.create`.
- LanceDB pins the schema's `vector` column with `pyarrow.list_(pa.float32(), 16)` (fixed-size list of length 16) — use this to avoid Arrow type-mismatch errors.
- `table.delete("id < 5")` accepts a SQL predicate string.
- The candidate is responsible for executing `python3 /home/user/myproject/solution.py` once before the verifier runs so that the three tags and the persisted data exist at `/app/db`.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 /home/user/myproject/solution.py`
- Database path: `/app/db`
- Table name: `documents_${run-id}` (read `run-id` from `ZEALT_RUN_ID`)
- After running the command, opening the table from `/app/db` MUST report the three tags `v1_baseline`, `v2_extended`, and `v3_pruned` via `table.tags.list()`.
- The live (latest) table version MUST contain exactly 65 rows.
- `solution.diff("/app/db", "documents_${run-id}", "v1_baseline", "v2_extended")` MUST return `{"added_ids": [50, 51, ..., 69], "removed_ids": [], "common_count": 50}`.
- `solution.diff("/app/db", "documents_${run-id}", "v2_extended", "v3_pruned")` MUST return `{"added_ids": [], "removed_ids": [0, 1, 2, 3, 4], "common_count": 65}`.

