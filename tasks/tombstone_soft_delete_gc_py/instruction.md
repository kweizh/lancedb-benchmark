# Soft-Delete Tombstoning and Garbage Collection with LanceDB

## Background
Production vector stores frequently need a two-phase deletion lifecycle: rows are first *soft deleted* (hidden from queries but physically retained so they can be audited or restored), and only later permanently reclaimed by a garbage-collection pass. In this task you implement such a lifecycle on top of a LanceDB table using an application-level tombstone column, then reclaim space using LanceDB's fragment compaction and version cleanup.

## Requirements
Implement a reusable Python module that manages a local, on-disk LanceDB table carrying a boolean `deleted` tombstone column and a `deleted_at` timestamp. It must support soft delete, tombstone-aware nearest-neighbour search, restore, and a garbage-collection pass that hard-deletes aged tombstones and physically reclaims space.

## Implementation Hints
- Use the `lancedb` Python client with a local database directory. No network access, hosted APIs, or model/dataset downloads are permitted; the caller supplies every vector.
- Project path: /home/user/myproject
- Implement everything in the module file `/home/user/myproject/solution.py`.
- Store the database under the directory `/home/user/myproject/lancedb_data`.
- Expose a class constructed as `TombstoneStore(table_name, db_path="/home/user/myproject/lancedb_data")`. It must open the named table when it already exists. Because multiple evaluations may run concurrently, the class must work with an arbitrary caller-supplied `table_name`.
- The table schema has exactly these columns: `id` (int64), `text` (utf8 string), `vector` (fixed-size list of 16 float32 values), `deleted` (boolean), and `deleted_at` (int64 holding Unix epoch seconds, `0` for rows that are not tombstoned).
- `add_documents(docs)`: `docs` is a list of dicts, each with keys `id`, `text`, and `vector` (a 16-element list of floats). The first call creates the table with every row initialized to `deleted = False` and `deleted_at = 0`; subsequent calls append rows to the same table (they must accumulate, not overwrite).
- `soft_delete(ids)`: tombstone the given ids by setting `deleted = True` and stamping `deleted_at` with the current Unix epoch seconds. The rows must remain physically present. Return the number of rows changed.
- `restore(ids)`: only for ids that are currently tombstoned, clear the tombstone by setting `deleted = False` and `deleted_at = 0`. Return the number of rows changed.
- `search(query_vector, k)`: run an L2 nearest-neighbour search for the given 16-dimensional query vector, transparently excluding tombstoned rows using a prefilter, and return at most `k` results. Each result must be a dict with exactly the keys `id`, `text`, and `distance`, ordered by ascending `distance` and breaking ties by ascending `id`.
- `gc(older_than_seconds)`: permanently remove rows whose tombstone has aged — those with `deleted = True` and `deleted_at > 0` and `deleted_at` older than `older_than_seconds` before the current time — then compact table fragments and prune obsolete table versions so that disk space is reclaimed in the same pass. Live rows and non-aged tombstones must be left untouched. Return the number of rows hard-deleted.
- Observable guarantees the grader relies on: after `soft_delete`, the affected rows disappear from `search` yet remain physically present and restorable; after `gc`, aged tombstones are physically gone, non-aged tombstones remain, every still-live row and its vector are unchanged, and both the number of physical data fragments and the length of the table version history are strictly smaller than they were immediately before the `gc` call.

