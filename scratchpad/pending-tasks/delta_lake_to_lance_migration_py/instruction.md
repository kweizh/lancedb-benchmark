# Delta Lake → LanceDB Migration Pipeline

## Background
A pre-seeded Delta Lake table lives at `/app/delta_data/products` inside the container. It contains product records with embedding vectors and has been written across multiple Delta versions (an initial bulk write plus an update commit that changes the `category` of some rows). Your job is to migrate the latest snapshot into a LanceDB table and to provide a small audit utility that diffs two arbitrary Delta versions and persists the diff into LanceDB.

## Requirements
- Read the latest version of the Delta Lake table at `/app/delta_data/products` using `deltalake`.
- Convert the Arrow data so the embedding column becomes a `fixed_size_list<float32, 32>` (LanceDB requires fixed-size vectors).
- Bulk-load every row from the latest Delta snapshot into a LanceDB table named `products_${ZEALT_RUN_ID}` inside `/app/lancedb_data`.
- Implement a function `historical_compare(delta_path: str, version_a: int, version_b: int) -> dict` that returns a dict with keys `added`, `removed`, `modified`, each being a sorted list of integer `id`s (semantics below) and that also writes the diff into a LanceDB table named `migration_audit_${ZEALT_RUN_ID}`.
- Provide a CLI entry point `run.py` that performs the full migration when executed with no arguments and writes both tables.

## Implementation Hints
- Use `from deltalake import DeltaTable` and `DeltaTable(path)` (optionally `.load_as_version(v)`) plus `.to_pyarrow_table()` to materialize each snapshot.
- The seeded vector column is variable-length `list<float32>`; convert it to `fixed_size_list<float32, 32>` (e.g. via a numpy roundtrip + `pyarrow.FixedSizeListArray.from_arrays`) before writing.
- Use `lancedb.connect("/app/lancedb_data")` and `db.create_table(name, data=..., mode="overwrite")`.
- Read the run id from the `ZEALT_RUN_ID` environment variable.
- Diff semantics for `historical_compare(path, a, b)` (compare snapshot at version `a` against snapshot at version `b`):
  - `added`: ids present in `b` but not in `a`.
  - `removed`: ids present in `a` but not in `b`.
  - `modified`: ids present in both versions whose `category` value differs between `a` and `b`.
- The `migration_audit_${ZEALT_RUN_ID}` table must have at least the columns `id` (int64), `change` (string, one of `added`/`removed`/`modified`), `version_a` (int64), `version_b` (int64). One row per affected id.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 run.py`
- After running the command, the directory `/app/lancedb_data` MUST contain:
  - Table `products_${ZEALT_RUN_ID}` with exactly the row count of the latest Delta snapshot and a schema whose vector column is `fixed_size_list<float32, 32>`.
  - Table `migration_audit_${ZEALT_RUN_ID}` populated by a call to `historical_compare('/app/delta_data/products', 0, 2)`.
- The Python module `solution.py` (importable from `/home/user/myproject`) MUST expose a callable `historical_compare(delta_path, version_a, version_b)` returning a dict with keys `added`, `removed`, `modified` (each a sorted list of ints).
- `ZEALT_RUN_ID` is read from the environment for all table names.

