# LanceDB Dataset Versioning Diff

## Background
LanceDB tables are versioned: every mutating operation (create, update, delete, add) produces a new version that can be retrieved later via `table.checkout(version)`. A LanceDB table named `customers` has been pre-seeded at `/data/lancedb` with four versions describing the evolution of a small customer catalog.

## Requirements
Implement a Python module at `/home/user/myproject/solution.py` that exposes a single function:

```python
def diff_versions(v_a: int, v_b: int) -> dict:
    ...
```

Given two integer version numbers from the same `customers` table, the function must return a JSON-serializable dictionary describing how the dataset changed between `v_a` and `v_b`. Rows are identified by the `id` column.

## Implementation Hints
- Open the LanceDB database at `/data/lancedb` and open the `customers` table.
- Use the table's versioning APIs to obtain the row set at each version, then compare the two row sets by primary key.
- You decide whether to use pandas joins, Arrow compute, or pure-Python set logic; only the returned dictionary is observed.
- The schema has four columns: `id` (int64), `name` (string), `price` (float64), `category` (string). Only the non-id columns participate in the modified-row comparison.
- Treat numeric values as floats; small floating-point drift on equal values is tolerated by the verifier.

## Acceptance Criteria
- Project path: `/home/user/myproject`
- File: `/home/user/myproject/solution.py`
- Exported callable: `diff_versions(v_a: int, v_b: int) -> dict`
- LanceDB database path: `/data/lancedb`
- Table name: `customers`
- The returned dictionary must contain exactly these three top-level keys:
  - `added`: list of integer `id`s present in `v_b` but absent from `v_a`.
  - `removed`: list of integer `id`s present in `v_a` but absent from `v_b`.
  - `modified`: list of objects describing rows present in both versions whose non-id columns differ. Each object has the shape `{"id": <int>, "old": {<col>: <value>, ...}, "new": {<col>: <value>, ...}}`, where `old` carries the values from `v_a` and `new` carries the values from `v_b`. Both `old` and `new` must include all non-id columns of the row.
- Ordering of the lists does not matter; the verifier sorts them by `id` before comparing.
- The function must not modify the table on disk. Multiple invocations with the same arguments must return equal results.

