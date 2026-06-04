# Query Plan Introspection with LanceDB

## Background
LanceDB exposes `explain_plan()` and `analyze_plan()` on its query builders so engineers can audit how vector, scalar-filtered, and hybrid (vector + FTS) searches are physically executed. Your job is to build a small Python utility that inspects three representative queries against an existing LanceDB table and emits a structured report of the operator chain plus the runtime metrics LanceDB reports for each.

A pre-seeded LanceDB database is provided. It already contains a single table with at least 256 deterministic rows, a 16-dimensional vector column, a text column with a native LanceDB FTS index, and a `category` scalar column. A fixed numpy query vector (seed=2026) is also written to disk.

## Requirements
Write `solution.py` so that running `python3 solution.py` produces a JSON report at `report.json` describing the physical plans for three queries.

1. **Plain vector search**: top-10 nearest neighbours of the provided query vector — no filter, no FTS.
2. **Prefiltered vector search**: same query vector, top-10, restricted by a SQL `where` clause on the `category` column (use the value provided via the `CATEGORY_FILTER` environment variable). The filter must be pushed in as a prefilter so the resulting physical plan contains a Filter / Projection / Take operator chain.
3. **Hybrid (vector + FTS) search**: top-10 results using both the query vector and the keyword in the `FTS_QUERY` environment variable, executed as `query_type="hybrid"`.

For each of the three queries the report must contain:
- The full string returned by `explain_plan()` (verbose plan).
- The full string returned by `analyze_plan()`.
- A list of operator class names parsed from the explain plan tree (e.g. `ProjectionExec`, `Take`, `FilterExec`, `KNNVectorDistance`, `MatchQuery`, etc.) — preserve the order they appear in.
- An integer `top_output_rows` extracted from the root `analyze_plan` operator's `output_rows=N` metric.

## Implementation Hints
- Open the existing database at the path supplied by the `LANCEDB_URI` environment variable and open the table named in `TABLE_NAME`.
- Load the query vector from `query_vector.npy`.
- Use the LanceDB sync Python SDK; `explain_plan()` / `analyze_plan()` are methods on the query builder returned by `table.search(...)`.
- For the hybrid query, build the query with `table.search(query_type="hybrid").vector(qvec).text(fts_query).limit(10)`.
- Parse operator names with a regex on the leading token before `:` on each non-empty line of the explain plan output.
- Extract `output_rows=N` from the first metrics block in the analyze plan.
- Write the report with `json.dump(..., indent=2)`.

## Acceptance Criteria
- Project path: /home/user/myproject
- Ensure `solution.py` is actually executed and produces the artifact below.
- Log / artifact file: /home/user/myproject/report.json
- The JSON report MUST be an object with the top-level keys `plain`, `prefilter`, and `hybrid`.
- Each section MUST contain the keys `explain_plan` (string), `analyze_plan` (string), `operators` (list of strings), and `top_output_rows` (integer).
- The `prefilter.operators` list MUST contain at least one entry whose name starts with `Filter`, at least one entry containing `Projection`, and at least one entry containing `Take` (matching `Take` or `RemoteTake`).
- The `plain.top_output_rows` and `prefilter.top_output_rows` values MUST equal `10` (the requested limit).
- The `hybrid.explain_plan` text MUST mention both a vector sub-plan and an FTS sub-plan (substrings `Vector` and `FTS`, case-insensitive).
- The `hybrid.operators` list MUST be non-empty.

