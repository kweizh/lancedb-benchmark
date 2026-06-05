# Ingest a Polars DataFrame into LanceDB via Arrow

## Background
You are building a small ingestion utility on top of LanceDB. The source dataset is held in memory as a [polars](https://docs.pola.rs/) `DataFrame` and must be persisted into a LanceDB table by going through Arrow (no per-row Python loops, no `pandas`). After ingestion, you expose a hybrid `search` function that combines a vector query with SQL `where` predicates.

## Requirements
In `/home/user/myproject/solution.py` implement the following:

1. `build_dataframe() -> polars.DataFrame`
   - Returns a polars DataFrame with **exactly 500 rows** generated deterministically with `numpy.random.default_rng(2026)`.
   - Columns and dtypes (in this exact order):
     - `id`: `pl.Int64`, values `0..499`.
     - `title`: `pl.Utf8`, values like `"item-<id>"`.
     - `score`: `pl.Float64`, drawn from `rng.uniform(0.0, 1.0, size=500)`.
     - `tag`: `pl.Utf8`, drawn from the closed list `["alpha", "beta", "gamma", "delta"]` using `rng.choice(...)` (in that order, after the score column was sampled).
     - `vector`: `pl.List(pl.Float32)`, one 32-dimensional list per row drawn from `rng.standard_normal((500, 32)).astype(np.float32)`.

2. `ingest(df: polars.DataFrame, db_uri: str, table_name: str) -> lancedb.table.Table`
   - Convert the polars DataFrame to a `pyarrow.Table` using `df.to_arrow()`.
   - Open a LanceDB connection at `db_uri` and create the table from that Arrow table (overwrite any pre-existing table).
   - The resulting Arrow schema must keep the original column order and the vector column must be a 32-wide list of float32.
   - Return the created `lancedb.table.Table`.

3. `search(table, vec, top_k=10, min_score=0.5, tag="alpha") -> list[dict]`
   - `vec` is a 32-dim `list[float]` / `numpy.ndarray`.
   - Run a vector search on the LanceDB table that also applies the SQL `where` clause `score >= <min_score> AND tag = '<tag>'`.
   - Return at most `top_k` matches as a list of dicts with the keys `id`, `title`, `score`, `tag`, and `_distance`, ordered by ascending `_distance`.
   - Every returned row MUST satisfy both filter predicates.

4. CLI entrypoint: `python3 solution.py` must:
   - Read `run_id = os.environ["ZEALT_RUN_ID"]`.
   - Use the LanceDB URI `/home/user/myproject/lancedb` and the table name `polars_ingest_${run_id}`.
   - Build the DataFrame, ingest it, then run a single demo search using `vec = numpy.zeros(32, dtype=numpy.float32)`, `top_k=10`, `min_score=0.5`, `tag="alpha"`.
   - Print the demo result as a single JSON line to stdout and append the same JSON line to `/home/user/myproject/output.log`.

## Implementation Hints
- Use `polars.DataFrame.to_arrow()` to obtain a `pyarrow.Table`. Do NOT round-trip through pandas. Do NOT iterate the rows in Python.
- LanceDB will accept the resulting `pa.Table` directly via `db.create_table(name, data=arrow_table, mode="overwrite")`. The polars `list[float32]` column becomes an Arrow `list<float32>` field — LanceDB treats it as a vector column for 32-d search.
- The vector dimensionality is fixed at 32. The `score` filter is a real-valued cutoff and the `tag` filter is an exact-match string.
- Combine vector search and SQL filtering with `table.search(vec).where("score >= ... AND tag = '...'").limit(top_k).to_list()`.
- Read the `run-id` from `ZEALT_RUN_ID` and append it to the table name so concurrent runs do not collide.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: python3 solution.py
- The script must read `ZEALT_RUN_ID` from the environment and use it as part of the LanceDB table name (`polars_ingest_${ZEALT_RUN_ID}`).
- The LanceDB database lives at `/home/user/myproject/lancedb`.
- After `python3 solution.py` finishes:
  - The LanceDB table `polars_ingest_${ZEALT_RUN_ID}` exists with 500 rows.
  - The Arrow schema preserves the column order `id, title, score, tag, vector` and the vector column is `list<float32>` with 32 elements per row.
  - `solution.search(...)` is importable as a Python function and returns rows that all satisfy `score >= min_score AND tag == tag`, ordered by ascending `_distance`, with at most `top_k` entries.
- `solution.py` prints the demo search result as a single JSON line to stdout and appends it to `/home/user/myproject/output.log`.

