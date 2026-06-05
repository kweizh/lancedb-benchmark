# Hive-Partitioned Parquet → LanceDB With Year-Filtered Search

## Background
A Hive-style partitioned Parquet dataset has been pre-generated at `/home/user/myproject/parquet_dataset/` with one directory per `year` partition (`year=2022/`, `year=2023/`, `year=2024/`). Each partition contains 200 rows. Every row has the columns:

- `id` (`int64`): row identifier, unique within the whole dataset.
- `title` (`string`): short document title.
- `embedding` (`fixed_size_list<float, 24>`): a 24-dimensional vector, generated deterministically at build time.

The `year` column is **encoded only as the Hive partition directory name** and is NOT a physical column in the per-partition Parquet files. Your job is to ingest this partitioned dataset into a fresh LanceDB table while preserving the `year` column, and to expose a vector search function that filters on `year` server-side.

## Requirements
- Implement a Python module `solution.py` (importable as `solution`) at `/home/user/myproject/solution.py` that, when imported, does **all** of the following in `/home/user/myproject/lancedb`:
  1. Opens the partitioned Parquet dataset with `pyarrow.dataset` and `partitioning="hive"` so that the `year` partition column is materialized as a real column in the produced batches.
  2. Streams batches from that dataset into a new LanceDB table named `articles_${ZEALT_RUN_ID}` (read `ZEALT_RUN_ID` from the environment). The destination table MUST contain the `year` column (typed as a 32-bit or 64-bit integer) alongside `id`, `title`, and `embedding`.
  3. The ingest must be idempotent: re-importing `solution` (or constructing a fresh connection) MUST NOT duplicate rows. If the destination table already exists with the expected 600-row content, leave it as is; otherwise (re)create it.
- Expose a top-level callable `search_year(vec, year, k=5) -> list[dict]` in `solution.py` that:
  - Accepts a Python list / numpy array `vec` of length 24, an integer `year`, and an integer `k`.
  - Runs a single LanceDB query that combines a vector search on `embedding` with a SQL `where` clause restricting the result to the requested `year`. The year filter MUST be applied server-side via LanceDB's `where` clause (no Python post-filtering of an unfiltered result).
  - Returns a list of up to `k` plain Python dicts, ordered by ascending vector distance. Each dict MUST contain at least the keys `id` (int), `title` (str), and `year` (int).

## Implementation Hints
- `pyarrow.dataset.write_dataset` writes Hive-partitioned datasets where the partition column is encoded only in the directory name. To get the `year` column back as a real Arrow column, you MUST open the dataset with `pyarrow.dataset.dataset(path, partitioning="hive")` and then iterate `to_batches()` (or call `to_table()`).
- Use `db.create_table(name, schema=..., mode="overwrite")` and then `table.add(batch)` per batch, or pass an iterator/reader directly to `create_table` — pick whichever you prefer, as long as every row from every partition lands in the destination table exactly once.
- For the year-filtered vector search, the LanceDB query builder accepts `.where("year = <int>")` and `.limit(k)`; convert the result to a list of dicts (e.g. via `.to_list()` and trimming columns).
- Build the destination table name at runtime from `os.environ["ZEALT_RUN_ID"]` so that parallel runs do not clash.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 -c "import solution; print(len(solution.search_year([0.0]*24, 2023, 5)))"`
- Destination LanceDB database: `/home/user/myproject/lancedb`
- Destination table name: `articles_${ZEALT_RUN_ID}` (read `ZEALT_RUN_ID` from the environment).
- The destination table MUST contain exactly 600 rows: 200 rows for each of `year ∈ {2022, 2023, 2024}` (`SELECT year, COUNT(*) GROUP BY year`).
- The destination table schema MUST include columns `id`, `title`, `embedding`, and `year`. The `embedding` column MUST be a 24-dimensional fixed-size list of float32. The `year` column MUST be an integer type.
- `solution.search_year(vec, year, k)` MUST:
  - Return a `list` of at most `k` dicts.
  - Every returned row MUST satisfy `row["year"] == year` (verifier will fail if any other year appears).
  - The returned rows MUST be ordered by ascending vector distance against `vec` within the requested year partition (i.e. equivalent to a pure vector search restricted to that year).
  - The year filter MUST be enforced server-side by LanceDB (the SQL `where` clause), not by Python post-processing of an unfiltered result.
- Importing `solution` multiple times MUST NOT duplicate rows in the destination table.

