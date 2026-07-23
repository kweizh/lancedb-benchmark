# Hybrid LanceDB + DuckDB Analytics Bridge

## Background
You are building a hybrid retrieval-and-analytics pipeline. Semantic candidates are retrieved from a local **LanceDB** table using vector search, and the retrieved candidates are then analyzed with **DuckDB** (embedded, local) SQL: predicate filters, a join against an auxiliary metadata table, a `GROUP BY` aggregation, and a window function. Everything runs locally with no network access.

Two input data files are provided in the environment:
- `/home/user/project/data/documents.jsonl` — one JSON object per line with the keys `id` (int), `title` (string), `category` (string), `price` (float), `in_stock` (bool), and `vector` (list of 8 floats).
- `/home/user/project/data/categories.csv` — columns `category`, `department`, `tax_rate`. Some category names contain single-quote/apostrophe characters (e.g. `Women's Apparel`).

## Requirements
- Ingest every row of `documents.jsonl` into a LanceDB table stored at `/home/user/project/lancedb`. The embedding/`vector` column must be a fixed-size list of 32-bit floats so that LanceDB vector search works against a float32 query vector.
- Build a rerunnable CLI that, given a query vector, retrieves the top-K nearest documents from LanceDB using L2 vector search (no metadata filtering at the LanceDB stage — the candidate pool is the raw nearest neighbors).
- Hand the retrieved candidates to DuckDB as the analytics engine. Inside DuckDB you must join the candidates to the auxiliary `categories` table on `category`, apply the predicate filters, run a per-department aggregation, and compute a per-department ranking window function. Predicate filtering and aggregation must happen in DuckDB, not in Python or LanceDB.

## Implementation Hints
- Use the LanceDB Python SDK to create/populate the table and PyArrow to control column typing; the vector column must be Arrow `float32` (a float64 list column is not a searchable vector column).
- Register/query the LanceDB result set and the CSV auxiliary table inside DuckDB and let DuckDB do the SQL work (`JOIN`, `WHERE`, `GROUP BY`, `ROW_NUMBER() OVER (...)`).
- Category values can contain single quotes; build your DuckDB SQL so such values are handled correctly (proper escaping or parameter binding) instead of breaking the query.
- Remember that rows added to a LanceDB table are not folded into indexes until you call `table.optimize()`.
- Project path: /home/user/project
- Command: `python3 run.py --query-vector <v> --top-k <k> --max-price <p> [--category <name>]`
  - `--query-vector`: comma-separated list of exactly 8 floats, e.g. `0.9,0.1,0.0,0.0,0.5,0.2,0.1,0.3`.
  - `--top-k`: integer size of the LanceDB nearest-neighbor candidate pool.
  - `--max-price`: float; only candidates with `price <= max_price` are kept.
  - `--category`: optional exact category filter (may contain apostrophes); when omitted, no category filter is applied.
- Retrieval + filtering semantics: fetch the top-K nearest documents from LanceDB by L2 distance (use the `_distance` value LanceDB returns). Then keep only candidates where `in_stock` is true AND `price <= max_price` AND (when `--category` is given) `category` equals that value.
- The CLI must print ONLY a single JSON object to stdout with exactly two keys:
  - `hits`: array of the surviving candidates ordered by ascending distance, breaking ties by ascending `id`. Each hit object must have exactly the keys `id` (int), `title` (string), `category` (string), `department` (string), `price` (float), `distance` (float, the LanceDB L2 `_distance`), and `dept_rank` (int). `dept_rank` is the 1-based rank of the hit within its department, ordered by ascending distance then ascending `id`, computed over the surviving hit set with a SQL window function.
  - `departments`: array with one object per department present in `hits`, ordered by ascending `department` name. Each object must have exactly the keys `department` (string), `num_docs` (int, count of surviving hits in that department), `avg_price` (float, mean price of those hits rounded to 4 decimals), and `min_distance` (float, minimum distance among those hits).

