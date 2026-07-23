# Dual-Store Search: SQLite FTS5 BM25 + LanceDB Vectors Fused with Weighted RRF

## Background
You are building the retrieval core of a hybrid search engine. The **same corpus** of documents lives in two co-located, purely local stores:

1. A **SQLite FTS5** virtual table that provides classic **BM25 keyword ranking** through the built-in `bm25()` auxiliary function (with per-column weights for `title` vs `body`).
2. A **LanceDB** table that provides **semantic vector ranking** over the same documents.

Both stores have already been seeded for you (identical document ids in both). Your job is to implement a fusion layer that pulls ranked candidates from each store and blends them with a **weighted Reciprocal Rank Fusion (RRF)**, exposing a knob to tune how much keyword vs semantic evidence matters. Everything runs in-process on the local machine — there is **no network, no hosted API, and no model download**.

## Requirements
- Implement `search(query, query_vector, k, keyword_weight, title_weight=1.0, body_weight=1.0)` in `solution.py`.
  - `query` (str): the keyword query issued to the SQLite FTS5 store via `MATCH`.
  - `query_vector` (list[float]): the semantic query embedding issued to LanceDB.
  - `k` (int): number of fused results to return.
  - `keyword_weight` (float in `[0, 1]`): blend factor between the keyword store and the vector store.
  - `title_weight`, `body_weight` (float): per-column BM25 weights passed to the FTS5 `bm25()` function.
- Pull a **generous candidate pool of at least 50 ranked candidates** from each store before fusing (fewer only if the store returns fewer).
- Keyword candidates come from the FTS5 table ordered by `bm25()` (remember FTS5 `bm25()` returns **more-negative = more relevant**, so best match is rank 1).
- Vector candidates come from LanceDB ordered by ascending distance (closest = rank 1).
- Fuse the two ranked lists with weighted RRF and return the top `k`.

## Implementation Hints
- Project path: `/home/user/myproject`
- Implement everything in `/home/user/myproject/solution.py` exposing a module-level function `search(...)` with the signature above.
- **Discover the stores from the config file** written by the seeder at `/app/data/config.json`. It contains exactly these keys: `sqlite_path`, `lancedb_uri`, `table_name`, `vector_dim`, `distance_metric`, and `rrf_k`. Both the SQLite FTS5 virtual table and the LanceDB table share the same `table_name` (suffixed with the run id for isolation). Do **not** hard-code table names or the RRF constant.
- Use Python's standard-library `sqlite3` to talk to the FTS5 table; the query shape is `... WHERE <table> MATCH ? ORDER BY bm25(<table>, <title_weight>, <body_weight>)`. The FTS5 table has columns `(title, body)` and its `rowid` equals the document id.
- Use `lancedb` for the vector side and honor the `distance_metric` from the config (e.g. `.distance_type(...)`).
- **Ranks are 1-based** within each store's returned candidate list.
- **Weighted RRF formula (use exactly this):** for each document `d`,
  `fused_score(d) = keyword_weight * (1.0 / (rrf_k + bm25_rank(d))) + (1.0 - keyword_weight) * (1.0 / (rrf_k + vector_rank(d)))`
  where a document that is absent from a store contributes `0` for that store's term. Use the `rrf_k` value from `config.json`.
- Rank the fused candidates by `fused_score` **descending**, breaking ties by **document id ascending**, and return the top `k`.
- Return a `list[dict]`, each dict having exactly the keys `id` (int) and `score` (float, the fused score), ordered from best to worst, with length `min(k, number_of_candidates)`.
- Results must be **fully deterministic**: identical arguments must always yield identical output (apply the id tie-break explicitly rather than relying on store row order).

