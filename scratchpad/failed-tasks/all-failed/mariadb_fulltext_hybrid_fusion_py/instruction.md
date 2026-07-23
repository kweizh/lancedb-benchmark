# Hybrid Keyword + Vector Search with Reciprocal Rank Fusion

## Background
You are building a hybrid retrieval bridge that fuses two independent search engines over the SAME corpus:
- A local **MariaDB** server owns the document **text** and a **FULLTEXT** index (keyword relevance via `MATCH ... AGAINST`).
- A local **LanceDB** table owns **precomputed embedding vectors** (semantic similarity via vector search).

Both stores are already seeded and keyed by a shared integer `id`. Your job is to query both engines for a user query, then combine their two ranked result lists into a single ranking using **Reciprocal Rank Fusion (RRF)**.

## Requirements
- Implement `hybrid_search(query_text, query_vector, k)` that:
  - Runs a keyword search against the MariaDB FULLTEXT index for `query_text`.
  - Runs a semantic vector search against the LanceDB table for `query_vector`.
  - Computes each document's rank within each result list (best match = rank 1).
  - Fuses the two lists with Reciprocal Rank Fusion and returns the top-`k` documents.
- The fused result must surface documents that are strong in BOTH lists above documents that appear in only one list, while still including single-list matches.

## Implementation Hints
- Project path: `/home/user/myproject`. Implement your solution in `/home/user/myproject/solution.py` exposing a module-level function `hybrid_search(query_text: str, query_vector: list[float], k: int) -> list[dict]`.
- Connection details are provided via environment variables: `MARIADB_HOST`, `MARIADB_PORT`, `MARIADB_USER`, `MARIADB_PASSWORD`, `MARIADB_DATABASE`, and `LANCEDB_PATH`. The MariaDB server and LanceDB database are already running/seeded when your code runs.
- Both engines store the same corpus in a table named `docs_${ZEALT_RUN_ID}` — read `ZEALT_RUN_ID` from the environment and build the exact table name (same base name in both MariaDB and LanceDB).
- In MariaDB, the text lives in a column named `body` with a FULLTEXT index; use `MATCH(body) AGAINST (...)` (NATURAL LANGUAGE or BOOLEAN mode) and treat the returned order as the keyword ranking. In LanceDB, the vectors live in the `vector` column; use cosine distance for the semantic ranking.
- Use the standard Reciprocal Rank Fusion formula with the constant set to **60**: a document's fused score is the sum over each list it appears in of `1 / (60 + rank)`, where `rank` is its 1-based position in that list. A document missing from a list contributes nothing for that list.
- Return at most `k` objects, each with exactly the keys `id` (int) and `score` (float, the fused RRF score), sorted by `score` descending and ties broken by `id` ascending. The function must be deterministic: repeated calls with identical arguments return identical output.

