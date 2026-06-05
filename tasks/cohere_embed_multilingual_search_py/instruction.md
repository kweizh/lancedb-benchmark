# Cross-Lingual Semantic Search with Cohere Multilingual Embeddings and LanceDB

## Background

LanceDB is a multimodal vector database built on the Lance columnar format. In this task you must build a small **cross-lingual** retrieval pipeline that can answer a query in one language with documents written in any of three languages (English, Spanish, French) using **Cohere's `embed-multilingual-v3.0`** embedding model.

A fixed corpus is provided at build time at `/home/user/myproject/corpus.json`. It contains exactly **90 rows**: 30 matched triples, one English / one Spanish / one French sentence per real-world concept (e.g. "The Eiffel Tower is in Paris" plus its Spanish and French translations). Each row has the following fields:

```
{"concept_id": <int 0..29>, "language": "en"|"es"|"fr", "text": "<sentence>"}
```

The three sentences sharing the same `concept_id` are translations of the same concept.

## Requirements

Implement a Python module that:

1. Reads `corpus.json` and uses the official `cohere` Python SDK with the real key in the `COHERE_API_KEY` environment variable to embed every text via `co.embed(texts=..., model="embed-multilingual-v3.0", input_type="search_document")`. The model produces **1024-dimensional** vectors.
2. Persists all 90 rows into a single LanceDB table (under `lancedb_data/`) with at minimum the columns `concept_id` (int), `language` (string), `text` (string), and a 1024-d `vector` (or `embedding`) column of `float32`. Use the run-scoped table name `multilingual_${ZEALT_RUN_ID}` (read `ZEALT_RUN_ID` from the environment) so concurrent runs do not collide.
3. Exposes a function `cross_lingual_search(query: str, k: int = 3) -> list[dict]` that embeds the query with `input_type="search_query"` and returns the top-`k` results across **all three languages** (no language filter), sorted by ascending distance. Each result dict must include the keys `concept_id`, `language`, and `text`.

## Implementation Hints

- Install both `lancedb` and the official `cohere` Python SDK (latest 5.x).
- `co.embed(...)` may return embeddings either as `response.embeddings` (legacy list-of-lists) or `response.embeddings.float_` / `.float` (when `embedding_types=["float"]` is set). Pick whichever form your installed SDK supports and convert to a Python list of `float32`-castable lists before storing.
- LanceDB needs a fixed-size list type for the vector column; declare it explicitly (e.g. via `pyarrow.list_(pa.float32(), 1024)`) or use a Pydantic `LanceModel` with `Vector(1024)`.
- The function may build the index lazily on first call OR you may also expose a `build_index()` helper that does the embedding+ingestion once. The verifier will call `build_index()` first if it exists, then `cross_lingual_search`.
- The query must use `input_type="search_query"`; the documents must use `input_type="search_document"`. Mixing them up will silently degrade cross-lingual accuracy.
- Cohere's `embed-multilingual-v3.0` returns vectors that are already L2-normalized — cosine and L2 produce the same ordering, but use `metric="cosine"` or LanceDB's default for safety.

## Acceptance Criteria

- Project path: `/home/user/myproject`
- A Python module `solution.py` exists at `/home/user/myproject/solution.py` exposing `cross_lingual_search(query: str, k: int = 3) -> list[dict]`. The module may optionally expose a no-argument `build_index()` helper.
- After invoking the candidate's setup logic (either `build_index()` or implicitly on first `cross_lingual_search` call), the LanceDB table named `multilingual_${ZEALT_RUN_ID}` under `/home/user/myproject/lancedb_data/` contains exactly 90 rows with the schema described above and a 1024-dimensional vector column.
- Each entry returned by `cross_lingual_search` is a dict containing at least the keys `concept_id` (int), `language` (one of `"en"`, `"es"`, `"fr"`), and `text` (str). Results are sorted from closest to farthest.
- For an English query about a single concept, the top-3 must (a) contain the corresponding `concept_id` and (b) collectively span **at least two different languages**.
- For a Spanish query about a single concept, the same two conditions must hold.
- No language filter is applied: results from all three languages are eligible.
- The candidate's code must read `COHERE_API_KEY` from the environment and must not hardcode credentials.

