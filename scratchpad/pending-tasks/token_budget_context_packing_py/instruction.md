# Token-Budget Context Packer with LanceDB

## Background
LLM RAG pipelines must fit retrieved context into a hard token budget while still maximising relevance and topical diversity. Build a packing retriever on top of a pre-seeded LanceDB `chunks` table that respects a per-call `max_tokens` budget and obeys a topic-diversity rule.

The container has already seeded a LanceDB table at `/app/lancedb_data/` containing 150 variable-length text chunks with real `text-embedding-3-small` embeddings (1536-d), a `topic_id` field, and a per-chunk `tokens` field counted with `tiktoken` (`cl100k_base`). The seeded table is named `chunks_${ZEALT_RUN_ID}` (the `run-id` is read from the `ZEALT_RUN_ID` environment variable).

## Requirements
Implement a Python module that:
- Exposes `retrieve_and_pack(query: str, max_tokens: int) -> dict` returning `{"chunks": list[dict], "total_tokens": int}`.
- Vector-searches the seeded LanceDB `chunks_${ZEALT_RUN_ID}` table for the query (cosine distance, real OpenAI `text-embedding-3-small`).
- Counts tokens per candidate via `tiktoken` (`cl100k_base`). The seeded `tokens` column already holds this value; the candidate may use it or recompute.
- Packs greedily by descending relevance until adding the next chunk would exceed `max_tokens`.
- Applies a topic-diversity rule: if the next chunk's `topic_id` already appears twice in the currently selected set, skip that chunk (do not stop) and continue to the next candidate.
- Returns each picked chunk as a dict containing the keys `id` (int), `text` (str), `topic_id` (int), `tokens` (int), `score` (float, the cosine distance from the search).
- `total_tokens` equals the sum of `tokens` across the returned `chunks` and is `<= max_tokens`.
- Degrades gracefully for very small budgets: when `max_tokens <= 50` the function returns at most one chunk (zero chunks if no chunk fits).

## Implementation Hints
- Read the `run-id` from the `ZEALT_RUN_ID` environment variable to construct the table name `chunks_${ZEALT_RUN_ID}`.
- Open the seeded LanceDB database at `/app/lancedb_data/` and search the chunks table with the query's embedding.
- Fetch enough candidates from LanceDB (use `.limit(...)` with a value comfortably larger than the expected output) before packing so the diversity rule has alternatives to fall back on.
- Walk candidates in descending relevance (ascending cosine distance) and maintain a per-topic counter while packing.
- Use a real OpenAI client (`OPENAI_API_KEY` is available at runtime) — do not local-mock the embedding model.

## Acceptance Criteria
- Project path: /home/user/myproject
- Module: `/home/user/myproject/solution.py` exposing `retrieve_and_pack(query: str, max_tokens: int) -> dict`.
- Return shape: `{"chunks": [{"id": int, "text": str, "topic_id": int, "tokens": int, "score": float}, ...], "total_tokens": int}`.
- `total_tokens` equals the integer sum of `tokens` over returned chunks and is `<= max_tokens` for every call.
- No `topic_id` appears more than 2 times in the returned `chunks` list.
- Chunks are returned in descending relevance order (ascending cosine distance).
- For any `max_tokens <= 50`, the function returns 0 or 1 chunks.
- The function reads from the LanceDB table named `chunks_${ZEALT_RUN_ID}` (where `${ZEALT_RUN_ID}` is read from the environment).
- The function uses the real OpenAI `text-embedding-3-small` model to embed queries (no local model, no mock).
- The function uses `tiktoken` with the `cl100k_base` encoding to count tokens.

