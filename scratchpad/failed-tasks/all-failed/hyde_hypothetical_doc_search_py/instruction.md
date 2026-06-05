# HyDE (Hypothetical Document Embeddings) Retrieval with LanceDB

## Background
Short, ambiguous user questions are a known weak point for plain semantic retrieval: an acronym-laden question like `GC differences between languages?` is often closer (in embedding space) to documents that share the surface tokens than to documents that actually answer the question. **HyDE** (Gao et al., 2022) addresses this by asking an LLM to write a hypothetical answer first, embedding that *answer*, and using THAT vector as the search query. The hypothetical document acts as a query expansion that lives in the same semantic space as the corpus.

In this task you will implement HyDE on top of LanceDB. A pre-seeded corpus of 30 long, detailed paragraphs about programming languages is baked into the Docker image at build time using real OpenAI `text-embedding-3-small` (1536-d cosine). Your job is to wire up the LLM-based query expansion and the comparison baseline.

## Requirements
Implement three functions in `/home/user/myproject/solution.py`:

1. `generate_hypothetical(query: str) -> str` — Calls real OpenAI `gpt-4o-mini` with `temperature=0` and an instruction asking for a detailed paragraph that answers the question. Returns the generated answer as a plain string.
2. `hyde_search(query: str, k: int = 5) -> list[int]` — Calls `generate_hypothetical(query)`, embeds the result with real OpenAI `text-embedding-3-small`, then runs LanceDB cosine top-k against the seeded corpus and returns the list of `id` integers (rank-ordered, length `k`).
3. `baseline_search(query: str, k: int = 5) -> list[int]` — Embeds the **raw** query directly with `text-embedding-3-small` and returns LanceDB cosine top-k `id` integers (no HyDE expansion). This is the comparison baseline.

## Implementation Hints
- The corpus is already in `/app/lancedb_data/` as a LanceDB table containing `{id int64, language string, text string, vector fixed_size_list<float32, 1536>}`. Connect with `lancedb.connect("/app/lancedb_data")` and open the table named `programming_qa`. Do NOT recreate or reseed it.
- Read `OPENAI_API_KEY` from the environment for both the chat completion and embedding calls.
- Use cosine similarity (the table is configured for cosine). LanceDB's `table.search(vector).limit(k)` is sufficient — you do not need to set `distance_type` explicitly.
- The OpenAI chat call must use `temperature=0`; the system prompt should ask for a detailed paragraph answering the question (e.g., `Please write a detailed paragraph answering this question:`).
- Return plain Python `int` IDs (not numpy scalars).

## Acceptance Criteria
- Project path: `/home/user/myproject`
- Command: `python3 -c "from solution import hyde_search, baseline_search, generate_hypothetical; print(hyde_search('GC differences between languages?'))"`
- `solution.py` must expose exactly the three functions named above with the signatures specified.
- `hyde_search(query, k)` and `baseline_search(query, k)` must each return a `list[int]` of length `k` (Python ints, no duplicates).
- For the deliberately ambiguous short query `GC differences between languages?` with `k=5`, the HyDE result's top-1 id MUST match the build-time "rigged" garbage-collection-essay document, while the baseline result's top-1 id MUST be different (i.e., HyDE must recover the correct document that the baseline misses).
- `generate_hypothetical(query)` must return a non-empty string of at least 100 characters produced by `gpt-4o-mini` at `temperature=0`.
- Solution must not mutate or rewrite the pre-seeded `programming_qa` table.

