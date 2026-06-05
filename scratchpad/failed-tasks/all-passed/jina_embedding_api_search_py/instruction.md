# News Headline Semantic Search with Jina Embeddings v3 + LanceDB

## Background
Build a multilingual-quality semantic search system over a fixed corpus of news headlines. Embeddings are produced by the hosted Jina AI Embeddings API (`jina-embeddings-v3` model) which exposes asymmetric retrieval via the `task` parameter — passages should be embedded with `retrieval.passage`, queries with `retrieval.query`. Embeddings are stored and searched in a local LanceDB table.

## Requirements
- A deterministic fixture of 50 news headlines (5 topics × 10 each) is baked into the image at `/home/user/myproject/headlines.json`. Each entry has fields `id` (int), `topic` (str), and `headline` (str). Do not modify this file.
- Implement a Python module `/home/user/myproject/solution.py` that:
  - Reads `JINA_API_KEY` from the environment.
  - Provides `build_index() -> None` which:
    - Embeds every headline in `headlines.json` using the Jina API with `task="retrieval.passage"`.
    - Creates / overwrites a LanceDB table named `headlines_${ZEALT_RUN_ID}` under `/home/user/myproject/lancedb_data/` whose schema includes at least `id` (int64), `headline` (str), `topic` (str), and a fixed-size float vector column matching the API's embedding dimension.
  - Provides `search(query: str, k: int = 5, task: str = "retrieval.query") -> list[dict]` which:
    - Embeds `query` via Jina with the supplied `task` parameter (defaulting to `retrieval.query`).
    - Runs a vector search over the LanceDB table and returns a Python list with **exactly `k`** dicts, ordered best-first. Each dict must contain at least the keys `id`, `headline`, `topic`.
- Provide a CLI entrypoint `/home/user/myproject/run.py` such that running `python3 run.py "<query>" --k <int>` prints the list returned by `search(...)` as a single JSON array on stdout. Running `python3 run.py --build` rebuilds the index.
- The LanceDB table name MUST be suffixed with the current `run-id` read from the `ZEALT_RUN_ID` environment variable so concurrent runs don't collide.

## Implementation Hints
- The Jina embeddings endpoint is `POST https://api.jina.ai/v1/embeddings`. Authenticate with `Authorization: Bearer ${JINA_API_KEY}` and `Content-Type: application/json`. The request body shape is `{"model": "jina-embeddings-v3", "task": <task>, "input": [<strings>]}` — verify the response shape and embedding dimension at runtime against the docs at `docs.jina.ai`. Use `httpx.post(...)` rather than `requests`.
- The Jina API accepts a list of input strings in one call. Batch all 50 headlines in a small number of calls (or a single call) to keep latency low.
- LanceDB tables prefer a `pyarrow.fixed_size_list(pa.float32(), <dim>)` column for embeddings. The vector dimension MUST match the dimension returned by the API.
- Use `tbl.search(query_embedding).limit(k).to_list()` to retrieve top-k results.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 run.py "<query>" --k <int>`
- Stdout: a single JSON array of length `k`. Each element MUST have keys `id` (int), `headline` (str), `topic` (str).
- Module API: `solution.build_index()` and `solution.search(query, k, task)` are importable from `/home/user/myproject`.
- The LanceDB directory `/home/user/myproject/lancedb_data/` exists after building and contains a table named `headlines_${ZEALT_RUN_ID}` where `${ZEALT_RUN_ID}` is read from the `ZEALT_RUN_ID` environment variable.
- Headline corpus comes from `/home/user/myproject/headlines.json` (50 entries, 5 topics × 10) and MUST NOT be modified.
- Queries are embedded with `task="retrieval.query"` (Jina asymmetric retrieval) by default; passages with `task="retrieval.passage"`.

