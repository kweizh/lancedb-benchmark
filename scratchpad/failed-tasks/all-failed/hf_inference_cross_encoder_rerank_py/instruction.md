# Two-Stage Retrieve + Cross-Encoder Rerank with LanceDB and Hugging Face Inference API

## Background

You are extending a small retrieval pipeline. A LanceDB database is already seeded at `/home/user/myproject/lancedb` with a table named `docs` that has 200 rows. Each row has the following columns:

- `id` (`string`): unique document identifier.
- `content` (`string`): the document body (one or two short English sentences).
- `embedding` (`fixed_size_list<float, 1536>`): a real OpenAI `text-embedding-3-small` embedding of `content`, precomputed once at image-build time.

The current code base only does plain vector search, which is good but not great: in particular it confuses keyword-stuffed documents with documents that actually answer a question. Your job is to add a second-stage cross-encoder reranker on top, using the **Hugging Face Inference API** for a hosted cross-encoder model.

## Requirements

Implement a Python module `solution.py` exposing a function with this exact signature:

```python
def rerank_search(query: str, k: int = 10) -> list[dict]: ...
```

`rerank_search` must implement a two-stage pipeline:

1. **Stage 1 — vector retrieval:** open the existing LanceDB table `docs` at `/home/user/myproject/lancedb`, embed `query` once with real OpenAI `text-embedding-3-small` (reading `OPENAI_API_KEY` from the environment), and pull the **top 30** rows by ascending vector distance.
2. **Stage 2 — cross-encoder rerank:** POST the 30 `(query, doc_content)` pairs to the Hugging Face Inference API endpoint `https://api-inference.huggingface.co/models/<reranker>` for a hosted cross-encoder model (use either `BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2`). Authenticate with the `HF_TOKEN` environment variable as a Bearer token. Parse the returned cross-encoder relevance scores and pick the top `k` rows by descending cross-encoder score.

Return a list of plain Python dicts, each with exactly these keys:

- `id` (str)
- `content` (str)
- `rerank_score` (float, the cross-encoder relevance score)

The list MUST be sorted by descending `rerank_score` and MUST have length `min(k, 30)`.

## Implementation Hints

- The LanceDB database, the `docs` table, the embeddings, and the FTS-free schema are already prepared at image-build time. Do **NOT** rebuild the table or recompute the embeddings.
- The Hugging Face Inference API expects a JSON POST body for cross-encoder / reranker models. Consult the model card you choose to determine the exact request body shape and how the response encodes per-pair scores; both `BAAI/bge-reranker-base` and `cross-encoder/ms-marco-MiniLM-L-6-v2` accept `[query, document]` text pairs and return one numeric score per pair.
- Read API credentials from environment variables `OPENAI_API_KEY` (for the query embedding) and `HF_TOKEN` (for the rerank call). Do not hardcode any keys.
- Use `lancedb==0.25.3` and `openai==1.54.5` for stage 1. The image already pins `httpx==0.27.2` so the OpenAI client constructs cleanly.
- Higher cross-encoder score MUST mean more relevant. If you pick a model whose raw output is a logit, you may pass it through a sigmoid or leave it as-is — relative ordering is what matters.

## Acceptance Criteria

- Project path: /home/user/myproject
- File: `/home/user/myproject/solution.py` exposing `rerank_search(query: str, k: int = 10) -> list[dict]`.
- For any call with `k <= 30`, the returned list MUST have length `k` (assuming 30 candidates exist in stage 1, which is guaranteed by the 200-row seed).
- Each returned element MUST be a dict with exactly the keys `id` (str), `content` (str), `rerank_score` (float). No additional keys are allowed.
- The list MUST be sorted by descending `rerank_score`.
- The query embedding MUST come from a real OpenAI `text-embedding-3-small` call using `OPENAI_API_KEY` (no offline embeddings, no cached vectors).
- The reranker MUST issue a real HTTPS POST to `https://api-inference.huggingface.co/models/<reranker>` using `HF_TOKEN` (no offline / mocked rerankers, no local model inference).
- For the anchor query `"what command undoes the most recent git commit"`, `rerank_search(query, k=10)[0]["id"]` MUST equal the rigged ground-truth document id `"rigged-correct"`. The vector-only top-1 for that query is intentionally a keyword-stuffed distractor (id `"rigged-distractor"`); only the cross-encoder rerank can recover the right answer.

