# Cohere Rerank Hybrid Search with Language Filter

## Background
You are building a retrieval pipeline on top of LanceDB. A local LanceDB database at `/home/user/cohere_rerank/lancedb` already contains a table named `docs` with 200 short documents. Each row has the following columns:

- `id` (`string`): unique document identifier.
- `content` (`string`): the document body (one to three short sentences in English, Spanish, or French).
- `embedding` (`fixed_size_list<float, 1536>`): a real OpenAI `text-embedding-3-small` embedding of `content`, computed once at build time.
- `language` (`string`): one of `en`, `es`, or `fr`.

Your job is to expose a hybrid search function that combines metadata filtering, vector + full-text search, and a real Cohere Rerank API call, then to make it runnable as a small CLI.

## Requirements
- Implement a Python module `hybrid_search.py` exposing a function `hybrid_search(query: str, language: str, k: int) -> list[dict]` that:
  1. Filters the `docs` table to only the rows whose `language` column equals the `language` argument, using a LanceDB SQL `where` clause.
  2. Embeds `query` once with OpenAI `text-embedding-3-small` (read `OPENAI_API_KEY` from the environment).
  3. Runs a LanceDB hybrid search (vector + full-text search using the BM25 FTS index that is already on `content`) and pulls **at least the top 30 candidates** that pass the language filter.
  4. Reranks those candidates with LanceDB's native `CohereReranker` (real Cohere Rerank API call using `COHERE_API_KEY`). The reranker must score against the `content` column.
  5. Returns the top-`k` reranked rows as a list of plain Python dicts with the keys `id` (`str`), `content` (`str`), `language` (`str`), and `rerank_score` (`float`). The list MUST be ordered by descending `rerank_score`.
- Build a CLI entrypoint `run_search.py` that:
  - Parses three CLI arguments: `--query` (string), `--language` (one of `en`/`es`/`fr`), and `-k` / `--k` (integer, default 5).
  - Calls `hybrid_search` and writes the result list as JSON (a JSON array) to stdout. Nothing else may be printed to stdout.

## Implementation Hints
- Open the existing LanceDB database at `/home/user/cohere_rerank/lancedb` and open the `docs` table — do NOT recreate the table or rebuild the FTS index. They are already in place.
- LanceDB ships a native `CohereReranker` integration; pick the correct import path yourself. The reranker requires a Cohere API key and runs against a text column (default `"text"`), so you will need to point it at this table's text column.
- The hybrid search builder must combine BOTH a vector query (the OpenAI embedding of `query`) AND a text query (the original `query` string).
- Order the final JSON list by descending rerank score so the most relevant document is first.
- Use the latest documentation: <https://docs.lancedb.com/integrations/reranking/cohere.md> and <https://docs.lancedb.com/search/hybrid-search.md>.

## Acceptance Criteria
- Project path: /home/user/cohere_rerank
- Command: `python3 run_search.py --query "<query>" --language <lang> -k <k>`
- Stdout MUST be a single JSON array. Each element MUST be an object with exactly the keys:
  - `id` (string)
  - `content` (string)
  - `language` (string, equal to the `--language` argument)
  - `rerank_score` (float)
- The array MUST have length `min(k, num_filtered_candidates)` and be sorted by descending `rerank_score`.
- Every returned row MUST satisfy `language == --language` (the metadata filter must be applied server-side via LanceDB's `where` clause, not in Python post-processing).
- The pipeline MUST issue a real Cohere Rerank API call using the `COHERE_API_KEY` environment variable (no offline / mocked rerankers).
- The query embedding MUST come from a real OpenAI `text-embedding-3-small` call using `OPENAI_API_KEY`.

