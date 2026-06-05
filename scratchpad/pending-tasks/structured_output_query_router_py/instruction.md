# LLM-Driven LanceDB Query Router with OpenAI Structured Outputs

## Background
You are building a multi-modal product search service over a LanceDB table named `products_${ZEALT_RUN_ID}`. The table is already pre-seeded by the container entrypoint with 500 product rows, real `text-embedding-3-small` 1536-d vectors on the `description` column, and a native Lance FTS index. Different end-user queries call for different retrieval strategies — some queries are best handled by vector similarity, some by keyword (FTS) search, some by hybrid vector + keyword, and some are pure SQL filter lookups.

Your job is to write a router that uses `gpt-4o-mini` (OpenAI Chat Completions API) with **structured outputs** (response_format = `json_schema`, `strict=true`) to classify each incoming query and extract any structured filters, then dispatch to the corresponding LanceDB query.

## Requirements

- Implement `route_and_search(query: str, k: int) -> dict` in `/home/user/myproject/solution.py`.
- The function must:
  1. Call `gpt-4o-mini` with an OpenAI structured-output `response_format` to classify the query into exactly one of `{"vector", "fts", "hybrid", "filter_only"}` AND extract optional structured filters (category, price range, date range).
  2. Execute the corresponding LanceDB query against the pre-seeded `products_${ZEALT_RUN_ID}` table and return up to `k` results.
- Return value must be a `dict` with exactly these top-level keys:
  - `mode` (str): one of `"vector"`, `"fts"`, `"hybrid"`, `"filter_only"`
  - `filters` (dict): the extracted structured filters (see schema in Implementation Hints)
  - `results` (list[dict]): up to `k` rows from the LanceDB table, ordered by relevance for the chosen mode; each row dict must contain at least the integer `id` field.

## Implementation Hints

- Read `ZEALT_RUN_ID` from the environment to build the table name (`products_${ZEALT_RUN_ID}`).
- Read `OPENAI_API_KEY` from the environment for both the routing LLM call and the embedding call. The same key is used for both the routing call (`gpt-4o-mini`) and the query-side embedding (`text-embedding-3-small`).
- Connect to LanceDB at `/home/user/myproject/data/lancedb`.
- The `products` table schema (already seeded for you) has columns: `id` (int64), `sku` (string), `name` (string), `description` (string), `category` (string), `price` (float64), `release_date` (string in `YYYY-MM-DD` form), `vector` (fixed_size_list<float32, 1536>). A native (non-Tantivy) FTS index is already built on the `description` column.
- Design a structured-output JSON schema with exactly two top-level fields. The JSON schema you register with the OpenAI `response_format` MUST have:
  - `mode`: `"string"` with `enum` `["vector", "fts", "hybrid", "filter_only"]`.
  - `filters`: object with these keys (all required by the schema, all nullable so the model can omit them by emitting `null`):
    - `category` (string|null) — the product category if explicitly mentioned (`electronics`, `clothing`, `home`, `sports`, `books`, `food`)
    - `price_min` (number|null) — lower bound (inclusive) when the user mentions a minimum or a range
    - `price_max` (number|null) — upper bound (inclusive) when the user mentions a maximum or a range. Wording such as "under $100" or "less than $100" maps to `price_max=100`.
    - `date_min` (string|null, `YYYY-MM-DD`) — release-date lower bound
    - `date_max` (string|null, `YYYY-MM-DD`) — release-date upper bound
  - Use `strict=true` and `additionalProperties=false` so that gpt-4o-mini reliably adheres to the schema.
- Translate the extracted filters into a single LanceDB SQL `where` clause (e.g. `price <= 100 AND category = 'clothing'`) and apply it on every mode that needs filtering. Skip the `where` if no filter field is populated.
- Mode dispatch:
  - `vector`: embed the user query with `text-embedding-3-small` and call `tbl.search(qvec).where(...).limit(k).to_list()`.
  - `fts`: call `tbl.search(query, query_type="fts").where(...).limit(k).to_list()`.
  - `hybrid`: embed the query and call `tbl.search(query_type="hybrid").vector(qvec).text(query).where(...).limit(k).to_list()` (default RRF reranker is fine).
  - `filter_only`: call `tbl.search().where(<sql>).limit(k).to_list()` — purely SQL-driven scan, no vector or FTS scoring.
- Design a clear system prompt that teaches gpt-4o-mini how to choose between the four modes. As guidance:
  - `filter_only` when the query is purely structural (e.g. `"products under $50"`, `"items in clothing category between $200 and $500"`) with no semantic or keyword payload.
  - `fts` when the user asks for a specific product code / SKU / unique identifier (e.g. `"find item ABC-123"`, `"search for code XYZ-789"`).
  - `vector` when the query is a free-form semantic description with no exact identifier and no structural filter (e.g. `"comfortable running shoes"`, `"wireless bluetooth earbuds"`).
  - `hybrid` when the query mixes semantic content with keyword OR with a price/date/category filter (e.g. `"red shoes under $100"`, `"premium leather wallet under $300"`).

## Acceptance Criteria
- Project path: `/home/user/myproject`
- Source file: `/home/user/myproject/solution.py` exposing a callable `route_and_search(query: str, k: int) -> dict`.
- Connection: open the table named `products_${ZEALT_RUN_ID}` at `/home/user/myproject/data/lancedb`. Do NOT recreate or re-seed the table — it is already populated by the container entrypoint.
- Structured output contract: every call MUST invoke `gpt-4o-mini` with an OpenAI `response_format` of type `json_schema` (or use the equivalent Python SDK helper) so the parsed `mode` is always one of `{"vector", "fts", "hybrid", "filter_only"}` and `filters` is a dict with the five keys above.
- Return value: `{"mode": <str>, "filters": <dict>, "results": <list of dict, each containing at least `id`>}`. `results` must be of length ≤ `k` and ordered by the chosen mode's relevance.
- The router must NOT mock the OpenAI API — it must hit the real `gpt-4o-mini` endpoint via the `openai` Python SDK.

