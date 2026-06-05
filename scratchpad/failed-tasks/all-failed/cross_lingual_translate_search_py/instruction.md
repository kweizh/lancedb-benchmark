# Cross-Lingual Translate-Then-Search over an English LanceDB Corpus

## Background
You are building the retrieval layer for a multilingual help-desk that only owns an English knowledge base.
Incoming user queries arrive in Chinese (ZH), Japanese (JA) or Arabic (AR). The pipeline must translate every
query to English at runtime via the **real OpenAI Chat Completions API** (`gpt-4o-mini`, deterministic
`temperature=0`), then look up the translated text against an English-only LanceDB table whose rows have
already been embedded at build time with the **real OpenAI** `text-embedding-3-small` model.

A 40-row English corpus covering 10 distinct topics (4 documents per topic) is pre-seeded into LanceDB at
docker build time. The corpus path, table name prefix and embedding/chat models are exposed through
environment variables. The candidate must NOT touch the seed step; only the runtime search code must be
written.

## Requirements
- Implement a Python module `solution.py` exposing two top-level callables:
  - `translate_to_english(query: str, source_lang: str) -> str` — calls the OpenAI chat completions API
    (`gpt-4o-mini`) with `temperature=0` to translate `query` from `source_lang` (one of `"zh"`, `"ja"`,
    `"ar"`) into English. The function must return ONLY the translated English string, with no preamble,
    quotes, or explanation.
  - `cross_lingual_search(query: str, source_lang: str, k: int = 5) -> list[dict]` — translates `query`
    via `translate_to_english`, embeds the translated text via the **real** OpenAI embedding API
    (`text-embedding-3-small`), runs a vector search against the pre-seeded LanceDB table, and returns
    the top-`k` rows as a list of dicts. Each dict must include AT LEAST the keys `id` (int), `topic`
    (str) and `content` (str), preserving the LanceDB rank order (rank-1 first).
- Both functions must use the real hosted OpenAI APIs (no mocks, no local models, no canned responses).
- The LanceDB table to search is `${LANCEDB_TABLE_PREFIX}${ZEALT_RUN_ID}` under `${LANCEDB_URI}`.
  The table is already populated with 40 rows and a 1536-d `embedding` column at image build time;
  do NOT re-create or re-embed it.

## Implementation Hints
- Read the run id from the `ZEALT_RUN_ID` environment variable and append it to the table prefix to get the
  table name to open with `lancedb.connect(...).open_table(name)`.
- Use `openai.OpenAI()` (the client reads `OPENAI_API_KEY` from the environment). For translation,
  send a short system+user prompt that forces an English-only literal translation. For embeddings,
  use `client.embeddings.create(model=..., input=text)` and pass the resulting vector to
  `table.search(vec).limit(k).to_list()`.
- The chat and embedding model names are exposed via `OPENAI_CHAT_MODEL` and `OPENAI_EMBED_MODEL` env vars
  (defaults `gpt-4o-mini` and `text-embedding-3-small`). The chat call MUST use `temperature=0`.
- Do NOT add any other API calls beyond translation + embedding; in particular do not call the chat model to
  re-rank or summarize results.

## Acceptance Criteria
- Project path: /home/user/myproject
- Solution file: /home/user/myproject/solution.py
- The module exposes the two callables described above with the documented signatures.
- `translate_to_english(query, source_lang)` returns a non-empty English string and routes through the real
  OpenAI Chat Completions API using `gpt-4o-mini` with `temperature=0`.
- `cross_lingual_search(query, source_lang, k=5)` returns a list of up to `k` dicts, each containing the
  keys `id` (int), `topic` (str) and `content` (str), in descending similarity order.
- The LanceDB table is opened (not re-created) at `${LANCEDB_URI}` with name `${LANCEDB_TABLE_PREFIX}${ZEALT_RUN_ID}`
  where `ZEALT_RUN_ID` is read from the environment.
- For each of the three anchor queries below, the rank-1 result's `topic` matches the expected topic:
  - Chinese (`source_lang="zh"`) query about machine learning → topic `machine_learning`.
  - Japanese (`source_lang="ja"`) query about sushi recipes → topic `sushi_recipes`.
  - Arabic (`source_lang="ar"`) query about the Sahara desert → topic `sahara_desert`.

