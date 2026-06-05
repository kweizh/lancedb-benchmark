# Multilingual Full-Text Search with Per-Language Tokenizers in LanceDB

## Background
You are extending a LanceDB-powered document search backend for a multilingual product help center. The same `content` text column needs to be indexed differently per language, because tokenization, stemming, and stop-word handling are language specific. English uses simple whitespace + English stemming, German uses German stemming, and Chinese has no word boundaries and must be pre-segmented before BM25 indexing.

The data is already loaded for you. A local LanceDB database lives at `/home/user/myproject/lancedb_data` and contains three sibling tables, each with 40 documents in the respective language:

- `docs_en_${ZEALT_RUN_ID}` (English documents)
- `docs_de_${ZEALT_RUN_ID}` (German documents)
- `docs_zh_${ZEALT_RUN_ID}` (Chinese documents, Simplified)

All three tables share the schema `{id: int64, content: string}`.

## Requirements
- Implement a Python module `solution.py` at `/home/user/myproject/solution.py` that:
  1. Builds an appropriate BM25 full-text-search index per language on the three tables.
  2. Exposes a function `search_per_lang(query: str, lang: str, k: int = 5) -> list[int]` that returns the **top-k `id` values, ordered by BM25 score (most relevant first)**, from the table corresponding to `lang`.
- Supported `lang` values are exactly `"en"`, `"de"`, and `"zh"`.
- The indexing step must be idempotent on re-import — if the module is imported a second time, it must not fail with "index already exists" or re-tokenize twice.
- The Chinese table cannot be searched correctly with the default tokenizer because Chinese text has no whitespace word boundaries. You must use `jieba` to pre-segment the Chinese text into whitespace-separated tokens before building the FTS index, and you must tokenize the query the same way at search time.
- The English and German tables must rely on LanceDB's native (Lance) FTS with the appropriate per-language stemming/stop-word configuration. Do **not** rely on Tantivy-only tokenizer options.
- Read the run id from the `ZEALT_RUN_ID` environment variable to resolve table names. The database directory and the table-name suffix must come from the environment — do not hard-code the run id.

## Implementation Hints
- LanceDB native FTS (`use_tantivy=False`) accepts a `language` argument that enables per-language stemming and stop-word lists. See [LanceDB Full-Text Search docs](https://docs.lancedb.com/search/full-text-search).
- LanceDB does not tokenize CJK text out of the box with the default tokenizer; pre-segmenting with `jieba.cut` and joining the resulting tokens with single spaces lets the default whitespace tokenizer work correctly.
- One reliable approach for Chinese is to add a derived column (e.g. `content_tokens`) populated with the jieba-segmented text and build the FTS index on that column. At search time, tokenize the query the same way and search the tokenized column.
- Building an FTS index that already exists raises an error in LanceDB 0.25.3; guard the index-creation step (e.g. by checking `table.list_indices()` first) so the module is safe to import repeatedly.
- Use `table.search(query, query_type="fts").limit(k).to_list()` for BM25-style ranked retrieval. Each result row will include the original `id`.
- All required dependencies (`lancedb==0.25.3`, `tantivy==0.22.0`, `jieba>=0.42`, `pyarrow`, `numpy`, `pandas`) are pre-installed in the environment.

## Acceptance Criteria
- Project path: /home/user/myproject
- Module path: /home/user/myproject/solution.py
- The module exposes a callable `search_per_lang(query: str, lang: str, k: int = 5) -> list[int]`.
- Importing the module from a fresh Python process must successfully create FTS indices on all three tables and must succeed on every subsequent import (idempotent).
- After import, each of the three tables (`docs_en_${ZEALT_RUN_ID}`, `docs_de_${ZEALT_RUN_ID}`, `docs_zh_${ZEALT_RUN_ID}`) reports at least one FTS index via `table.list_indices()`.
- `search_per_lang` always returns a Python `list[int]` of length `min(k, table_size)`, ordered by BM25 score descending.
- `search_per_lang` raises `ValueError` for unsupported `lang` values.
- The LanceDB database directory is `/home/user/myproject/lancedb_data`.

