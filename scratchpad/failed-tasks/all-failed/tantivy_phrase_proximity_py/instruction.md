# LanceDB Native FTS Phrase Proximity & Boolean Search

## Background
LanceDB ships a native (Lance-based) full-text search engine that supports positional indexing. With positional indexing enabled, you can run phrase queries with a configurable slop (the maximum number of intervening tokens allowed between the phrase terms) and boolean queries that combine `MUST`/`MUST_NOT` clauses. Your task is to build a small search module that exposes these two query shapes on top of a LanceDB table seeded with documents about "machine learning".

## Requirements
- Open the LanceDB database at `/app/lancedb_data` and read the pre-seeded table `phrase_docs_${ZEALT_RUN_ID}` (read `ZEALT_RUN_ID` from the environment).
- The table is seeded for you with 50 rows and the schema `{id: int64, content: string}`. **Do not re-seed it.**
- Ensure a native (Lance) FTS index exists on the `content` column. The index MUST be built with positional information so that phrase queries with slop work. The index is provided pre-built by the entrypoint but your solution must be tolerant of either state: it is fine to call the index-creation API with the proper flags whenever the index is missing.
- Implement a module at `/home/user/myproject/solution.py` exposing two functions:
  - `phrase_search(words: list[str], slop: int, k: int = 10) -> list[int]` — Run a positional phrase query for the given ordered list of tokens with the given slop and return the matching document `id`s, ordered by descending relevance, limited to `k` results.
  - `boolean_must_search(must_terms: list[str], must_not_terms: list[str], k: int = 10) -> list[int]` — Run a boolean query that requires every term in `must_terms` and forbids every term in `must_not_terms`, returning matching `id`s, ordered by descending relevance, limited to `k`.

## Implementation Hints
- Connect with `lancedb.connect("/app/lancedb_data")` and `db.open_table(...)`.
- For phrase / boolean queries on the native FTS engine, look at `lancedb.query.PhraseQuery`, `lancedb.query.MatchQuery`, `lancedb.query.BooleanQuery`, and `lancedb.query.Occur`. Pass an instance of the query object straight into `table.search(query)`.
- When (re)creating the FTS index, the native engine needs to be told to keep token positions and to keep stop words in the index. Otherwise `slop` will not work and stop-word terms cannot be matched. Consult the [LanceDB FTS docs](https://docs.lancedb.com/search/full-text-search) for the exact keyword arguments.
- Results from `table.search(...).to_list()` come back as a list of dicts; project the `id` column and respect ordering.

## Acceptance Criteria
- Project path: /home/user/myproject
- The module `/home/user/myproject/solution.py` must define top-level callables `phrase_search` and `boolean_must_search` with the exact signatures above.
- Both functions must be importable without side effects (no top-level prints, no top-level mutation of the table).
- The LanceDB table name is `phrase_docs_${ZEALT_RUN_ID}` (read `ZEALT_RUN_ID` from the environment).
- `phrase_search` must use a positional phrase query so that `slop=0` returns only documents where the words appear consecutively in order, while larger `slop` values broaden matches to within that many intervening tokens.
- `boolean_must_search` must combine all `must_terms` (AND) and exclude any document containing any of the `must_not_terms`.
- Each function returns a `list[int]` of document ids, length ≤ `k`, ordered by descending relevance score returned by LanceDB.

