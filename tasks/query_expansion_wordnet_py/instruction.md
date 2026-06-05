# Query Expansion with WordNet and LanceDB Native FTS

## Background
Lexical search (BM25 / full-text search) is brittle: a query for `car` will not find documents that mention `automobile` or `motorcar` instead, even though they describe the same concept. A classic, well-understood remedy is **query expansion**: each query term is enriched with its synonyms before the lexical engine runs. In this task you will combine two off-the-shelf components:

- **WordNet** (the static lexical database shipped with `nltk`) as the synonym source.
- **LanceDB native FTS** (`use_tantivy=False`) as the lexical search engine.

The corpus is a fixed set of 100 short English documents that has been seeded into a LanceDB table for you at container build time. The corpus is hand-crafted so that several documents use words like `automobile`, `motorcar`, or `vehicle` but never mention `car`, while a smaller set uses `car` explicitly. A plain FTS search for `car` therefore misses the synonym-only documents — your expanded pipeline must recover them.

## Requirements
- Implement a Python module at `/home/user/myproject/solution.py` that exposes a callable `expanded_search(query: str, k: int = 10) -> list[int]`.
- Open the pre-seeded LanceDB database at the path provided in the `LANCEDB_URI` environment variable (a directory under `/app/lancedb_data`) and use the table whose name is provided in the `LANCEDB_TABLE` environment variable.
- Build a **native** LanceDB FTS index on the `content` column (`use_tantivy=False`). The index must be created if it does not yet exist; calling `expanded_search` more than once must not error.
- For each whitespace-separated token in `query`, look up WordNet synonyms (lowercased), keep at most **3** synonyms per token, and only keep synonyms that are a single token (i.e. no `_` or whitespace in the WordNet lemma name). The original query term must also remain in the expansion.
- Construct an `OR`/`SHOULD` boolean FTS query over the expansion (you may use a space-separated terms query, `MatchQuery(...) | MatchQuery(...)`, or `BooleanQuery([(Occur.SHOULD, ...), ...])`).
- Return the **top-k document IDs**, ordered by FTS score descending, as a list of Python `int`.

## Implementation Hints
- Install the WordNet and OMW-1.4 corpora at build time with `python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"`. WordNet is a lexical database, **not** a model — no inference is required and no network access is needed at runtime.
- Useful entrypoints: `from nltk.corpus import wordnet as wn`; `wn.synonyms(token)` returns synonyms grouped by sense, `wn.synsets(token)` returns the synsets directly. Either is acceptable.
- For LanceDB native FTS use `table.create_fts_index("content", use_tantivy=False, replace=False)` once, then `table.search(<query>, query_type="fts").limit(k).to_list()` (or the `MatchQuery`/`BooleanQuery` builders) for retrieval.
- The schema of the seeded table is `{ id: int64, content: string }`. Document IDs are integers `0..99`.
- Lowercase the query before expansion. Do **not** add or remove any other words; the verifier compares behaviour against the plain `table.search(query, query_type="fts")` baseline.

## Acceptance Criteria
- Project path: `/home/user/myproject`
- Module: `/home/user/myproject/solution.py` exports `expanded_search(query: str, k: int = 10) -> list[int]`.
- The function must read the LanceDB URI from `LANCEDB_URI` and the table name from `LANCEDB_TABLE` (both are set in the container environment).
- A native FTS index (`use_tantivy=False`) must exist on the `content` column of the seeded table after the first invocation of `expanded_search`.
- Return contract: a `list[int]` of length at most `k`, sorted by FTS score descending, containing document IDs drawn from the seeded table.
- The function must materially expand the query: for the seeded corpus, the unexpanded baseline (`table.search(query, query_type="fts").limit(k).to_list()`) and `expanded_search(query, k)` must produce different result sets for at least one synonym-rich query (`car`).
- No outbound network calls are needed at runtime; WordNet is fully available offline after the build-time download.

