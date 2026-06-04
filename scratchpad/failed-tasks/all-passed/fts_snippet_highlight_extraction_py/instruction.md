# LanceDB FTS Snippet Highlight Extraction

## Background
LanceDB's native (Lance-based) full-text search index can be configured with `with_position=True` so that positional information is stored alongside the BM25 posting list. This positional information is essential for downstream features such as snippet generation, where a short context window is extracted from a long document body and the matched term is highlighted.

In this task you will build a search helper on top of a pre-seeded LanceDB table of articles. The table has already been created at container build time, populated with 50 articles whose bodies are 200+ words each, and a native FTS index with positional information has been built on the `body` column. Your job is to implement the snippet-extraction layer that returns short context windows around the first occurrence of the query term, with the term wrapped in `<mark>...</mark>` tags.

## Requirements
- Implement a Python module `solution.py` at `/home/user/myproject/solution.py` that exposes a single callable: `search_with_snippets(query: str, k: int, snippet_chars: int = 120)`.
- The function must execute a full-text search against the existing LanceDB table (whose name is provided via the `LANCE_TABLE` environment variable, in the database at `/home/user/myproject/data`) and return a Python `list[dict]` of at most `k` results.
- Each result dict must contain exactly the following keys:
  - `id` (int): the row id from the table.
  - `score` (float): the BM25 score returned by LanceDB's FTS (the `_score` column).
  - `snippet` (str): a substring window taken from the article `body`. The window length must be at most `snippet_chars` characters of original body text, centered (as best as possible) on the first case-insensitive occurrence of the query term. Within the snippet, the matched query term must be wrapped in `<mark>` and `</mark>` tags. The case of the wrapped text must match the case used in the original body.
  - `snippet_offset` (int): the character offset in the original `body` at which the snippet window begins (i.e., the index into `body[snippet_offset:snippet_offset+window_len]` that yields the un-highlighted snippet text).
- Results must be ordered by descending `score` (highest first), matching LanceDB's default FTS ordering.
- If a hit's body does not actually contain the query term (e.g., due to stemming), fall back to wrapping the first whole-word match you can find for any token of the query, and if even that fails, return the leading `snippet_chars` characters with no `<mark>` markup. The `snippet_offset` must still be the true offset of the returned window inside the original body.

## Implementation Hints
- Open the existing table with `lancedb.connect("/home/user/myproject/data").open_table(os.environ["LANCE_TABLE"])`. Do NOT recreate or drop the table — the fixture rows and FTS index are already in place.
- Use `table.search(query, query_type="fts").limit(k).to_list()` (or the equivalent builder API) to obtain BM25-ranked candidates with a `_score` field.
- Locate the first occurrence of the query term in `body` using either LanceDB's positional FTS facilities or a plain Python string search (e.g., `body.lower().find(query.lower())`). Either approach is acceptable.
- When building the window, clamp the start/end indices so they remain inside `[0, len(body)]`, and ensure the un-highlighted character count does not exceed `snippet_chars`. The `<mark>` tags themselves do not count toward the `snippet_chars` budget but the final snippet string will obviously be slightly longer.
- Pure-Python string slicing is sufficient; no extra third-party dependencies are required.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 -c "import solution, json; print(json.dumps(solution.search_with_snippets('example query', 3, 120)))"` (the verifier will import `solution` directly rather than rely on this command).
- `solution.search_with_snippets(query, k, snippet_chars)` must return a `list[dict]` with the schema described in Requirements.
- For every dict in the result list:
  - `id` is an integer matching a row in the `LANCE_TABLE` table.
  - `score` is a float (the BM25 `_score`).
  - `snippet` is a string whose length excluding the `<mark>...</mark>` markup is ≤ `snippet_chars` (i.e., `len(snippet) - len('<mark></mark>') ≤ snippet_chars` whenever `<mark>` markup is present).
  - When the query term occurs in `body`, the `<mark>` and `</mark>` tags surround the exact matched substring of `body` (case-preserved), and `snippet_offset` ≤ position-of-match ≤ `snippet_offset + snippet_chars`.
  - The substring `body[snippet_offset : snippet_offset + window_len]` (where `window_len` is the snippet length without markup) equals the snippet with the `<mark>` and `</mark>` tags removed.
- The function is order-stable: results are sorted by descending `_score`.

