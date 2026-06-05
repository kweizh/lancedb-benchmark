# LanceDB Autocomplete Service: SQL Prefix Matching with Semantic Fallback

## Background
You are building an autocomplete service for a movie discovery product. The catalog is stored in LanceDB and contains 500 movie titles, each with a precomputed 32-dimensional embedding and a popularity score. A secondary table (`prefix_vectors`) maps 50 anchor prefix strings to their own precomputed 32-dimensional embeddings. Both tables have already been seeded at container build time and are read-only at runtime.

The goal is to ship a Python module that returns autocomplete suggestions by combining two strategies: a fast SQL prefix match for typed-prefix exactness, and a vector-search fallback (using the prefix's precomputed embedding) when too few prefix matches exist.

## Requirements
Implement a Python module `solution.py` at `/home/user/myproject/solution.py` that exposes:

```python
def autocomplete(prefix: str, k: int = 10) -> list[dict]: ...
```

The function MUST:
1. Open the LanceDB instance at the URI given by the `LANCE_DB_URI` environment variable. Read the movies table whose name is in the `MOVIES_TABLE` environment variable and the anchor-prefix table whose name is in the `PREFIX_TABLE` environment variable.
2. Perform a **case-insensitive** SQL prefix match (i.e., match every row whose `lower(title)` starts with `lower(prefix)`).
3. If the SQL prefix match yields **at least `k`** rows, return the top `k` rows ordered by `popularity` **descending**. Each row's `source` field MUST be the string `"prefix"`.
4. If the SQL prefix match yields **fewer than `k`** rows (M < k), return:
   - All M prefix matches first, ordered by `popularity` descending (each with `source == "prefix"`), followed by
   - The top `(k - M)` results of a vector search where the query vector is looked up from the `prefix_vectors` table for the lowercased `prefix`, executed against the movies table, **excluding** any movie `id` already returned in the prefix portion. Each of these rows' `source` field MUST be the string `"semantic"`.
5. Each element of the returned list MUST be a dict with at minimum these keys:
   - `id` (int): movie row id
   - `title` (str): original title (preserving case)
   - `popularity` (float): popularity score
   - `source` (str): either `"prefix"` or `"semantic"`

The verifier will only call `autocomplete` with prefixes that are present in the `prefix_vectors` table (lowercased keys).

## Implementation Hints
- Connect with `lancedb.connect(...)` and open both tables via `db.open_table(...)`.
- The titles table has been seeded with a `title_lower` column to make the SQL prefix predicate trivial; you may use it or compute `lower(...)` yourself.
- Use `table.search(...).where("...", prefilter=True)` to combine the vector query with an SQL exclusion of already-returned ids.
- Look up a prefix's vector by filtering `prefix_vectors` on the lowercased prefix string. Convert the returned arrow/pandas vector cell into a Python list / numpy array before passing it to `search()`.
- Be defensive about empty results: a prefix with zero SQL matches must still produce `k` semantic results (assuming the prefix is in `prefix_vectors`).

## Acceptance Criteria
- Project path: /home/user/myproject
- Module: `solution.py` exposing `autocomplete(prefix: str, k: int = 10) -> list[dict]`
- Configuration is read from the environment:
  - `LANCE_DB_URI`: filesystem URI of the LanceDB instance
  - `MOVIES_TABLE`: name of the seeded movies table
  - `PREFIX_TABLE`: name of the seeded prefix-vectors table
- For any call `autocomplete(prefix, k)`:
  - The returned object is a Python `list` of length **exactly** `k` (assuming sufficient data exists, which is guaranteed for verifier-chosen prefixes).
  - Each element is a `dict` containing at least the keys `id`, `title`, `popularity`, `source`.
  - When the SQL prefix predicate `lower(title) LIKE 'lower(prefix)%'` yields M ≥ k movies, every returned element has `source == "prefix"` and the elements are ordered by `popularity` strictly non-increasing.
  - When M < k, the first M elements have `source == "prefix"` (ordered by `popularity` non-increasing) and the remaining (k − M) elements have `source == "semantic"`. None of the semantic elements may have an `id` that appears in the prefix portion.
  - For every element with `source == "prefix"`, the lowercased `title` MUST start with the lowercased `prefix`.
  - The set of `id`s returned in the semantic portion MUST equal the set of top `(k − M)` movie ids when the precomputed `prefix_vectors[lower(prefix)]` vector is used as the LanceDB query vector (default L2 distance), excluding any id in the prefix portion.

