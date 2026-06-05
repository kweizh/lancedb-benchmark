# Faceted Product Search with LanceDB Boolean Metadata Filters

## Background
You are building a faceted product search backend on top of LanceDB. The container ships a pre-seeded `products_${ZEALT_RUN_ID}` table (1000 rows, 32-dimensional float32 embeddings) where each row also carries the metadata facets `brand`, `category`, `color`, `in_stock`, and `price`. Your job is to expose a single Python function that combines vector similarity search with arbitrary boolean metadata filters and additionally reports aggregate facet counts over the filtered candidate set.

## Requirements
- Implement `faceted_search(query_vec, facets, k)` in `/home/user/myproject/solution.py`.
- All metadata filtering MUST be pushed down to LanceDB via a server-side `where` clause built from the `facets` dict.
- Return both the ranked vector-search results AND group-by facet counts computed over the filtered set.
- Be deterministic: same `(query_vec, facets, k)` must produce identical output across calls.

## Implementation Hints
- The seeded LanceDB database lives at `/home/user/myproject/data/lancedb` and the table is named `products_${ZEALT_RUN_ID}` (read `ZEALT_RUN_ID` from the environment).
- The `facets` dict uses these optional keys:
  - `"brand"`: list of allowed brand strings (`IN` filter)
  - `"category"`: list of allowed category strings (`IN` filter)
  - `"color"`: list of allowed color strings (`IN` filter)
  - `"in_stock"`: bool (exact match)
  - `"price_max"`: float (price `<=` price_max)
  - `"price_min"`: float (price `>=` price_min)
- Combine all active filters with `AND` and pass them via `tbl.search(query_vec).where(<sql>).limit(k)`.
- Compute `facet_counts` for `brand`, `category`, `color`, and `in_stock` by grouping the rows that satisfy the same `where` predicate (without the vector-search top-k cutoff). A clean approach is `tbl.search().where(<sql>).limit(<large>).to_pandas()` (or `tbl.to_pandas()` + the same predicate) then `groupby(col).size()`.
- Use the L2 distance metric (the table default) and tie-break by ascending `id` to match the ground-truth ordering.
- Quote string literals in SQL filters with single quotes and escape embedded apostrophes by doubling them.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 -c "from solution import faceted_search; ..."` (the verifier imports `solution.faceted_search` directly).
- The function signature MUST be `faceted_search(query_vec, facets: dict, k: int) -> dict` and the return value MUST have exactly these top-level keys:
  - `"results"`: a list of dicts ordered by ascending `_distance` then ascending `id`, each containing `{"id": int, "brand": str, "category": str, "color": str, "in_stock": bool, "price": float, "distance": float}` (no `vector` field).
  - `"facet_counts"`: a dict `{"brand": {value: count}, "category": {value: count}, "color": {value: count}, "in_stock": {"true": count, "false": count}}` whose counts are computed over the FULL filtered set (NOT just the top-k).
- The metadata filter MUST be evaluated on the LanceDB server side via a single `where` clause; reading the full table into pandas and filtering in Python is not acceptable for the result list.
- `len(results) <= k` and contains exactly the top-k matches by L2 distance over the filtered set (with `id` ASC as the tie-break).
- An empty `facets` dict means "no filter" and the function must still return facet counts over all 1000 rows.
- The seeded table name MUST use the `${ZEALT_RUN_ID}` suffix.

