# ColBERT-Style Multi-Vector Late-Interaction Retrieval on LanceDB

## Background
Modern retrievers such as ColBERT represent each document as a *bag of token-level vectors* rather than a single pooled embedding. At query time a multi-vector query is scored against each candidate document via the **MaxSim** late-interaction operator: for every query token vector, take the maximum cosine similarity against any of the document's token vectors, then sum those per-token maxima.

You are given a pre-seeded LanceDB table containing 60 documents, each represented by 4 token-level vectors of dimension 32 (240 rows total). Your job is to implement the retrieval logic.

## Requirements
- Connect to the LanceDB database at `/home/user/myproject/lancedb_data/` and open the table named `colbert_tokens`.
- The table schema is `doc_id: int64`, `token_idx: int32`, `embedding: fixed_size_list<float32, 32>`. Every `(doc_id, token_idx)` pair is unique. `doc_id` ranges over `0..59` and `token_idx` over `0..3`.
- Implement a function `colbert_search(query_token_vecs: np.ndarray, k: int = 5) -> list[int]` in `/home/user/myproject/solution.py` that:
  1. Accepts an `(M, 32)` float numpy array (typically `M = 3`).
  2. Runs **one LanceDB vector search per query token** using the `cosine` distance metric, retrieving the top-50 `(doc_id, token_idx, _distance)` rows for that query token.
  3. Converts each `_distance` to a cosine similarity (`sim = 1 - distance`) and, for every candidate document that appears in any of the M result sets, computes `MaxSim(q_i, doc) = max over the doc's retrieved token rows of cosine(q_i, t)`. Documents whose token did not appear in a particular query's top-50 contribute `0` for that `q_i`. The **late-interaction score** for a doc is the sum of MaxSim values across the M query tokens.
  4. Returns the **top-k `doc_id` values** as a Python `list[int]`, ordered by descending late-interaction score. Ties may be broken by ascending `doc_id`.

## Implementation Hints
- Use `lancedb.connect(...)` and `tbl.search(query_vec).distance_type("cosine").limit(50).to_pandas()` (or `.to_list()`) to issue each per-token search.
- The aggregation step is plain pandas/numpy — group by `doc_id`, take the max similarity per group, sum across the M searches, sort descending.
- The `embedding` column is a `fixed_size_list<float32, 32>`. Make sure your query vectors are 1-D `np.float32` arrays of length 32.
- Cosine similarity in LanceDB is `1 - distance`, where `distance` is what the search returns.

## Acceptance Criteria
- Project path: /home/user/myproject
- Solution file: /home/user/myproject/solution.py
- The module must expose a callable `colbert_search(query_token_vecs: np.ndarray, k: int = 5) -> list[int]` with the signature and contract described above.
- Calling `colbert_search` must not raise for valid `(M, 32)` float arrays and must return at most `k` distinct `doc_id` integers in descending late-interaction score order.
- The function must perform exactly **M** independent LanceDB vector searches (one per query token) — it must not rely on average-pooling or sum-pooling the query into a single vector.

