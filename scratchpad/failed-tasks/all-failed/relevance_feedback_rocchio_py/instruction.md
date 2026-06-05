# Rocchio Pseudo-Relevance Feedback over LanceDB

## Background
Pseudo-relevance feedback (PRF) is a classic information-retrieval technique that improves a noisy first-pass query by pushing it toward documents the user marked as relevant and away from non-relevant ones. The Rocchio update is the canonical formulation:

```
q' = alpha * q0 + beta * mean(rel_vecs) - gamma * mean(nrel_vecs)
```

You are given a pre-seeded LanceDB table that already contains 32-dimensional document vectors. Your job is to implement a function that performs an initial vector search, asks the user (via parameters) which results were relevant, applies the Rocchio update, and re-queries LanceDB with the improved vector.

## Requirements
- Implement `rocchio_search(q0, relevant_ids, alpha=1.0, beta=0.75, gamma=0.15, n_rel=5, n_nrel=5, k=10) -> list[int]` in `/home/user/myproject/solution.py`.
- The function MUST:
  1. Run an initial **cosine** vector search against the pre-seeded LanceDB table for the top `(n_rel + n_nrel)` candidates.
  2. Treat every id in `relevant_ids` as a pseudo-relevant document and every other id in the initial top-`(n_rel + n_nrel)` as non-relevant.
  3. Look up the stored vectors of the relevant and non-relevant ids from the table and compute `q' = alpha * q0 + beta * mean(rel_vecs) - gamma * mean(nrel_vecs)`.
  4. L2-normalize `q'`.
  5. Run a second cosine search with `q'` and return the **top-`k` document ids** as a Python `list[int]`, ordered best-first.
- Read the run-id from the `ZEALT_RUN_ID` environment variable and connect to the pre-seeded table named `documents_${ZEALT_RUN_ID}` under the LanceDB directory `/home/user/myproject/lancedb_data`.
- Provide a small `/home/user/myproject/run.py` CLI that loads `q0` and `relevant_ids` from `/home/user/myproject/query.json`, calls `rocchio_search`, and writes the resulting list to `/home/user/myproject/output.json` as JSON.

## Implementation Hints
- The fixture is deterministic — do **not** re-seed or modify the table. The seed script already ran at container build time and wrote the table under `/home/user/myproject/lancedb_data` together with `query.json`.
- Use `tbl.search(vec).distance_type("cosine").limit(...)` for vector search.
- Look up vectors by id using `tbl.search().where("id IN (...)").limit(...).to_list()` (or `to_pandas()`) — be careful with prefilter semantics.
- Make sure `q'` is L2-normalized before the second search; LanceDB cosine search uses cosine distance and a normalized query is required for correctness.
- Convert all numpy types to plain Python `int`s before returning.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 run.py`
- Inputs are read from `/home/user/myproject/query.json` (created at build time), which contains keys `q0` (list[float] of length 32) and `relevant_ids` (list[int]).
- The CLI writes the result to `/home/user/myproject/output.json` as a JSON list of ints (the top-`k=10` ids returned by `rocchio_search`).
- `solution.py` must expose the exact callable `rocchio_search(q0, relevant_ids, alpha=1.0, beta=0.75, gamma=0.15, n_rel=5, n_nrel=5, k=10) -> list[int]` (importable from `solution`).
- Table name: `documents_${ZEALT_RUN_ID}` under `/home/user/myproject/lancedb_data`.
- Behavior on the rigged fixture (verified by the test):
  - With the build-time `q0` and `relevant_ids` from `query.json`, the returned list of length 10 must have `result[0]` equal to the document id that sits at the centroid of the intended cluster.
  - At least 4 of the top-5 returned ids must belong to the intended cluster (cluster 0, ids 0..99).
  - The result must be stable across two consecutive invocations of `rocchio_search` with the same arguments.

