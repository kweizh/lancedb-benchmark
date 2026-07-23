# LanceDB ANN Recall / Latency Parameter Sweep

## Background
Approximate-nearest-neighbor (ANN) indexes trade recall for latency. To pick good production settings you must measure, on a fixed benchmark, how recall and query latency change as you turn the search-time knobs of the index. In this task you build a LanceDB ANN vector index over a pre-seeded table of a few thousand deterministic vectors and implement a reusable sweep that evaluates a grid of search-time parameters, reporting recall@k (against exact brute-force ground truth) and mean query latency for every configuration.

Everything runs locally against an embedded LanceDB database on disk. There is no network access, no hosted API, and no model or dataset download.

## Requirements
- Build an ANN vector index on the vector column of the pre-seeded table so that approximate search can be performed.
- Implement a parameter sweep that, for each search-time configuration in a caller-supplied grid, runs vector searches for every query in a caller-supplied query set and computes:
  - `recall@k`: the mean fraction of the exact top-`k` nearest neighbors (by the index's distance metric) that the ANN search returns, averaged over all queries. Exact top-`k` must be computed by brute force (an exhaustive / index-bypassing scan) so it is ground truth.
  - the mean per-query search latency for that configuration.
- The sweep must be deterministic: calling it twice with the same built index, query set, and grid must return identical recall values.
- Increasing search effort (scanning more index partitions and/or reranking more candidates on full vectors) must yield non-decreasing recall, and the highest-effort configuration must reach high recall while the lowest-effort configuration must retrieve strictly fewer correct neighbors.

## Implementation Hints
- Project path: /home/user/myproject
- Put your code in `/home/user/myproject/solution.py`.
- The database already exists on disk. Connect to the LanceDB directory given by the environment variable `LANCEDB_URI` (it is set to `/app/lancedb`). The benchmark table is named `vectors_<RUN_ID>`, where `<RUN_ID>` is the value of the environment variable `ZEALT_RUN_ID`; read it at runtime and do not hard-code a run id. The table has an integer primary key column `id` (values `0..N-1`) and a fixed-size-list float32 column `vector`.
- Build the ANN index as an `IVF_PQ` index on the `vector` column with `num_partitions=64`, `num_sub_vectors=8`, and the `l2` distance metric so the sweep is well defined. Remember that LanceDB index builds run in the background — wait for the index to finish before searching.
- Expose exactly these two callables in `solution.py`:
  - `build_index()` — (re)builds the ANN index on the table's `vector` column; must be safe to call more than once.
  - `sweep(query_set, param_grid, k=10)` — runs the benchmark and returns the results table.
- `query_set` is passed as a 2-D NumPy float32 array of shape `(num_queries, dim)`. `param_grid` is a dict with keys `"nprobes"` and `"refine_factor"`, each mapping to an ascending list of positive integers; evaluate the full Cartesian product of the two lists (one configuration per pair).
- `sweep` must return a `list` of `dict`, one entry per configuration, each dict having EXACTLY these keys: `nprobes` (int), `refine_factor` (int), `recall` (float in [0.0, 1.0]), `mean_latency_ms` (float > 0). The list must be sorted ascending by `(nprobes, refine_factor)`.
- Compute recall as `|ANN_top_k_ids ∩ exact_top_k_ids| / k` per query, then average over all queries. Ground-truth exact neighbors must come from a brute-force / index-bypassing scan on the same table so they are correct.
- Determinism note: LanceDB search only orders rows by distance; break ties by ascending `id` where needed so repeated runs match exactly.

