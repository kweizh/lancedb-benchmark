# Build a Persisted KNN Graph on LanceDB and Query Paths

## Background
A pre-seeded LanceDB database lives at `/home/user/myproject/lancedb_data` containing the table `embeddings_${ZEALT_RUN_ID}` with 500 rows of 64-dimensional float32 vectors. Your job is to materialize a k-nearest-neighbor (KNN) graph over the embeddings as a second LanceDB table, then expose a graph-path query that traverses the persisted edges with LanceDB SQL filtering.

## Requirements
- Read the source table `embeddings_${ZEALT_RUN_ID}` (schema: `id` int64, `vector` fixed_size_list<float32, 64>). Do NOT modify the source table.
- For every row, compute the 10 nearest neighbors using LanceDB vector search against the same table.
- Persist the result as a new LanceDB table `knn_edges_${ZEALT_RUN_ID}` with PyArrow schema `{src_id: int64, dst_id: int64, rank: int32, distance: float32}` containing exactly 5000 rows (500 sources × 10 edges).
- Expose `find_path(src_id, dst_id, max_hops)` that runs a breadth-first search over the persisted KNN edge table and returns the node-id path from `src_id` to `dst_id` if one exists within `max_hops`, otherwise `None`. The path must start with `src_id` and end with `dst_id`.
- The BFS must traverse the edges by issuing LanceDB SQL `where` queries against `knn_edges_${ZEALT_RUN_ID}` per frontier (no precomputed in-memory adjacency list).

## Implementation Hints
- Use `lancedb.connect("/home/user/myproject/lancedb_data")` and read the run id from the `ZEALT_RUN_ID` environment variable.
- Use `table.search(query_vector).distance_type("l2").limit(11)` per row and decide whether `rank=0` is the self-match (rank 0 may be the self-row; ranks 1..9 must not be self-loops).
- Build the persisted edges as a PyArrow table so the dtypes match the required schema exactly (int64, int64, int32, float32).
- For BFS, expand a frontier with something like `edges.search().where(f"src_id IN ({csv})").to_arrow()` once per hop and track newly-seen destinations.

## Acceptance Criteria
- Project path: /home/user/myproject
- LanceDB directory: /home/user/myproject/lancedb_data
- Output table: `knn_edges_${ZEALT_RUN_ID}` with PyArrow fields `src_id: int64`, `dst_id: int64`, `rank: int32`, `distance: float32` and exactly 5000 rows.
- For each `src_id` in the source table, the 10 corresponding rows must have `rank` values forming the set `{0,1,2,3,4,5,6,7,8,9}` and `distance` non-decreasing in `rank` order.
- For any row with `rank >= 1`, `dst_id != src_id`.
- Build command: `python3 build_graph.py` — must (re)build the `knn_edges_${ZEALT_RUN_ID}` table and print `Built knn_edges rows=<N>` to stdout where `<N>` is the persisted row count.
- Path command: `python3 query_path.py --src <S> --dst <D> --max-hops <H>` — must print a single JSON line of shape `{"path": [<int>, ...]}` (list of node ids starting with `<S>` and ending with `<D>`) when a path of length ≤ H exists, otherwise `{"path": null}`.
- Python module: `/home/user/myproject/solution.py` exposes `find_path(src_id: int, dst_id: int, max_hops: int) -> list[int] | None` and `build_knn_graph() -> int` (returns the persisted row count).

