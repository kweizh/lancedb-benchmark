"""
LanceDB Recall Benchmark: FP32 Brute-Force vs Aggressively Quantized IVF_PQ

Generates a synthetic clustered embedding dataset, builds an IVF_PQ vector index,
and measures recall@k vs brute-force FP32 ground truth.
"""

import os
import datetime
import numpy as np
import pyarrow as pa
import lancedb

# ── Configuration ─────────────────────────────────────────────────────────────
SEED          = 42
NUM_ROWS      = 1024
DIM           = 384
NUM_CLUSTERS  = 10
NUM_PARTITIONS = 10       # IVF coarse partitions  (≈ NUM_CLUSTERS)
NUM_SUB_VECTORS = 12      # PQ sub-vectors (384 / 12 = 32 floats each) — aggressive
NOISE_SCALE   = 0.15      # intra-cluster Gaussian σ
DB_PATH       = "/home/user/myproject/lancedb"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _table_name() -> str:
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"bench_{run_id}"


def _build_clustered_vectors(rng: np.random.Generator) -> np.ndarray:
    """Return (NUM_ROWS, DIM) float32 array of well-separated clustered vectors."""
    # Random unit-norm cluster centres
    centers_raw = rng.standard_normal((NUM_CLUSTERS, DIM)).astype(np.float32)
    norms = np.linalg.norm(centers_raw, axis=1, keepdims=True)
    centers = centers_raw / norms          # unit-norm → well-separated

    # Assign rows round-robin so every cluster has equal size
    assignments = np.arange(NUM_ROWS, dtype=np.int32) % NUM_CLUSTERS

    # Per-row vectors = cluster centre + small Gaussian noise
    noise = rng.standard_normal((NUM_ROWS, DIM)).astype(np.float32) * NOISE_SCALE
    vectors = centers[assignments] + noise
    return vectors


def _schema() -> pa.Schema:
    return pa.schema([
        pa.field("id",     pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), DIM)),
    ])


def _build_table(db: lancedb.DBConnection, table_name: str, vectors: np.ndarray) -> lancedb.Table:
    """(Re-)create the Lance table, populate it, build IVF_PQ index."""
    # Drop existing table if present so the function is idempotent
    existing = db.table_names()
    if table_name in existing:
        db.drop_table(table_name)

    schema = _schema()
    table = db.create_table(table_name, schema=schema)

    # Batch-insert rows
    batch_size = 256
    for start in range(0, NUM_ROWS, batch_size):
        end = min(start + batch_size, NUM_ROWS)
        rows = [
            {"id": i, "vector": vectors[i].tolist()}
            for i in range(start, end)
        ]
        table.add(rows)

    # Build IVF_PQ index
    index_name = "vector_idx"
    table.create_index(
        metric="L2",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=NUM_PARTITIONS,
        num_sub_vectors=NUM_SUB_VECTORS,
        replace=True,
        name=index_name,
    )

    # Wait until index is ready (up to 60 s)
    table.wait_for_index(
        [index_name],
        timeout=datetime.timedelta(seconds=60),
    )

    return table


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate_recall(num_queries: int = 50, k: int = 10) -> float:
    """
    Draw `num_queries` query vectors deterministically, compute brute-force
    top-k ground truth, then run the IVF_PQ index and return average recall@k.

    Returns a float in [0.0, 1.0].
    """
    db         = lancedb.connect(DB_PATH)
    table_name = _table_name()

    if table_name not in db.table_names():
        raise RuntimeError(
            f"Table '{table_name}' not found in {DB_PATH}. "
            "Run solution.py first to build the index."
        )

    table = db.open_table(table_name)

    # ── Load all stored vectors for brute-force ground truth ──────────────────
    arrow_tbl  = table.to_arrow()
    all_ids    = arrow_tbl.column("id").to_pylist()
    # Each cell is a list of floats; convert to 2-D numpy array
    all_vecs   = np.array([v.as_py() for v in arrow_tbl.column("vector")], dtype=np.float32)
    # Map id → row index in all_vecs
    id_to_idx  = {id_val: idx for idx, id_val in enumerate(all_ids)}

    # ── Generate query vectors (deterministic, independent of build RNG) ──────
    query_rng = np.random.default_rng(SEED + 1000)
    # Sample query vectors as noisy copies of random stored vectors
    chosen_row_indices = query_rng.integers(0, NUM_ROWS, size=num_queries)
    query_noise = query_rng.standard_normal((num_queries, DIM)).astype(np.float32) * (NOISE_SCALE * 0.5)
    query_vecs  = all_vecs[chosen_row_indices] + query_noise

    # ── Brute-force top-k ground truth ───────────────────────────────────────
    def brute_force_topk(qv: np.ndarray) -> set:
        # Squared L2 distances to every stored vector
        diff  = all_vecs - qv[np.newaxis, :]     # (N, DIM)
        dists = (diff * diff).sum(axis=1)         # (N,)
        top_k_idx = np.argpartition(dists, k)[:k]
        # resolve to id values
        return {all_ids[i] for i in top_k_idx}

    # ── IVF_PQ search (with nprobes + refine_factor to recover recall) ────────
    NPROBES       = 20   # probe 2× partitions (generous for recall)
    REFINE_FACTOR = 10   # re-rank 10× candidates by exact L2

    total_recall = 0.0
    for qv in query_vecs:
        gt_ids = brute_force_topk(qv)

        result_df = (
            table
            .search(qv.tolist())
            .limit(k)
            .nprobes(NPROBES)
            .refine_factor(REFINE_FACTOR)
            .to_pandas()
        )
        returned_ids = set(result_df["id"].tolist())
        total_recall += len(returned_ids & gt_ids) / k

    return total_recall / num_queries


# ── Main: build table + index, then print recall ───────────────────────────────

def main():
    print(f"LanceDB version : {lancedb.__version__}")
    print(f"DB path         : {DB_PATH}")
    table_name = _table_name()
    print(f"Table name      : {table_name}")

    # Deterministic data generation
    rng     = np.random.default_rng(SEED)
    vectors = _build_clustered_vectors(rng)
    print(f"Generated {NUM_ROWS} × {DIM}-dim clustered vectors "
          f"({NUM_CLUSTERS} clusters, noise σ={NOISE_SCALE})")

    db    = lancedb.connect(DB_PATH)
    table = _build_table(db, table_name, vectors)
    print(f"Table rows      : {table.count_rows()}")
    print(f"Indices present : {[idx.name for idx in table.list_indices()]}")

    recall = evaluate_recall(num_queries=50, k=10)
    print(f"\nRecall@10 (IVF_PQ vs brute-force FP32): {recall:.4f}")
    if recall >= 0.70:
        print("✓ Meets quality bar (≥ 0.70)")
    else:
        print("✗ Below quality bar (≥ 0.70)")

    return recall


if __name__ == "__main__":
    main()
