import os
import datetime
import numpy as np
import lancedb
import pyarrow as pa

# ── Configuration ──────────────────────────────────────────────────────────
DB_PATH = "/home/user/myproject/lancedb"
VECTOR_DIM = 384
NUM_CLUSTERS = 10
NUM_ROWS = 2048          # well above the 256 training minimum & 1024 requirement
SEED = 42
NUM_SUB_VECTORS = 8      # aggressive: 384/8 = 48 floats per PQ codeword
NUM_PARTITIONS = 10      # matches number of clusters
CENTER_SCALE = 10.0      # inter-centre separation multiplier
NOISE_STD = 0.5          # per-dimension Gaussian noise


def get_table_name() -> str:
    """Derive the table name from the ZEALT_RUN_ID env var."""
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"bench_{run_id}"


def _generate_clustered_data(rng: np.random.Generator):
    """Return (ids, vectors, centres) deterministically from *rng*."""
    # Random unit-norm centres, then scaled for separation
    centres = rng.standard_normal((NUM_CLUSTERS, VECTOR_DIM)).astype(np.float32)
    norms = np.linalg.norm(centres, axis=1, keepdims=True)
    centres = centres / norms * CENTER_SCALE

    base, remainder = divmod(NUM_ROWS, NUM_CLUSTERS)
    chunks = []
    for ci in range(NUM_CLUSTERS):
        n = base + (1 if ci < remainder else 0)
        noise = rng.standard_normal((n, VECTOR_DIM)).astype(np.float32) * NOISE_STD
        chunks.append(centres[ci] + noise)

    vectors = np.vstack(chunks).astype(np.float32)
    ids = np.arange(len(vectors), dtype=np.int64)
    return ids, vectors, centres


def create_table_and_index():
    """(Re)create the LanceDB table and IVF_PQ index from scratch."""
    rng = np.random.default_rng(SEED)
    ids, vectors, centres = _generate_clustered_data(rng)

    table_name = get_table_name()

    # Build an Arrow table with fixed-size-list[384] of float32
    arrow_table = pa.table(
        {
            "id": pa.array(ids.tolist(), type=pa.int64()),
            "vector": pa.array(
                [v.tolist() for v in vectors],
                type=pa.list_(pa.float32(), VECTOR_DIM),
            ),
        }
    )

    db = lancedb.connect(DB_PATH)

    # Idempotent: drop existing table if present
    if table_name in db.table_names():
        db.drop_table(table_name)

    table = db.create_table(table_name, arrow_table)

    # Build the aggressively-quantised IVF_PQ index
    table.create_index(
        metric="L2",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=NUM_PARTITIONS,
        num_sub_vectors=NUM_SUB_VECTORS,
        replace=True,
    )

    # Wait until the index is usable
    try:
        indices = table.list_indices()
        table.wait_for_index(indices, timeout=datetime.timedelta(seconds=60))
    except Exception:
        pass  # some lancedb versions create the index synchronously

    return table


def evaluate_recall(num_queries: int = 50, k: int = 10) -> float:
    """Average recall@k of IVF_PQ search vs brute-force ground truth.

    Parameters
    ----------
    num_queries : int
        Number of query vectors (drawn deterministically).
    k : int
        Number of nearest neighbours to retrieve.

    Returns
    -------
    float
        Mean fraction of true top-k neighbours found by the index,
        in [0.0, 1.0].
    """
    # Reproduce the same cluster centres used during table creation
    rng = np.random.default_rng(SEED)
    _, _, centres = _generate_clustered_data(rng)

    # Deterministic query vectors from the same distribution
    q_rng = np.random.default_rng(SEED + 1000)
    query_vectors = []
    for _ in range(num_queries):
        ci = q_rng.integers(0, NUM_CLUSTERS)
        noise = q_rng.standard_normal(VECTOR_DIM).astype(np.float32) * NOISE_STD
        query_vectors.append(centres[ci] + noise)

    # Open the table
    db = lancedb.connect(DB_PATH)
    table_name = get_table_name()
    table = db.open_table(table_name)

    # Load all stored vectors for brute-force ground truth
    all_data = table.to_arrow()
    all_ids = all_data["id"].to_numpy()
    all_vectors = np.array(
        [v.as_py() for v in all_data["vector"]], dtype=np.float32
    )

    recalls = []
    for i in range(num_queries):
        qv = query_vectors[i]

        # ── Brute-force ground truth (exact L2) ──
        dists = np.sum((all_vectors - qv) ** 2, axis=1)
        gt_idx = np.argsort(dists)[:k]
        gt_ids = set(int(x) for x in all_ids[gt_idx])

        # ── IVF_PQ ANN search with recall-boosting query params ──
        results = (
            table.search(qv.tolist())
            .limit(k)
            .nprobes(NUM_PARTITIONS)   # probe every partition (10 is tiny)
            .refine_factor(10)          # re-rank 10×k candidates by exact L2
            .to_arrow()
        )
        ret_ids = set(int(x) for x in results["id"].to_numpy())

        recalls.append(len(gt_ids & ret_ids) / k)

    return float(np.mean(recalls))


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    create_table_and_index()
    recall = evaluate_recall()
    print(f"Average recall@10: {recall:.4f}")