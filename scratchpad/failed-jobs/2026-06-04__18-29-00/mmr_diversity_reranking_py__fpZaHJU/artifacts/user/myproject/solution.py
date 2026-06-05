"""LanceDB MMR Diversity Re-ranking solution."""

import os
import numpy as np
import pyarrow as pa
import lancedb


def build_dataset() -> None:
    """Build the deterministic fixture table at /app/db."""
    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"mmr_docs_{run_id}"

    rng = np.random.default_rng(seed=2026)

    # Step 1: Draw (32, 32) standard-normal matrix and compute QR
    A = rng.standard_normal((32, 32))
    Q, _ = np.linalg.qr(A)

    # Q[:, c] is the centroid of cluster c for c in range(10)
    # Each centroid is a unit vector and they are mutually orthogonal

    # Step 2: Generate rows
    ids = []
    cluster_ids = []
    vectors = []

    for c in range(10):
        centroid = Q[:, c]  # shape (32,)
        for j in range(12):
            noise = rng.standard_normal(32)  # fresh (32,) noise vector
            row_vec = centroid + 0.05 * noise
            row_vec = row_vec.astype(np.float32)

            ids.append(c * 12 + j)
            cluster_ids.append(c)
            vectors.append(row_vec.tolist())

    # Step 3: Create LanceDB table
    db = lancedb.connect("/app/db")

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("cluster_id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 32)),
    ])

    table = db.create_table(
        table_name,
        schema=schema,
        mode="overwrite",
    )

    # Insert data
    data = pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "cluster_id": pa.array(cluster_ids, type=pa.int64()),
            "vector": pa.array(vectors, type=pa.list_(pa.float32(), 32)),
        },
        schema=schema,
    )

    table.add(data)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors using float64."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def mmr_search(query_vec, k=10, lambda_=0.5) -> list[int]:
    """Run MMR re-ranking against the LanceDB table.

    Args:
        query_vec: 1-D iterable of length 32 (numpy array or list of floats).
        k: Number of results to return.
        lambda_: Trade-off between relevance (1.0) and diversity (0.0).

    Returns:
        List of k document ids in MMR selection order.
    """
    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"mmr_docs_{run_id}"

    db = lancedb.connect("/app/db")
    table = db.open_table(table_name)

    # Convert query_vec to numpy float64 for precision
    q = np.asarray(query_vec, dtype=np.float64)

    # Step 1: Get top-30 candidates from cosine vector search
    results = table.search(q.tolist()).distance_type("cosine").limit(30).to_list()

    # Build candidate pool with vectors converted to float64
    candidates = []
    for row in results:
        vec = np.asarray(row["vector"], dtype=np.float64)
        candidates.append({
            "id": row["id"],
            "cluster_id": row["cluster_id"],
            "vector": vec,
        })

    # Precompute query similarity for all candidates
    for cand in candidates:
        cand["sim_q"] = _cosine_sim(q, cand["vector"])

    # Step 2: MMR iterative selection
    selected = []  # list of candidate dicts
    selected_ids = []
    remaining = list(candidates)

    while len(selected) < k and remaining:
        best_score = -float("inf")
        best_idx = -1

        for i, cand in enumerate(remaining):
            # Relevance term: lambda * sim(q, d)
            relevance = lambda_ * cand["sim_q"]

            # Redundancy term: (1 - lambda) * max sim(d, d') for d' in selected
            if selected:
                max_sim = max(
                    _cosine_sim(cand["vector"], sel["vector"])
                    for sel in selected
                )
                redundancy = (1 - lambda_) * max_sim
            else:
                redundancy = 0.0

            score = relevance - redundancy

            if score > best_score:
                best_score = score
                best_idx = i

        # Pick the best candidate
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_ids.append(chosen["id"])

    return selected_ids


if __name__ == "__main__":
    build_dataset()
    print("Dataset built successfully.")