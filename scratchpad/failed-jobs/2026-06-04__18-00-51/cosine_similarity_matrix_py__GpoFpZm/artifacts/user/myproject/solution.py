"""
Cosine Similarity Matrix from LanceDB Self-Search.

All similarity values are derived from LanceDB cosine search results:
  similarity = 1 - cosine_distance
"""

import os
import numpy as np
import lancedb

# ---------------------------------------------------------------------------
# Module-level cache so the matrix is only built once per process lifetime.
# ---------------------------------------------------------------------------
_cached_matrix: np.ndarray | None = None


def _build_matrix() -> np.ndarray:
    """Connect to LanceDB and build the (200, 200) cosine similarity matrix."""
    uri = os.environ.get("LANCEDB_URI", "/home/user/myproject/lancedb_data")
    table_name = os.environ.get("LANCEDB_TABLE", "vectors")

    db = lancedb.connect(uri)
    table = db.open_table(table_name)

    # Load all rows once so we can look up each row's stored vector by id.
    df = table.to_pandas()
    n = len(df)  # should be 200

    # Build an id -> (vector, row_index) mapping.
    # We index the matrix by the 'id' column value.
    id_to_vector = {int(row["id"]): np.array(row["vector"], dtype=np.float32)
                    for _, row in df.iterrows()}

    all_ids = sorted(id_to_vector.keys())  # 0..199

    S = np.zeros((n, n), dtype=np.float64)

    for i in all_ids:
        vec = id_to_vector[i]

        # Issue a LanceDB cosine search for this row's vector.
        results = (
            table.search(vec)
            .distance_type("cosine")
            .limit(n)
            .to_pandas()
        )

        # Populate row i of the matrix using the returned distances.
        for _, result_row in results.iterrows():
            j = int(result_row["id"])
            distance = float(result_row["_distance"])
            S[i, j] = 1.0 - distance

    return S


def similarity_matrix() -> np.ndarray:
    """Return the (200, 200) cosine similarity matrix derived from LanceDB searches.

    S[i, j] is the cosine similarity between the vectors of the rows
    with id == i and id == j.  The result is cached after the first call.
    """
    global _cached_matrix
    if _cached_matrix is None:
        _cached_matrix = _build_matrix()
    return _cached_matrix


def intra_class_mean(label: int) -> float:
    """Return the mean off-diagonal cosine similarity for rows with the given label.

    Parameters
    ----------
    label : int
        An integer label in [0, 4].

    Returns
    -------
    float
        Mean cosine similarity between distinct rows that share the label.
    """
    S = similarity_matrix()

    uri = os.environ.get("LANCEDB_URI", "/home/user/myproject/lancedb_data")
    table_name = os.environ.get("LANCEDB_TABLE", "vectors")

    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    df = table.to_pandas()

    ids = df[df["label"] == label]["id"].astype(int).tolist()

    sub = S[np.ix_(ids, ids)]
    # Exclude the diagonal (self-similarity = 1.0).
    n = len(ids)
    off_diag_sum = sub.sum() - np.trace(sub)
    off_diag_count = n * (n - 1)

    if off_diag_count == 0:
        return 0.0

    return float(off_diag_sum / off_diag_count)
