"""Solution module: derives cosine similarity matrix from LanceDB cosine searches."""

import os
import numpy as np
import lancedb

# Module-level cache
_cached_matrix = None
_cached_labels = None


def _get_table():
    """Connect to the LanceDB table using environment variables."""
    db = lancedb.connect(os.environ["LANCEDB_URI"])
    return db.open_table(os.environ["LANCEDB_TABLE"])


def _build_matrix():
    """Build the 200x200 cosine similarity matrix from LanceDB searches."""
    global _cached_matrix, _cached_labels

    if _cached_matrix is not None:
        return _cached_matrix, _cached_labels

    table = _get_table()
    data = table.to_arrow()

    # Read all ids and vectors from the table
    ids = data.column("id").to_pylist()
    vectors = data.column("vector").to_pylist()
    labels = data.column("label").to_pylist()

    n = len(ids)
    S = np.zeros((n, n), dtype=np.float64)

    # For each row, run a cosine self-search and populate the similarity matrix
    for idx in range(n):
        vec = vectors[idx]
        row_id = ids[idx]

        results = table.search(vec).distance_type("cosine").limit(n).to_pandas()

        for _, row in results.iterrows():
            j = int(row["id"])
            distance = float(row["_distance"])
            S[row_id, j] = 1.0 - distance

    # Force diagonal to exactly 1.0
    np.fill_diagonal(S, 1.0)

    _cached_matrix = S
    _cached_labels = labels

    return _cached_matrix, _cached_labels


def similarity_matrix() -> np.ndarray:
    """Return the (200, 200) cosine similarity matrix derived from LanceDB searches.

    S[i, j] is the cosine similarity between stored vectors with id==i and id==j.
    """
    S, _ = _build_matrix()
    return S


def intra_class_mean(label: int) -> float:
    """Return the mean off-diagonal cosine similarity for rows with the given label."""
    S, labels = _build_matrix()

    # Find all ids whose label column equals the requested label
    indices = [i for i, l in enumerate(labels) if l == label]

    # Extract the submatrix and compute mean of off-diagonal entries
    sub = S[np.ix_(indices, indices)]
    n_sub = len(indices)
    total = sub.sum() - np.trace(sub)  # subtract diagonal
    count = n_sub * (n_sub - 1)
    return float(total / count)