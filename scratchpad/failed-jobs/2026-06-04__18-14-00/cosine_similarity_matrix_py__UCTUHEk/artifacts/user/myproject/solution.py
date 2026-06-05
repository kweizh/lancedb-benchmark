"""
Cosine Similarity Matrix from LanceDB Self-Search
"""

import os
import numpy as np
import lancedb

_cached_matrix: np.ndarray | None = None


def _get_table():
    uri = os.environ["LANCEDB_URI"]
    tbl_name = os.environ["LANCEDB_TABLE"]
    db = lancedb.connect(uri)
    return db.open_table(tbl_name)


def similarity_matrix() -> np.ndarray:
    """Return the (200, 200) cosine similarity matrix derived from LanceDB searches.

    S[i, j] is the cosine similarity between the rows with id==i and id==j.
    Values are obtained via LanceDB cosine distance search (similarity = 1 - distance).
    The result is cached after the first call.
    """
    global _cached_matrix
    if _cached_matrix is not None:
        return _cached_matrix

    tbl = _get_table()

    # Load all rows sorted by id so index == id
    df = tbl.to_pandas().sort_values("id").reset_index(drop=True)
    n = len(df)  # 200

    S = np.zeros((n, n), dtype=np.float64)

    for idx in range(n):
        row_id = int(df.loc[idx, "id"])
        vec = df.loc[idx, "vector"]

        # Issue a cosine self-search returning all rows
        results = (
            tbl.search(vec)
            .distance_type("cosine")
            .limit(n)
            .to_pandas()
        )

        for _, res_row in results.iterrows():
            j = int(res_row["id"])
            dist = float(res_row["_distance"])
            sim = 1.0 - dist
            S[row_id, j] = sim

    _cached_matrix = S
    return S


def intra_class_mean(label: int) -> float:
    """Return the mean off-diagonal cosine similarity for rows whose label == label.

    Parameters
    ----------
    label : int
        An integer label in [0, 4].

    Returns
    -------
    float
        Mean cosine similarity between all distinct pairs of rows sharing the
        given label (off-diagonal entries of the label's sub-matrix).
    """
    S = similarity_matrix()

    tbl = _get_table()
    df = tbl.to_pandas()
    ids = df.loc[df["label"] == label, "id"].tolist()

    sub = S[np.ix_(ids, ids)]
    # Off-diagonal mask
    mask = ~np.eye(len(ids), dtype=bool)
    return float(sub[mask].mean())
