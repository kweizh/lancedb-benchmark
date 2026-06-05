"""Solution module exposing a search(query_vec, k) function that projects
a 128-d query vector into 16-d PCA space and searches the compressed
LanceDB table for the k most similar rows."""

import os
import numpy as np
import lancedb

# ---------------------------------------------------------------------------
# Lazily-loaded module-level state (db connection, table, PCA model)
# ---------------------------------------------------------------------------
_pca_model = None      # dict with 'components' and 'mean'
_db = None
_table = None


def _load_pca_model():
    """Load (and cache) the PCA model from /app/pca_model.npz."""
    global _pca_model
    if _pca_model is None:
        archive = np.load("/app/pca_model.npz")
        _pca_model = {
            "components": archive["components"],   # (16, 128)
            "mean": archive["mean"],               # (128,)
        }
    return _pca_model


def _get_table():
    """Open (and cache) a reference to the PCA LanceDB table."""
    global _db, _table
    if _db is None:
        _db = lancedb.connect("/home/user/myproject/lancedb")
    if _table is None:
        run_id = os.environ["ZEALT_RUN_ID"]
        table_name = f"articles_pca_{run_id}"
        _table = _db.open_table(table_name)
    return _table


def _project(query_vec):
    """Project a 128-d query vector into the 16-d PCA space using the
    persisted model.

    This follows the sklearn convention:
        projected = (query - mean) @ components^T
    """
    model = _load_pca_model()
    q = np.asarray(query_vec, dtype=np.float64).reshape(1, -1)  # (1, 128)
    centered = q - model["mean"]                                  # (1, 128)
    projected = centered @ model["components"].T                   # (1, 16)
    return projected.flatten().tolist()                            # length 16


def search(query_vec, k):
    """Search the PCA-compressed LanceDB table for the *k* nearest neighbours
    of ``query_vec`` in the 16-d PCA space.

    Parameters
    ----------
    query_vec : sequence of 128 floats
        The original-space query vector.
    k : int
        Number of results to return.

    Returns
    -------
    list[dict]
        A JSON-serialisable list of length *k*.  Each element is a dict with
        keys ``id`` (int), ``title`` (str), and ``original_id`` (int).
    """
    projected_query = _project(query_vec)

    tbl = _get_table()

    # LanceDB vector search – returns the k closest rows by L2 distance.
    results = tbl.search(projected_query).limit(k).to_list()

    output = []
    for row in results:
        output.append({
            "id": int(row["id"]),
            "title": str(row["title"]),
            "original_id": int(row["original_id"]),
        })
    return output