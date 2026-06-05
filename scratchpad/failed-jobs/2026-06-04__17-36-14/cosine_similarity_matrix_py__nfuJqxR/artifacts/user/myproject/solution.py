import os
import lancedb
import numpy as np

_cached_matrix = None
_cached_labels = None
_cached_key = None

def _load_data():
    uri = os.environ.get("LANCEDB_URI", "/home/user/myproject/lancedb_data")
    tbl_name = os.environ.get("LANCEDB_TABLE", "vectors")
    db = lancedb.connect(uri)
    tbl = db.open_table(tbl_name)
    return tbl, uri, tbl_name

def similarity_matrix() -> np.ndarray:
    """
    Returns a (200, 200) float matrix S such that S[i, j] is the cosine similarity
    between the stored vectors of the rows with id == i and id == j.
    The similarity values MUST be derived from LanceDB cosine search results.
    """
    global _cached_matrix, _cached_labels, _cached_key
    
    tbl, uri, tbl_name = _load_data()
    version = getattr(tbl, "version", None)
    try:
        count_rows = tbl.count_rows()
    except Exception:
        count_rows = None
        
    current_key = (uri, tbl_name, version, count_rows)
    
    if _cached_matrix is not None and _cached_key == current_key:
        return _cached_matrix

    df = tbl.to_pandas()
    
    # Ensure df is sorted by id so that we map correctly to 0..199
    df = df.sort_values("id").reset_index(drop=True)
    
    num_rows = len(df)
    S = np.zeros((num_rows, num_rows), dtype=np.float64)
    
    # Cache the labels for intra_class_mean
    _cached_labels = df["label"].values
    
    for i in range(num_rows):
        vector_i = df.loc[i, "vector"]
        # Query LanceDB search
        res = tbl.search(vector_i).distance_type("cosine").limit(num_rows).to_pandas()
        for _, row in res.iterrows():
            j = int(row["id"])
            distance = float(row["_distance"])
            similarity = 1.0 - distance
            S[i, j] = similarity
        # Ensure diagonal is exactly 1.0
        S[i, i] = 1.0
        
    _cached_matrix = S
    _cached_key = current_key
    return S

def intra_class_mean(label: int) -> float:
    """
    Returns the mean off-diagonal cosine similarity restricted to the rows
    whose label column equals label.
    """
    global _cached_matrix, _cached_labels
    # Calling similarity_matrix() ensures S and labels are loaded and cached
    S = similarity_matrix()
        
    # Find indices where label matches
    indices = np.where(_cached_labels == label)[0]
    if len(indices) <= 1:
        return 0.0
        
    submatrix = S[np.ix_(indices, indices)]
    n = len(indices)
    mask = ~np.eye(n, dtype=bool)
    return float(submatrix[mask].mean())
