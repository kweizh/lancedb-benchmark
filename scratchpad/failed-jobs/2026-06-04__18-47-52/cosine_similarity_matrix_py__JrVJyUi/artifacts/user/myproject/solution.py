import os
import lancedb
import numpy as np

_cached_matrix = None
_cached_labels = None

def _compute_matrix_and_labels():
    global _cached_matrix, _cached_labels
    if _cached_matrix is not None and _cached_labels is not None:
        return _cached_matrix, _cached_labels

    uri = os.environ.get('LANCEDB_URI')
    table_name = os.environ.get('LANCEDB_TABLE')
    
    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    
    df = table.to_pandas()
    
    n = len(df)
    
    labels = np.zeros(n, dtype=int)
    for _, row in df.iterrows():
        labels[int(row['id'])] = int(row['label'])
        
    _cached_labels = labels
    
    _cached_matrix = np.zeros((n, n), dtype=np.float64)
    
    for _, row in df.iterrows():
        row_id = int(row['id'])
        vector = row['vector']
        
        res = table.search(vector).metric("cosine").limit(n).to_pandas()
        
        for _, res_row in res.iterrows():
            res_id = int(res_row['id'])
            distance = float(res_row['_distance'])
            similarity = 1.0 - distance
            # Clamp to [-1.0, 1.0] to be safe
            similarity = max(-1.0, min(1.0, similarity))
            _cached_matrix[row_id, res_id] = similarity
            
    for i in range(n):
        _cached_matrix[i, i] = 1.0
        
    return _cached_matrix, _cached_labels

def similarity_matrix() -> np.ndarray:
    matrix, _ = _compute_matrix_and_labels()
    return matrix

def intra_class_mean(label: int) -> float:
    matrix, labels = _compute_matrix_and_labels()
    
    indices = np.where(labels == label)[0]
    if len(indices) <= 1:
        return 0.0
        
    submatrix = matrix[np.ix_(indices, indices)]
    mask = ~np.eye(len(indices), dtype=bool)
    
    return float(np.mean(submatrix[mask]))
