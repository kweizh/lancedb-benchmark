import math
import lancedb

def diff_versions(v_a: int, v_b: int) -> dict:
    """
    Given two integer version numbers from the same customers table,
    returns a JSON-serializable dictionary describing how the dataset changed
    between v_a and v_b.
    """
    # Open the LanceDB database and customers table
    db = lancedb.connect('/data/lancedb')
    tbl = db.open_table('customers')
    
    # Get non-id column names dynamically
    non_id_cols = [name for name in tbl.schema.names if name != 'id']
    
    # Retrieve row set at version v_a
    tbl.checkout(v_a)
    data_a = tbl.to_arrow().to_pylist()
    rows_a = {row['id']: {col: row[col] for col in non_id_cols} for row in data_a}
    
    # Retrieve row set at version v_b
    tbl.checkout(v_b)
    data_b = tbl.to_arrow().to_pylist()
    rows_b = {row['id']: {col: row[col] for col in non_id_cols} for row in data_b}
    
    # Helper to check if two rows differ
    def rows_differ(row_x, row_y):
        for col in non_id_cols:
            val_x = row_x[col]
            val_y = row_y[col]
            if isinstance(val_x, float) or isinstance(val_y, float):
                try:
                    f_x = float(val_x)
                    f_y = float(val_y)
                    if not math.isclose(f_x, f_y, rel_tol=1e-9, abs_tol=1e-9):
                        return True
                except (ValueError, TypeError):
                    if val_x != val_y:
                        return True
            else:
                if val_x != val_y:
                    return True
        return False

    added = []
    removed = []
    modified = []
    
    # Find removed and modified
    for id_a, row_a in rows_a.items():
        if id_a not in rows_b:
            removed.append(id_a)
        else:
            row_b = rows_b[id_a]
            if rows_differ(row_a, row_b):
                modified.append({
                    "id": id_a,
                    "old": row_a,
                    "new": row_b
                })
                
    # Find added
    for id_b in rows_b:
        if id_b not in rows_a:
            added.append(id_b)
            
    # Sort for deterministic output and ease of verification
    added.sort()
    removed.sort()
    modified.sort(key=lambda x: x['id'])
    
    return {
        "added": added,
        "removed": removed,
        "modified": modified
    }
