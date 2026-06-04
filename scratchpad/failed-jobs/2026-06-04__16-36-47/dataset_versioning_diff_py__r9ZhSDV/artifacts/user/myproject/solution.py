import lancedb
import math
import pandas as pd

def diff_versions(v_a: int, v_b: int) -> dict:
    db = lancedb.connect("/data/lancedb")
    table = db.open_table("customers")
    
    table.checkout(v_a)
    df_a = table.to_pandas()
    
    table.checkout(v_b)
    df_b = table.to_pandas()
    
    records_a = df_a.to_dict('records')
    records_b = df_b.to_dict('records')
    
    dict_a = {int(r['id']): r for r in records_a}
    dict_b = {int(r['id']): r for r in records_b}
    
    ids_a = set(dict_a.keys())
    ids_b = set(dict_b.keys())
    
    added = list(ids_b - ids_a)
    removed = list(ids_a - ids_b)
    
    modified = []
    for i in ids_a & ids_b:
        ra = dict_a[i]
        rb = dict_b[i]
        
        is_diff = False
        
        name_a = str(ra['name']) if pd.notna(ra['name']) else None
        price_a = float(ra['price']) if pd.notna(ra['price']) else None
        cat_a = str(ra['category']) if pd.notna(ra['category']) else None
        
        name_b = str(rb['name']) if pd.notna(rb['name']) else None
        price_b = float(rb['price']) if pd.notna(rb['price']) else None
        cat_b = str(rb['category']) if pd.notna(rb['category']) else None
        
        if name_a != name_b:
            is_diff = True
        elif cat_a != cat_b:
            is_diff = True
        elif price_a is not None and price_b is not None:
            if not math.isclose(price_a, price_b, rel_tol=1e-5, abs_tol=1e-8):
                is_diff = True
        elif price_a != price_b:
            is_diff = True
            
        if is_diff:
            modified.append({
                "id": i,
                "old": {"name": name_a, "price": price_a, "category": cat_a},
                "new": {"name": name_b, "price": price_b, "category": cat_b}
            })
            
    return {
        "added": added,
        "removed": removed,
        "modified": modified
    }
