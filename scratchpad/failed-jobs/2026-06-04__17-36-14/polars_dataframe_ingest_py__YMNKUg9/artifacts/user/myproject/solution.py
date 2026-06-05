import os
import json
import numpy as np
import polars as pl
import pyarrow as pa
import lancedb

def build_dataframe() -> pl.DataFrame:
    """
    Returns a polars DataFrame with exactly 500 rows generated deterministically 
    with numpy.random.default_rng(2026).
    """
    rng = np.random.default_rng(2026)
    ids = list(range(500))
    titles = [f"item-{i}" for i in ids]
    scores = rng.uniform(0.0, 1.0, size=500)
    tags = rng.choice(["alpha", "beta", "gamma", "delta"], size=500)
    vectors = rng.standard_normal((500, 32)).astype(np.float32)
    
    # Convert vectors to a list of lists of floats
    vectors_list = vectors.tolist()
    
    df = pl.DataFrame({
        "id": ids,
        "title": titles,
        "score": scores,
        "tag": tags,
        "vector": vectors_list
    }, schema={
        "id": pl.Int64,
        "title": pl.Utf8,
        "score": pl.Float64,
        "tag": pl.Utf8,
        "vector": pl.List(pl.Float32)
    })
    return df

def ingest(df: pl.DataFrame, db_uri: str, table_name: str) -> lancedb.table.Table:
    """
    Convert the polars DataFrame to a pyarrow.Table using df.to_arrow().
    Open a LanceDB connection at db_uri and create the table from that Arrow table 
    (overwrite any pre-existing table).
    The resulting Arrow schema must keep the original column order and the vector column 
    must be a 32-wide list of float32.
    Return the created lancedb.table.Table.
    """
    arrow_table = df.to_arrow()
    
    # Explicitly cast the vector column to a 32-wide fixed_size_list of float32
    vector_index = arrow_table.schema.get_field_index("vector")
    new_type = pa.list_(pa.float32(), 32)
    casted_column = arrow_table.column("vector").cast(new_type)
    arrow_table = arrow_table.set_column(vector_index, pa.field("vector", new_type), casted_column)
    
    db = lancedb.connect(db_uri)
    table = db.create_table(table_name, data=arrow_table, mode="overwrite")
    return table

def search(table, vec, top_k=10, min_score=0.5, tag="alpha") -> list[dict]:
    """
    vec is a 32-dim list[float] / numpy.ndarray.
    Run a vector search on the LanceDB table that also applies the SQL where clause 
    score >= <min_score> AND tag = '<tag>'.
    Return at most top_k matches as a list of dicts with the keys id, title, score, tag, and _distance, 
    ordered by ascending _distance.
    Every returned row MUST satisfy both filter predicates.
    """
    # Escape single quotes in tag to prevent SQL injection/syntax errors
    escaped_tag = str(tag).replace("'", "''")
    where_clause = f"score >= {min_score} AND tag = '{escaped_tag}'"
    
    raw_results = table.search(vec).where(where_clause).limit(top_k).to_list()
    
    # Process and return with the exact requested keys, ordered by ascending _distance
    results = []
    for r in raw_results:
        results.append({
            "id": r["id"],
            "title": r["title"],
            "score": r["score"],
            "tag": r["tag"],
            "_distance": r["_distance"]
        })
    
    # Ensure they are sorted by _distance ascending
    results.sort(key=lambda x: x["_distance"])
    return results

if __name__ == "__main__":
    # CLI entrypoint
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        raise ValueError("ZEALT_RUN_ID environment variable is not set")
    
    db_uri = "/home/user/myproject/lancedb"
    table_name = f"polars_ingest_{run_id}"
    
    # Build
    df = build_dataframe()
    
    # Ingest
    table = ingest(df, db_uri, table_name)
    
    # Search
    vec = np.zeros(32, dtype=np.float32)
    demo_results = search(table, vec, top_k=10, min_score=0.5, tag="alpha")
    
    # Print to stdout and log file
    json_line = json.dumps(demo_results)
    print(json_line)
    
    log_dir = "/home/user/myproject"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "output.log")
    with open(log_path, "a") as f:
        f.write(json_line + "\n")
