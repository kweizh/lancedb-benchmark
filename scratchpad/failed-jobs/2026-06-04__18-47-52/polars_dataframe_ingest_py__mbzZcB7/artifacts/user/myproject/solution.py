import os
import json
import numpy as np
import polars as pl
import pyarrow as pa
import lancedb

def build_dataframe() -> pl.DataFrame:
    rng = np.random.default_rng(2026)
    
    ids = np.arange(500, dtype=np.int64)
    titles = [f"item-{i}" for i in ids]
    scores = rng.uniform(0.0, 1.0, size=500)
    tags = rng.choice(["alpha", "beta", "gamma", "delta"], size=500)
    vectors = rng.standard_normal((500, 32)).astype(np.float32)
    
    df = pl.DataFrame({
        "id": pl.Series(ids, dtype=pl.Int64),
        "title": pl.Series(titles, dtype=pl.Utf8),
        "score": pl.Series(scores, dtype=pl.Float64),
        "tag": pl.Series(tags, dtype=pl.Utf8),
        "vector": pl.Series(vectors.tolist(), dtype=pl.List(pl.Float32))
    })
    
    return df

def ingest(df: pl.DataFrame, db_uri: str, table_name: str):
    arrow_table = df.to_arrow()
    db = lancedb.connect(db_uri)
    table = db.create_table(table_name, data=arrow_table, mode="overwrite")
    return table

def search(table, vec, top_k=10, min_score=0.5, tag="alpha") -> list[dict]:
    res = table.search(vec).where(f"score >= {min_score} AND tag = '{tag}'").limit(top_k).to_list()
    
    out = []
    for row in res:
        out.append({
            "id": row["id"],
            "title": row["title"],
            "score": row["score"],
            "tag": row["tag"],
            "_distance": row["_distance"]
        })
    return out

if __name__ == "__main__":
    run_id = os.environ["ZEALT_RUN_ID"]
    db_uri = "/home/user/myproject/lancedb"
    table_name = f"polars_ingest_{run_id}"
    
    df = build_dataframe()
    table = ingest(df, db_uri, table_name)
    
    vec = np.zeros(32, dtype=np.float32)
    res = search(table, vec, top_k=10, min_score=0.5, tag="alpha")
    
    res_json = json.dumps(res)
    print(res_json)
    
    with open("/home/user/myproject/output.log", "a") as f:
        f.write(res_json + "\n")
