import json
import os

import lancedb
import numpy as np
import polars as pl
import pyarrow as pa


def build_dataframe() -> pl.DataFrame:
    """Build a deterministic 500-row polars DataFrame."""
    rng = np.random.default_rng(2026)

    ids = list(range(500))
    titles = [f"item-{i}" for i in ids]
    scores = rng.uniform(0.0, 1.0, size=500)
    tags = rng.choice(["alpha", "beta", "gamma", "delta"], size=500)
    vectors = rng.standard_normal((500, 32)).astype(np.float32)

    df = pl.DataFrame(
        {
            "id": pl.Series(ids, dtype=pl.Int64),
            "title": pl.Series(titles, dtype=pl.Utf8),
            "score": pl.Series(scores, dtype=pl.Float64),
            "tag": pl.Series(tags, dtype=pl.Utf8),
            "vector": pl.Series(vectors.tolist(), dtype=pl.List(pl.Float32)),
        }
    )
    return df


def ingest(df: pl.DataFrame, db_uri: str, table_name: str) -> lancedb.table.Table:
    """Convert polars DataFrame to Arrow and ingest into LanceDB."""
    arrow_table: pa.Table = df.to_arrow()
    db = lancedb.connect(db_uri)
    table = db.create_table(table_name, data=arrow_table, mode="overwrite")
    return table


def search(
    table,
    vec,
    top_k: int = 10,
    min_score: float = 0.5,
    tag: str = "alpha",
) -> list[dict]:
    """Vector search with SQL filter predicates on score and tag."""
    where_clause = f"score >= {min_score} AND tag = '{tag}'"
    results = (
        table.search(vec)
        .where(where_clause)
        .limit(top_k)
        .to_list()
    )
    # Keep only the required keys, ordered by ascending _distance
    output = []
    for row in results:
        output.append(
            {
                "id": row["id"],
                "title": row["title"],
                "score": row["score"],
                "tag": row["tag"],
                "_distance": row["_distance"],
            }
        )
    # Results from LanceDB are already ordered by ascending _distance,
    # but sort explicitly to guarantee the contract.
    output.sort(key=lambda r: r["_distance"])
    return output


if __name__ == "__main__":
    run_id = os.environ["ZEALT_RUN_ID"]
    db_uri = "/home/user/myproject/lancedb"
    table_name = f"polars_ingest_{run_id}"

    df = build_dataframe()
    table = ingest(df, db_uri, table_name)

    vec = np.zeros(32, dtype=np.float32)
    results = search(table, vec, top_k=10, min_score=0.5, tag="alpha")

    line = json.dumps(results)
    print(line)

    log_path = "/home/user/myproject/output.log"
    with open(log_path, "a") as f:
        f.write(line + "\n")
