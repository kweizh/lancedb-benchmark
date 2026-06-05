"""
Ingestion utility: Polars DataFrame -> Arrow -> LanceDB
"""

import json
import os

import lancedb
import numpy as np
import polars as pl
import pyarrow as pa


# ---------------------------------------------------------------------------
# 1. Build the source DataFrame
# ---------------------------------------------------------------------------

def build_dataframe() -> pl.DataFrame:
    """Return a deterministic 500-row Polars DataFrame."""
    rng = np.random.default_rng(2026)

    ids = np.arange(500, dtype=np.int64)
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


# ---------------------------------------------------------------------------
# 2. Ingest into LanceDB via Arrow
# ---------------------------------------------------------------------------

def ingest(df: pl.DataFrame, db_uri: str, table_name: str) -> lancedb.table.Table:
    """
    Convert *df* to a PyArrow Table and persist it into LanceDB.

    Column order is preserved; the ``vector`` column becomes an Arrow
    ``list<float32>`` field that LanceDB treats as a 32-d vector column.
    """
    arrow_table: pa.Table = df.to_arrow()

    db = lancedb.connect(db_uri)
    table = db.create_table(table_name, data=arrow_table, mode="overwrite")
    return table


# ---------------------------------------------------------------------------
# 3. Hybrid vector + SQL search
# ---------------------------------------------------------------------------

def search(
    table,
    vec,
    top_k: int = 10,
    min_score: float = 0.5,
    tag: str = "alpha",
) -> list[dict]:
    """
    Vector-similarity search with SQL filter predicates.

    Parameters
    ----------
    table    : lancedb.table.Table
    vec      : 32-dim list[float] or numpy.ndarray
    top_k    : maximum number of results
    min_score: lower bound on the ``score`` column (inclusive)
    tag      : exact match on the ``tag`` column

    Returns
    -------
    list of dicts with keys ``id, title, score, tag, _distance``,
    sorted by ascending ``_distance``.
    """
    where_clause = f"score >= {min_score} AND tag = '{tag}'"

    rows = (
        table.search(vec)
        .where(where_clause)
        .limit(top_k)
        .to_list()
    )

    results = [
        {
            "id": int(row["id"]),
            "title": row["title"],
            "score": float(row["score"]),
            "tag": row["tag"],
            "_distance": float(row["_distance"]),
        }
        for row in rows
    ]

    # Guarantee ascending _distance order (LanceDB already returns this,
    # but be explicit for correctness).
    results.sort(key=lambda r: r["_distance"])
    return results


# ---------------------------------------------------------------------------
# 4. CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    run_id = os.environ["ZEALT_RUN_ID"]

    db_uri = "/home/user/myproject/lancedb"
    table_name = f"polars_ingest_{run_id}"

    df = build_dataframe()
    table = ingest(df, db_uri, table_name)

    demo_vec = np.zeros(32, dtype=np.float32)
    results = search(table, demo_vec, top_k=10, min_score=0.5, tag="alpha")

    json_line = json.dumps(results)

    print(json_line)

    log_path = "/home/user/myproject/output.log"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json_line + "\n")


if __name__ == "__main__":
    main()
