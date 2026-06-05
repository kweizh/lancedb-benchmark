"""Ingest a Polars DataFrame into LanceDB via Arrow and expose hybrid search."""

from __future__ import annotations

import json
import os

import numpy as np
import polars as pl
import lancedb


# ---------------------------------------------------------------------------
# 1. build_dataframe
# ---------------------------------------------------------------------------

def build_dataframe() -> pl.DataFrame:
    """Return a deterministic 500-row Polars DataFrame.

    Columns (in order): id, title, score, tag, vector.
    """
    rng = np.random.default_rng(2026)

    ids = np.arange(500, dtype=np.int64)
    titles = [f"item-{i}" for i in ids]
    scores = rng.uniform(0.0, 1.0, size=500)
    tags = rng.choice(["alpha", "beta", "gamma", "delta"], size=500)
    vectors = rng.standard_normal((500, 32)).astype(np.float32)

    df = pl.DataFrame(
        {
            "id": ids,
            "title": titles,
            "score": scores,
            "tag": tags,
            "vector": [v.tolist() for v in vectors],
        },
        schema={
            "id": pl.Int64,
            "title": pl.Utf8,
            "score": pl.Float64,
            "tag": pl.Utf8,
            "vector": pl.List(pl.Float32),
        },
    )
    return df


# ---------------------------------------------------------------------------
# 2. ingest
# ---------------------------------------------------------------------------

def ingest(df: pl.DataFrame, db_uri: str, table_name: str) -> lancedb.table.Table:
    """Convert *df* to Arrow and create (overwrite) a LanceDB table.

    Returns the created LanceDB Table handle.
    """
    arrow_table = df.to_arrow()
    db = lancedb.connect(db_uri)
    table = db.create_table(table_name, data=arrow_table, mode="overwrite")
    return table


# ---------------------------------------------------------------------------
# 3. search
# ---------------------------------------------------------------------------

def search(
    table,
    vec: list[float] | np.ndarray,
    top_k: int = 10,
    min_score: float = 0.5,
    tag: str = "alpha",
) -> list[dict]:
    """Hybrid vector + SQL search on a LanceDB table.

    Returns at most *top_k* rows as dicts with keys
    ``id, title, score, tag, _distance``, ordered by ascending ``_distance``.
    Every returned row satisfies ``score >= min_score AND tag == tag``.
    """
    where_clause = f"score >= {min_score} AND tag = '{tag}'"
    results = (
        table.search(vec)
        .where(where_clause)
        .limit(top_k)
        .to_list()
    )
    # Ensure consistent key ordering and required keys
    out: list[dict] = []
    for row in results:
        out.append(
            {
                "id": row["id"],
                "title": row["title"],
                "score": row["score"],
                "tag": row["tag"],
                "_distance": row["_distance"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# 4. CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_id = os.environ["ZEALT_RUN_ID"]
    db_uri = "/home/user/myproject/lancedb"
    table_name = f"polars_ingest_{run_id}"

    df = build_dataframe()
    table = ingest(df, db_uri, table_name)

    vec = np.zeros(32, dtype=np.float32)
    results = search(table, vec, top_k=10, min_score=0.5, tag="alpha")

    output = json.dumps(results)
    print(output)

    with open("/home/user/myproject/output.log", "a") as fh:
        fh.write(output + "\n")