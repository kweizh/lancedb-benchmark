"""
Compare LanceDB vector search results across L2, cosine, and dot distance metrics.
"""

import json
import os

import numpy as np
import pyarrow as pa
import lancedb


def main():
    # Connect to LanceDB
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(uri)

    # Define Arrow schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("label", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), 16)),
    ])

    # Generate deterministic data with a single RNG instance
    rng = np.random.default_rng(123)

    # Generate 32 vectors BEFORE the query vector
    vectors = rng.standard_normal(size=(32, 16)).astype("float32")

    # Generate query vector from the SAME RNG instance (after data vectors)
    query_vector = rng.standard_normal(size=(16,)).astype("float32")

    # Build the table data
    ids = list(range(32))
    labels = [f"item-{i}" for i in range(32)]

    table_data = pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "label": pa.array(labels, type=pa.utf8()),
            "vector": pa.array(
                [v.tolist() for v in vectors],
                type=pa.list_(pa.float32(), 16),
            ),
        },
        schema=schema,
    )

    # Drop existing table if present, then create fresh
    if "vectors" in db.table_names():
        db.drop_table("vectors")
    table = db.create_table("vectors", data=table_data, schema=schema)

    # Run searches for each distance metric
    results = {}
    for metric in ("l2", "cosine", "dot"):
        result_arrow = (
            table.search(query_vector)
            .distance_type(metric)
            .limit(5)
            .to_arrow()
        )
        results[metric] = [int(v) for v in result_arrow["id"].to_pylist()]

    # Ensure output directory exists
    os.makedirs("/workspace/output", exist_ok=True)

    # Write results to JSON
    output_path = "/workspace/output/distances.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
