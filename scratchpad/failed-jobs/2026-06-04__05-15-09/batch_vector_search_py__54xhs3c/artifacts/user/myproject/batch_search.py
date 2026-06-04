"""
Batch Vector Search with LanceDB
Connects to a LanceDB database, creates and seeds a table with deterministic
vectors, runs a batch vector search for 5 query vectors, and writes the
per-query top-3 neighbor ids to a JSON file.
"""

import json
import os

import numpy as np
import pyarrow as pa
import lancedb


def main():
    # --- Configuration ---
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    output_path = "/workspace/output/batch_search.json"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Connect to LanceDB ---
    db = lancedb.connect(uri)

    # --- Build Arrow schema ---
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 12)),
    ])

    # --- Seed data with a single deterministic RNG ---
    rng = np.random.default_rng(33)

    # Draw 64 rows (draws 0..63 from the RNG)
    rows = []
    for i in range(64):
        vec = rng.random(12, dtype=np.float32)
        rows.append({
            "id": i,
            "name": f"item-{i}",
            "vector": vec.tolist(),
        })

    # --- Create (or overwrite) the table ---
    if "items" in db.table_names():
        db.drop_table("items")

    tbl = db.create_table("items", schema=schema)
    tbl.add(rows)

    print(f"Table 'items' seeded with {tbl.count_rows()} rows.")

    # --- Build 5 query vectors (draws 64..68 from the same RNG) ---
    query_vectors = np.stack(
        [rng.random(12, dtype=np.float32) for _ in range(5)],
        axis=0,
    )  # shape (5, 12), dtype float32

    print(f"Query vectors shape: {query_vectors.shape}, dtype: {query_vectors.dtype}")

    # --- Batched search ---
    results_per_query = []

    try:
        # Attempt batched 2-D search (lancedb >= some version returns query_index)
        result_df = tbl.search(query_vectors).limit(3).to_pandas()

        if "query_index" in result_df.columns:
            print("Using batched search with query_index column.")
            for qi in range(5):
                group = result_df[result_df["query_index"] == qi]
                ids = group["id"].tolist()
                results_per_query.append(ids)
        else:
            raise ValueError("query_index column not present — falling back to per-query loop.")

    except Exception as e:
        print(f"Batched search unavailable ({e}); falling back to per-query loop.")
        for q in query_vectors:
            rows_found = tbl.search(q).limit(3).to_list()
            ids = [r["id"] for r in rows_found]
            results_per_query.append(ids)

    # Ensure each inner list has exactly 3 ids (cast to plain Python int)
    final_results = [[int(x) for x in group[:3]] for group in results_per_query]

    print("Results per query:", final_results)

    # --- Write JSON output ---
    output = {"results": final_results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
