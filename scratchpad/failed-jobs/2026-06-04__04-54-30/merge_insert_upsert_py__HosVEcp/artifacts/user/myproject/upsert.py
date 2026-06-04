import os
import json
import lancedb
import pyarrow as pa
import numpy as np

def main():
    # 1. Connect to database
    lancedb_uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(lancedb_uri)

    # 2. Define schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("email", pa.string()),
        pa.field("score", pa.float32()),
        pa.field("vector", pa.list_(pa.float32(), 8))
    ])

    # 3. Drop existing table to ensure workflow is idempotent, then create fresh
    db.drop_table("users", ignore_missing=True)

    # 4. Generate seed data
    ids = list(range(1, 11))
    emails = [f"user_{id}@example.com" for id in ids]

    # score: cast the first 10 floats of numpy.random.default_rng(0).random(10) to float32
    rng_score = np.random.default_rng(0)
    scores = rng_score.random(10).astype(np.float32)

    # vector: a length-8 float32 array, generated as numpy.random.default_rng(100 + id).random(8).astype("float32") for each row
    vectors = [np.random.default_rng(100 + id).random(8).astype(np.float32) for id in ids]
    flat_vectors = np.concatenate(vectors)
    vector_array = pa.FixedSizeListArray.from_arrays(pa.array(flat_vectors, type=pa.float32()), 8)

    # Build Arrow Table for seed data
    seed_table = pa.Table.from_pydict({
        "id": ids,
        "email": emails,
        "score": scores,
        "vector": vector_array
    }, schema=schema)

    # Create table with seed data
    table = db.create_table("users", data=seed_table)

    # 5. Perform TWO upsert batches
    # Batch 1: UPDATE batch — incoming rows for id in {2, 5, 7} with brand-new values
    update_ids = [2, 5, 7]
    update_emails = [f"updated_{id}@example.com" for id in update_ids]
    update_scores = [np.float32(0.5 + 0.1 * id) for id in update_ids]
    update_vectors = [np.random.default_rng(200 + id).random(8).astype(np.float32) for id in update_ids]
    flat_update_vectors = np.concatenate(update_vectors)
    update_vector_array = pa.FixedSizeListArray.from_arrays(pa.array(flat_update_vectors, type=pa.float32()), 8)

    update_table = pa.Table.from_pydict({
        "id": update_ids,
        "email": update_emails,
        "score": update_scores,
        "vector": update_vector_array
    }, schema=schema)

    # Apply first merge_insert
    table.merge_insert("id") \
         .when_matched_update_all() \
         .when_not_matched_insert_all() \
         .execute(update_table)

    # Batch 2: INSERT batch — incoming rows for id in {11, 12}
    insert_ids = [11, 12]
    insert_emails = [f"new_{id}@example.com" for id in insert_ids]
    insert_scores = [np.float32(0.5 + 0.1 * id) for id in insert_ids]
    insert_vectors = [np.random.default_rng(200 + id).random(8).astype(np.float32) for id in insert_ids]
    flat_insert_vectors = np.concatenate(insert_vectors)
    insert_vector_array = pa.FixedSizeListArray.from_arrays(pa.array(flat_insert_vectors, type=pa.float32()), 8)

    insert_table = pa.Table.from_pydict({
        "id": insert_ids,
        "email": insert_emails,
        "score": insert_scores,
        "vector": insert_vector_array
    }, schema=schema)

    # Apply second merge_insert
    table.merge_insert("id") \
         .when_matched_update_all() \
         .when_not_matched_insert_all() \
         .execute(insert_table)

    # 6. Call table.count_rows() and print/log
    row_count = table.count_rows()
    print(f"Final table row count: {row_count}")

    # 7. Write the final state of selected rows to /workspace/output/upsert_state.json
    # Ensure output directory exists
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)

    # Filter to ids {1, 2, 5, 7, 10, 11, 12} and sort by id ascending
    target_ids = {1, 2, 5, 7, 10, 11, 12}
    all_rows = table.to_arrow().to_pylist()

    filtered_rows = []
    for row in all_rows:
        row_id = row["id"]
        if row_id in target_ids:
            filtered_rows.append({
                "id": int(row_id),
                "email": str(row["email"]),
                "score": round(float(row["score"]), 6)
            })

    # Sort by id ascending
    filtered_rows.sort(key=lambda x: x["id"])

    # Write to JSON
    output_path = os.path.join(output_dir, "upsert_state.json")
    with open(output_path, "w") as f:
        json.dump(filtered_rows, f, indent=2)

    print(f"Final state exported to {output_path}")

if __name__ == "__main__":
    main()
