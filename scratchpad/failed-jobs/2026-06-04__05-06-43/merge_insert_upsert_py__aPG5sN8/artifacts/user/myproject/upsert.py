import os
import json
import numpy as np
import pyarrow as pa
import lancedb

def main():
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(uri)

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("email", pa.string()),
        pa.field("score", pa.float32()),
        pa.field("vector", pa.list_(pa.float32(), 8))
    ])

    # Seed data
    ids = list(range(1, 11))
    emails = [f"user_{i}@example.com" for i in ids]
    scores = np.random.default_rng(0).random(10).astype(np.float32)

    vectors = []
    for i in ids:
        v = np.random.default_rng(100 + i).random(8).astype(np.float32)
        vectors.append(v)
    
    flat_vectors = np.concatenate(vectors)
    vector_array = pa.FixedSizeListArray.from_arrays(pa.array(flat_vectors, type=pa.float32()), 8)

    seed_table = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array(emails, type=pa.string()),
            pa.array(scores, type=pa.float32()),
            vector_array
        ],
        schema=schema
    )

    db.drop_table("users", ignore_missing=True)
    table = db.create_table("users", data=seed_table)

    # 1. Update batch
    update_ids = [2, 5, 7]
    update_emails = [f"updated_{i}@example.com" for i in update_ids]
    update_scores = [np.float32(0.5 + 0.1 * i) for i in update_ids]
    
    update_vectors = []
    for i in update_ids:
        v = np.random.default_rng(200 + i).random(8).astype(np.float32)
        update_vectors.append(v)
    
    update_flat_vectors = np.concatenate(update_vectors)
    update_vector_array = pa.FixedSizeListArray.from_arrays(pa.array(update_flat_vectors, type=pa.float32()), 8)

    update_table = pa.Table.from_arrays(
        [
            pa.array(update_ids, type=pa.int64()),
            pa.array(update_emails, type=pa.string()),
            pa.array(update_scores, type=pa.float32()),
            update_vector_array
        ],
        schema=schema
    )

    table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(update_table)

    # 2. Insert batch
    insert_ids = [11, 12]
    insert_emails = [f"new_{i}@example.com" for i in insert_ids]
    insert_scores = [np.float32(0.5 + 0.1 * i) for i in insert_ids]
    
    insert_vectors = []
    for i in insert_ids:
        v = np.random.default_rng(200 + i).random(8).astype(np.float32)
        insert_vectors.append(v)
    
    insert_flat_vectors = np.concatenate(insert_vectors)
    insert_vector_array = pa.FixedSizeListArray.from_arrays(pa.array(insert_flat_vectors, type=pa.float32()), 8)

    insert_table = pa.Table.from_arrays(
        [
            pa.array(insert_ids, type=pa.int64()),
            pa.array(insert_emails, type=pa.string()),
            pa.array(insert_scores, type=pa.float32()),
            insert_vector_array
        ],
        schema=schema
    )

    table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(insert_table)

    count = table.count_rows()
    print(f"Total rows: {count}")

    # Export state
    target_ids = {1, 2, 5, 7, 10, 11, 12}
    arrow_table = table.to_arrow()
    data = arrow_table.to_pylist()
    
    filtered_data = [d for d in data if d["id"] in target_ids]
    filtered_data.sort(key=lambda x: x["id"])

    output_data = []
    for d in filtered_data:
        output_data.append({
            "id": d["id"],
            "email": d["email"],
            "score": float(d["score"])
        })

    os.makedirs("/workspace/output", exist_ok=True)
    with open("/workspace/output/upsert_state.json", "w") as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    main()
