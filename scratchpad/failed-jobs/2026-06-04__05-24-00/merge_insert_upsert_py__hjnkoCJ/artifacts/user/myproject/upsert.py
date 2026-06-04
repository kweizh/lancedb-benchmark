#!/usr/bin/env python3
"""LanceDB merge_insert upsert workflow."""

import json
import os

import lancedb
import numpy as np
import pyarrow as pa

# --- Configuration ---
URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "users"
OUTPUT_DIR = "/workspace/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "upsert_state.json")

# --- Schema definition ---
schema = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("email", pa.string()),
        pa.field("score", pa.float32()),
        pa.field("vector", pa.list_(pa.float32(), 8)),
    ]
)


def make_vector(rng_seed: int) -> pa.FixedSizeListArray:
    """Generate a length-8 float32 vector from a given RNG seed."""
    flat = np.random.default_rng(rng_seed).random(8).astype("float32")
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), 8)


def make_batch_rows(ids: list[int], email_fn, score_fn, vector_seed_offset: int) -> pa.Table:
    """Build a pyarrow Table for a batch of rows."""
    ids_arr = pa.array(ids, type=pa.int64())
    emails_arr = pa.array([email_fn(i) for i in ids], type=pa.string())
    scores_arr = pa.array([score_fn(i) for i in ids], type=pa.float32())
    vectors_arr = pa.FixedSizeListArray.from_arrays(
        pa.array(
            np.concatenate(
                [np.random.default_rng(vector_seed_offset + i).random(8).astype("float32") for i in ids]
            ),
            type=pa.float32(),
        ),
        8,
    )
    return pa.Table.from_arrays(
        [ids_arr, emails_arr, scores_arr, vectors_arr],
        schema=schema,
    )


def main():
    # --- Connect and create table ---
    db = lancedb.connect(URI)
    db.drop_table(TABLE_NAME, ignore_missing=True)

    # Seed data: id 1..10
    seed_ids = list(range(1, 11))
    seed_scores = np.random.default_rng(0).random(10).astype("float32")

    seed_rows = {
        "id": pa.array(seed_ids, type=pa.int64()),
        "email": pa.array([f"user_{i}@example.com" for i in seed_ids], type=pa.string()),
        "score": pa.array([float(seed_scores[i - 1]) for i in seed_ids], type=pa.float32()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(
                np.concatenate(
                    [np.random.default_rng(100 + i).random(8).astype("float32") for i in seed_ids]
                ),
                type=pa.float32(),
            ),
            8,
        ),
    }
    seed_table = pa.Table.from_pydict(seed_rows, schema=schema)

    table = db.create_table(TABLE_NAME, data=seed_table, mode="overwrite")

    # --- Upsert batch 1: UPDATE rows (id 2, 5, 7) ---
    update_ids = [2, 5, 7]
    update_rows = {
        "id": pa.array(update_ids, type=pa.int64()),
        "email": pa.array([f"updated_{i}@example.com" for i in update_ids], type=pa.string()),
        "score": pa.array([0.5 + 0.1 * i for i in update_ids], type=pa.float32()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(
                np.concatenate(
                    [np.random.default_rng(200 + i).random(8).astype("float32") for i in update_ids]
                ),
                type=pa.float32(),
            ),
            8,
        ),
    }
    update_table = pa.Table.from_pydict(update_rows, schema=schema)

    table.merge_insert("id") \
        .when_matched_update_all() \
        .when_not_matched_insert_all() \
        .execute(update_table)

    # --- Upsert batch 2: INSERT rows (id 11, 12) ---
    insert_ids = [11, 12]
    insert_rows = {
        "id": pa.array(insert_ids, type=pa.int64()),
        "email": pa.array([f"new_{i}@example.com" for i in insert_ids], type=pa.string()),
        "score": pa.array([0.5 + 0.1 * i for i in insert_ids], type=pa.float32()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(
                np.concatenate(
                    [np.random.default_rng(200 + i).random(8).astype("float32") for i in insert_ids]
                ),
                type=pa.float32(),
            ),
            8,
        ),
    }
    insert_table = pa.Table.from_pydict(insert_rows, schema=schema)

    table.merge_insert("id") \
        .when_matched_update_all() \
        .when_not_matched_insert_all() \
        .execute(insert_table)

    # --- Verify row count ---
    row_count = table.count_rows()
    print(f"Row count: {row_count}")
    assert row_count == 12, f"Expected 12 rows, got {row_count}"

    # --- Export filtered results ---
    filter_ids = {1, 2, 5, 7, 10, 11, 12}
    result = table.to_pandas()
    filtered = result[result["id"].isin(filter_ids)].sort_values("id")

    output = []
    for _, row in filtered.iterrows():
        output.append({
            "id": int(row["id"]),
            "email": str(row["email"]),
            "score": round(float(row["score"]), 10),
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Output written to {OUTPUT_FILE}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()