import os
import lancedb
import pyarrow as pa
import numpy as np

def build_snapshots(db_path: str, table_name: str) -> None:
    db = lancedb.connect(db_path)
    
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 16))
    ])
    
    rng = np.random.default_rng(seed=2026)
    
    # Seed phase
    ids_1 = list(range(0, 50))
    vectors_1 = rng.random((50, 16), dtype=np.float32)
    texts_1 = [f"doc-{i}" for i in ids_1]
    
    data_1 = pa.Table.from_arrays(
        [
            pa.array(ids_1, type=pa.int64()),
            pa.array(texts_1, type=pa.string()),
            pa.array(vectors_1.tolist(), type=pa.list_(pa.float32(), 16))
        ],
        schema=schema
    )
    
    db.drop_table(table_name, ignore_missing=True)
    table = db.create_table(table_name, data=data_1, schema=schema)
    table.tags.create("v1_baseline", table.version)
    
    # Extend phase
    ids_2 = list(range(50, 70))
    vectors_2 = rng.random((20, 16), dtype=np.float32)
    texts_2 = [f"doc-{i}" for i in ids_2]
    
    data_2 = pa.Table.from_arrays(
        [
            pa.array(ids_2, type=pa.int64()),
            pa.array(texts_2, type=pa.string()),
            pa.array(vectors_2.tolist(), type=pa.list_(pa.float32(), 16))
        ],
        schema=schema
    )
    
    table.add(data_2)
    table.tags.create("v2_extended", table.version)
    
    # Prune phase
    table.delete("id < 5")
    table.tags.create("v3_pruned", table.version)


def diff(db_path: str, table_name: str, tag_a: str, tag_b: str) -> dict:
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    
    table.checkout(tag_a)
    ids_a = set(table.search().to_arrow().column("id").to_pylist())
    
    table.checkout(tag_b)
    ids_b = set(table.search().to_arrow().column("id").to_pylist())
    
    table.checkout_latest()
    
    added_ids = sorted(list(ids_b - ids_a))
    removed_ids = sorted(list(ids_a - ids_b))
    common_count = len(ids_a.intersection(ids_b))
    
    return {
        "added_ids": added_ids,
        "removed_ids": removed_ids,
        "common_count": common_count
    }

if __name__ == "__main__":
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    db_path = "/app/db"
    table_name = f"documents_{run_id}"
    build_snapshots(db_path, table_name)
