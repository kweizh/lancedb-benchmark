import os
import json
import pyarrow as pa
import numpy as np
import lancedb
from pathlib import Path

def get_run_id():
    return os.environ.get("ZEALT_RUN_ID", "default")

def get_table_names():
    run_id = get_run_id()
    return f"default_{run_id}", f"zstd_{run_id}"

def compare_sizes():
    run_id = get_run_id()
    default_name, zstd_name = get_table_names()
    
    base_dir = Path("/home/user/myproject/lancedb_data")
    default_dir = base_dir / f"{default_name}.lance"
    zstd_dir = base_dir / f"{zstd_name}.lance"
    
    def get_dir_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    default_bytes = get_dir_size(default_dir)
    zstd_bytes = get_dir_size(zstd_dir)
    ratio = zstd_bytes / default_bytes if default_bytes > 0 else 0.0
    
    return {
        "default_bytes": default_bytes,
        "zstd_bytes": zstd_bytes,
        "ratio": ratio
    }

def main():
    run_id = get_run_id()
    default_name, zstd_name = get_table_names()
    
    # Generate data
    num_rows = 5000
    rng = np.random.default_rng(2026)
    embeddings = rng.random((num_rows, 32)).astype(np.float32)
    
    # Compressible payload
    base_payload = "This is a highly compressible repeating string payload. " * 50
    
    ids = list(range(num_rows))
    payloads = [f"ID:{i} " + base_payload for i in range(num_rows)]
    
    # Default schema
    schema_default = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("payload", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 32))
    ])
    
    # Zstd schema
    schema_zstd = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("payload", pa.string(), metadata={"lance-encoding:compression": "zstd", "lance-encoding:compression-level": "3"}),
        pa.field("embedding", pa.list_(pa.float32(), 32))
    ])
    
    emb_array = pa.FixedSizeListArray.from_arrays(embeddings.flatten(), 32)
    
    table_default = pa.Table.from_arrays(
        [pa.array(ids), pa.array(payloads), emb_array],
        schema=schema_default
    )
    
    table_zstd = pa.Table.from_arrays(
        [pa.array(ids), pa.array(payloads), emb_array],
        schema=schema_zstd
    )
    
    db = lancedb.connect("/home/user/myproject/lancedb_data")
    
    db.create_table(default_name, data=table_default, mode="overwrite")
    db.create_table(zstd_name, data=table_zstd, mode="overwrite")
    
    sizes = compare_sizes()
    
    with open("/home/user/myproject/size_report.json", "w") as f:
        json.dump(sizes, f)
        
    print(f"default_bytes={sizes['default_bytes']} zstd_bytes={sizes['zstd_bytes']} ratio={sizes['ratio']:.4f}")

if __name__ == "__main__":
    main()
