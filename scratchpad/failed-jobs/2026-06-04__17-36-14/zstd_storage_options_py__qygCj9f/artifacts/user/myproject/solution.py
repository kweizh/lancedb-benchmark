import os
import json
import shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import lancedb

def compare_sizes() -> dict:
    """
    Exposes a top-level function that returns a Python dict of the shape:
    {"default_bytes": int, "zstd_bytes": int, "ratio": float}
    where ratio = zstd_bytes / default_bytes.
    
    The byte counts must be computed by walking the per-table on-disk directories
    and summing file sizes of every file under each table directory.
    """
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    db_dir = Path("/home/user/myproject/lancedb_data")
    default_dir = db_dir / f"default_{run_id}.lance"
    zstd_dir = db_dir / f"zstd_{run_id}.lance"
    
    def get_dir_size(path: Path) -> int:
        total = 0
        if not path.exists():
            return 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = Path(root) / f
                try:
                    total += fp.stat().st_size
                except OSError:
                    pass
        return total

    default_bytes = get_dir_size(default_dir)
    zstd_bytes = get_dir_size(zstd_dir)
    ratio = zstd_bytes / default_bytes if default_bytes > 0 else 0.0
    
    return {
        "default_bytes": default_bytes,
        "zstd_bytes": zstd_bytes,
        "ratio": ratio
    }

def main():
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    db_path = "/home/user/myproject/lancedb_data"
    
    # Clean up existing table directories to ensure a fresh, single-version recreate
    db_dir = Path(db_path)
    default_dir = db_dir / f"default_{run_id}.lance"
    zstd_dir = db_dir / f"zstd_{run_id}.lance"
    if default_dir.exists():
        shutil.rmtree(default_dir, ignore_errors=True)
    if zstd_dir.exists():
        shutil.rmtree(zstd_dir, ignore_errors=True)
        
    # 1. Generate deterministic vectors and compressible payload
    rng = np.random.default_rng(2026)
    embeddings = rng.standard_normal((5000, 32), dtype=np.float32).tolist()
    ids = list(range(5000))
    # highly compressible text payload: repeating a template per row
    payloads = [("This is row number {:04d} with some repeating text. ".format(i) * 20) for i in ids]
    
    # Define schemas
    schema_default = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("payload", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 32))
    ])
    
    schema_zstd = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("payload", pa.string(), metadata={
            "lance-encoding:compression": "zstd",
            "lance-encoding:compression-level": "7"
        }),
        pa.field("embedding", pa.list_(pa.float32(), 32))
    ])
    
    data_dict = {
        "id": ids,
        "payload": payloads,
        "embedding": embeddings
    }
    
    # 2. Connect to LanceDB and write both tables
    db = lancedb.connect(db_path)
    
    default_table_name = f"default_{run_id}"
    zstd_table_name = f"zstd_{run_id}"
    
    # Write default table
    t_default = pa.Table.from_pydict(data_dict, schema=schema_default)
    db.create_table(default_table_name, schema=schema_default, data=t_default, mode="overwrite")
    
    # Write zstd table
    t_zstd = pa.Table.from_pydict(data_dict, schema=schema_zstd)
    db.create_table(zstd_table_name, schema=schema_zstd, data=t_zstd, mode="overwrite")
    
    # 3. Call compare_sizes()
    sizes = compare_sizes()
    
    # 4. Write to size_report.json
    report_path = Path("/home/user/myproject/size_report.json")
    with open(report_path, "w") as f:
        json.dump(sizes, f, indent=2)
        
    # 5. Print one-line summary to stdout
    print(f"default_bytes={sizes['default_bytes']} zstd_bytes={sizes['zstd_bytes']} ratio={sizes['ratio']:.4f}")

if __name__ == "__main__":
    main()
