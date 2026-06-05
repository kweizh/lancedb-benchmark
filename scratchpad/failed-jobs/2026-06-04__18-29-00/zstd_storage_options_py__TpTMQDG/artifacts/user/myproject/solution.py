#!/usr/bin/env python3
"""LanceDB Zstd Column Compression Sizing comparison."""

import json
import os
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa


DB_PATH = "/home/user/myproject/lancedb_data"
NUM_ROWS = 5000
VECTOR_DIM = 32
RNG_SEED = 2026


def _get_run_id() -> str:
    return os.environ.get("ZEALT_RUN_ID", "default")


def _generate_payloads(n: int) -> list[str]:
    """Generate compressible text payloads — a repeating template per row."""
    payloads = []
    for i in range(n):
        # Each payload is ~400 chars of highly compressible repeating text
        template = (
            f"row_{i}_payload: This is a compressible text entry for row {i}. "
            f"The quick brown fox jumps over the lazy dog. "
            f"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            f"Repeat repeat repeat repeat repeat repeat repeat. "
        )
        # Repeat the template enough times to reach hundreds of characters
        payload = template * 4
        payloads.append(payload)
    return payloads


def _generate_embeddings(n: int, dim: int, seed: int) -> pa.FixedSizeListArray:
    """Generate deterministic embedding vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.random((n, dim)).astype(np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(vectors.flatten(), type=pa.float32()), list_size=dim)


def build_tables() -> None:
    """Create both default and zstd-compressed LanceDB tables."""
    run_id = _get_run_id()
    default_name = f"default_{run_id}"
    zstd_name = f"zstd_{run_id}"

    ids = list(range(NUM_ROWS))
    payloads = _generate_payloads(NUM_ROWS)
    embeddings = _generate_embeddings(NUM_ROWS, VECTOR_DIM, RNG_SEED)

    # --- Default table (no compression metadata) ---
    schema_default = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("payload", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), VECTOR_DIM)),
    ])

    data_default = pa.table({
        "id": pa.array(ids, type=pa.int64()),
        "payload": pa.array(payloads, type=pa.string()),
        "embedding": embeddings,
    }, schema=schema_default)

    db = lancedb.connect(DB_PATH)
    db.create_table(default_name, data=data_default, mode="overwrite")

    # --- Zstd-compressed table ---
    schema_zstd = pa.schema([
        pa.field("id", pa.int64()),
        pa.field(
            "payload",
            pa.string(),
            metadata={
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "3",
            },
        ),
        pa.field("embedding", pa.list_(pa.float32(), VECTOR_DIM)),
    ])

    data_zstd = pa.table({
        "id": pa.array(ids, type=pa.int64()),
        "payload": pa.array(payloads, type=pa.string()),
        "embedding": embeddings,
    }, schema=schema_zstd)

    db.create_table(zstd_name, data=data_zstd, mode="overwrite")


def _dir_size(path: Path) -> int:
    """Recursively sum file sizes under a directory."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            total += (Path(dirpath) / f).stat().st_size
    return total


def compare_sizes() -> dict:
    """Compare on-disk sizes of default vs zstd tables.

    Returns dict with keys: default_bytes, zstd_bytes, ratio.
    """
    run_id = _get_run_id()
    default_dir = Path(DB_PATH) / f"default_{run_id}.lance"
    zstd_dir = Path(DB_PATH) / f"zstd_{run_id}.lance"

    default_bytes = _dir_size(default_dir)
    zstd_bytes = _dir_size(zstd_dir)
    ratio = zstd_bytes / default_bytes if default_bytes > 0 else 0.0

    return {
        "default_bytes": default_bytes,
        "zstd_bytes": zstd_bytes,
        "ratio": ratio,
    }


def main() -> None:
    build_tables()
    result = compare_sizes()

    report_path = "/home/user/myproject/size_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"default_bytes={result['default_bytes']} "
        f"zstd_bytes={result['zstd_bytes']} "
        f"ratio={result['ratio']:.4f}"
    )


if __name__ == "__main__":
    main()