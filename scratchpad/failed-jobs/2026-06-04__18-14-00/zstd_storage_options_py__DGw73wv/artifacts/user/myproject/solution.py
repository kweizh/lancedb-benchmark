"""
LanceDB Zstd Column Compression Sizing

Compares on-disk footprint of two LanceDB tables holding identical 5,000-row
content where one uses zstd compression on the textual payload column.
"""

import json
import os
import pathlib

import lancedb
import numpy as np
import pyarrow as pa

# ── Constants ─────────────────────────────────────────────────────────────────
LANCEDB_PATH = "/home/user/myproject/lancedb_data"
NUM_ROWS = 5_000
VECTOR_DIM = 32
REPORT_PATH = "/home/user/myproject/size_report.json"

# ── Data generation ───────────────────────────────────────────────────────────

def _make_payload(i: int) -> str:
    """Return a long, highly compressible string for row i."""
    template = (
        f"Record identifier={i} | category=alpha | status=active | "
        "description=This is a repeating boilerplate sentence that exists solely "
        "to provide compressible textual content for benchmarking purposes. "
        "The quick brown fox jumps over the lazy dog. "
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua. "
    )
    # Repeat to reach ~400 chars of compressible content
    return (template * 3)[:600]


def _build_arrays():
    """Build id, payload, and embedding arrays shared by both tables."""
    rng = np.random.default_rng(2026)

    ids = list(range(NUM_ROWS))
    payloads = [_make_payload(i) for i in range(NUM_ROWS)]
    embeddings = rng.standard_normal((NUM_ROWS, VECTOR_DIM)).astype(np.float32)

    return ids, payloads, embeddings


# ── Schema helpers ─────────────────────────────────────────────────────────────

def _default_schema() -> pa.Schema:
    """Schema with no compression metadata."""
    return pa.schema([
        pa.field("id", pa.int64()),
        pa.field("payload", pa.string()),
        pa.field(
            "embedding",
            pa.list_(pa.float32(), VECTOR_DIM),
        ),
    ])


def _zstd_schema(compression_level: int = 9) -> pa.Schema:
    """Schema with zstd compression metadata on the payload column."""
    payload_field = pa.field(
        "payload",
        pa.string(),
        metadata={
            "lance-encoding:compression": "zstd",
            "lance-encoding:compression-level": str(compression_level),
        },
    )
    return pa.schema([
        pa.field("id", pa.int64()),
        payload_field,
        pa.field(
            "embedding",
            pa.list_(pa.float32(), VECTOR_DIM),
        ),
    ])


# ── Table creation ─────────────────────────────────────────────────────────────

def _make_record_batch(schema: pa.Schema, ids, payloads, embeddings) -> pa.RecordBatch:
    embedding_type = pa.list_(pa.float32(), VECTOR_DIM)
    embedding_array = pa.array(
        [row.tolist() for row in embeddings],
        type=embedding_type,
    )
    return pa.record_batch(
        {
            "id": pa.array(ids, type=pa.int64()),
            "payload": pa.array(payloads, type=pa.string()),
            "embedding": embedding_array,
        },
        schema=schema,
    )


def create_tables(run_id: str) -> tuple[str, str]:
    """
    (Re)create both tables for the given run_id.
    Returns (default_table_name, zstd_table_name).
    """
    ids, payloads, embeddings = _build_arrays()

    db = lancedb.connect(LANCEDB_PATH)

    default_name = f"default_{run_id}"
    zstd_name = f"zstd_{run_id}"

    # --- default table ---
    default_schema = _default_schema()
    default_batch = _make_record_batch(default_schema, ids, payloads, embeddings)
    db.create_table(default_name, schema=default_schema, data=default_batch, mode="overwrite")

    # --- zstd table ---
    zstd_schema = _zstd_schema(compression_level=9)
    zstd_batch = _make_record_batch(zstd_schema, ids, payloads, embeddings)
    db.create_table(zstd_name, schema=zstd_schema, data=zstd_batch, mode="overwrite")

    return default_name, zstd_name


# ── Size computation ───────────────────────────────────────────────────────────

def _dir_size(table_name: str) -> int:
    """Return total on-disk bytes for a LanceDB table directory."""
    table_dir = pathlib.Path(LANCEDB_PATH) / f"{table_name}.lance"
    total = 0
    for root, _dirs, files in os.walk(table_dir):
        for fname in files:
            fpath = pathlib.Path(root) / fname
            try:
                total += fpath.stat().st_size
            except OSError:
                pass
    return total


def compare_sizes(run_id: str | None = None) -> dict:
    """
    Re-compute on-disk sizes from disk and return a dict with keys:
      default_bytes, zstd_bytes, ratio  (ratio = zstd_bytes / default_bytes)
    """
    if run_id is None:
        run_id = os.environ.get("ZEALT_RUN_ID", "default")

    default_bytes = _dir_size(f"default_{run_id}")
    zstd_bytes = _dir_size(f"zstd_{run_id}")
    ratio = zstd_bytes / default_bytes if default_bytes else float("inf")

    return {
        "default_bytes": default_bytes,
        "zstd_bytes": zstd_bytes,
        "ratio": ratio,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    run_id = os.environ.get("ZEALT_RUN_ID", "default")

    # 1. (Re)create both tables
    create_tables(run_id)

    # 2. Compute sizes
    report = compare_sizes(run_id)

    # 3. Write JSON report
    report_path = pathlib.Path(REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(report, fh)

    # 4. Print one-line summary
    print(
        f"default_bytes={report['default_bytes']} "
        f"zstd_bytes={report['zstd_bytes']} "
        f"ratio={report['ratio']:.4f}"
    )


if __name__ == "__main__":
    main()
