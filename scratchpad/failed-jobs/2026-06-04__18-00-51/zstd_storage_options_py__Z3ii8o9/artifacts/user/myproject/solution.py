"""
LanceDB Zstd Column Compression Sizing
Compares on-disk footprint of two tables: one with default encoding, one with zstd compression.
"""

import json
import os
import pathlib
import numpy as np
import pyarrow as pa
import lancedb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_PATH = "/home/user/myproject/lancedb_data"
NUM_ROWS = 5_000
EMBEDDING_DIM = 32
RNG_SEED = 2026


def _get_run_id() -> str:
    """Read run ID from env, default to 'local' if not set."""
    return os.environ.get("ZEALT_RUN_ID", "local")


def _build_data() -> tuple[list[int], list[str], list[list[float]]]:
    """Build the deterministic dataset shared by both tables."""
    rng = np.random.default_rng(RNG_SEED)

    ids = list(range(NUM_ROWS))

    # Compressible payload: a long repeating template unique to each row
    payloads = [
        (
            f"ROW-{i:05d}|CATEGORY:{i % 20:02d}|STATUS:ACTIVE|"
            "DESCRIPTION:This is a highly compressible text payload that repeats "
            "a standard template so that the zstd codec can achieve meaningful "
            "space savings compared to the uncompressed baseline. "
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            f"ROW-{i:05d}|END"
        )
        for i in range(NUM_ROWS)
    ]

    # Deterministic float32 embeddings, shape (NUM_ROWS, EMBEDDING_DIM)
    embeddings = rng.standard_normal((NUM_ROWS, EMBEDDING_DIM)).astype(np.float32)
    embedding_list = embeddings.tolist()

    return ids, payloads, embedding_list


def _build_schema(with_zstd: bool) -> pa.Schema:
    """Return a PyArrow schema, optionally with zstd metadata on the payload field."""
    if with_zstd:
        payload_field = pa.field(
            "payload",
            pa.string(),
            metadata={
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "9",
            },
        )
    else:
        payload_field = pa.field("payload", pa.string())

    return pa.schema(
        [
            pa.field("id", pa.int64()),
            payload_field,
            pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ]
    )


def _build_table(
    ids: list[int],
    payloads: list[str],
    embedding_list: list[list[float]],
    schema: pa.Schema,
) -> pa.Table:
    """Construct a PyArrow Table from the given data and schema."""
    return pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "payload": pa.array(payloads, type=pa.string()),
            "embedding": pa.array(
                embedding_list,
                type=pa.list_(pa.float32(), EMBEDDING_DIM),
            ),
        },
        schema=schema,
    )


def create_tables() -> None:
    """(Re)create both LanceDB tables with identical data."""
    run_id = _get_run_id()
    ids, payloads, embedding_list = _build_data()

    db = lancedb.connect(DB_PATH)

    for compressed in (False, True):
        table_name = f"{'zstd' if compressed else 'default'}_{run_id}"
        schema = _build_schema(with_zstd=compressed)
        arrow_table = _build_table(ids, payloads, embedding_list, schema)
        db.create_table(table_name, schema=schema, data=arrow_table, mode="overwrite")
        print(f"  Created table: {table_name}")


def _dir_size(path: pathlib.Path) -> int:
    """Sum the sizes of all files under *path* recursively."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for fname in files:
            total += pathlib.Path(root, fname).stat().st_size
    return total


def compare_sizes() -> dict:
    """
    Return {"default_bytes": int, "zstd_bytes": int, "ratio": float}.
    Re-computes from disk on every call.
    """
    run_id = _get_run_id()
    base = pathlib.Path(DB_PATH)
    default_dir = base / f"default_{run_id}.lance"
    zstd_dir = base / f"zstd_{run_id}.lance"

    default_bytes = _dir_size(default_dir)
    zstd_bytes = _dir_size(zstd_dir)
    ratio = zstd_bytes / default_bytes

    return {
        "default_bytes": default_bytes,
        "zstd_bytes": zstd_bytes,
        "ratio": ratio,
    }


def main() -> None:
    print("Creating LanceDB tables …")
    create_tables()

    print("Computing on-disk sizes …")
    result = compare_sizes()

    report_path = pathlib.Path("/home/user/myproject/size_report.json")
    report_path.write_text(json.dumps(result, indent=2))

    print(
        f"default_bytes={result['default_bytes']} "
        f"zstd_bytes={result['zstd_bytes']} "
        f"ratio={result['ratio']:.4f}"
    )


if __name__ == "__main__":
    main()
