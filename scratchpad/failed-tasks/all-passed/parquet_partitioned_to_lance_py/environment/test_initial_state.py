import os

import pytest

PROJECT_DIR = "/home/user/myproject"
PARQUET_DIR = os.path.join(PROJECT_DIR, "parquet_dataset")
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401
    import pyarrow.dataset  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project dir {PROJECT_DIR} does not exist."


def test_parquet_dataset_dir_exists():
    assert os.path.isdir(PARQUET_DIR), (
        f"Parquet dataset dir {PARQUET_DIR} should be pre-generated at build time."
    )


@pytest.mark.parametrize("year", [2022, 2023, 2024])
def test_year_partition_dir_exists(year):
    partition_dir = os.path.join(PARQUET_DIR, f"year={year}")
    assert os.path.isdir(partition_dir), (
        f"Hive partition dir {partition_dir} should be pre-generated at build time."
    )
    # There must be at least one parquet file in the partition.
    files = [f for f in os.listdir(partition_dir) if f.endswith(".parquet")]
    assert len(files) >= 1, f"Partition {partition_dir} must contain at least one .parquet file."


def test_partition_files_do_not_contain_year_column():
    """Hive partitioning encodes `year` only in the directory name, not as a physical column.

    Use ParquetFile (no partition discovery) since pq.read_table auto-discovers Hive
    partition columns from the parent directory in pyarrow >= 11.
    """
    import pyarrow.parquet as pq

    partition_dir = os.path.join(PARQUET_DIR, "year=2022")
    files = sorted(f for f in os.listdir(partition_dir) if f.endswith(".parquet"))
    assert files, "No parquet files found in year=2022 partition."
    pf = pq.ParquetFile(os.path.join(partition_dir, files[0]))
    file_cols = pf.schema_arrow.names
    assert "year" not in file_cols, (
        "Partition files must NOT contain `year` as a physical column; "
        f"it should only be encoded in the Hive partition directory name. Got: {file_cols}"
    )
    for col in ("id", "title", "embedding"):
        assert col in file_cols, (
            f"Partition file is missing required column `{col}`. Got: {file_cols}"
        )


def test_lancedb_dir_is_empty_or_missing():
    """Candidate's solution.py is responsible for creating the destination table."""
    if not os.path.exists(LANCEDB_DIR):
        return
    assert os.path.isdir(LANCEDB_DIR), f"{LANCEDB_DIR} exists but is not a directory."
    entries = [e for e in os.listdir(LANCEDB_DIR) if not e.startswith(".")]
    assert entries == [], (
        f"LanceDB dir {LANCEDB_DIR} should be empty before the candidate runs; "
        f"found: {entries}"
    )


def test_solution_module_not_yet_created():
    """Candidate must author /home/user/myproject/solution.py themselves."""
    solution_path = os.path.join(PROJECT_DIR, "solution.py")
    assert not os.path.exists(solution_path), (
        f"{solution_path} should NOT exist before the candidate begins."
    )


def test_zealt_run_id_env_set():
    assert os.environ.get("ZEALT_RUN_ID"), (
        "ZEALT_RUN_ID environment variable must be set in the task environment."
    )
