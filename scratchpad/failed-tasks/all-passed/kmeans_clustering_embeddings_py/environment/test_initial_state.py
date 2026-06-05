import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_sklearn_importable():
    import sklearn.cluster  # noqa: F401
    import sklearn.metrics  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB data directory {LANCEDB_DIR} does not exist."


def test_embeddings_table_present_on_disk():
    # Native Lance tables are stored as `<table_name>.lance` directories.
    embeddings_path = os.path.join(LANCEDB_DIR, "embeddings.lance")
    assert os.path.isdir(embeddings_path), (
        f"Expected pre-seeded embeddings table at {embeddings_path}."
    )


def test_embeddings_table_has_800_rows_and_correct_schema():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    assert "embeddings" in db.table_names(), "Table 'embeddings' not found in LanceDB."
    tbl = db.open_table("embeddings")
    assert tbl.count_rows() == 800, f"Expected 800 rows in 'embeddings', got {tbl.count_rows()}."
    field_names = {f.name for f in tbl.schema}
    assert "id" in field_names, "embeddings schema missing 'id' field."
    assert "vector" in field_names, "embeddings schema missing 'vector' field."


def test_ground_truth_labels_file_exists():
    gt_path = os.path.join(LANCEDB_DIR, "ground_truth.npy")
    assert os.path.isfile(gt_path), f"Ground truth labels file {gt_path} does not exist."


def test_no_pre_existing_clusters_table():
    # The candidate is responsible for creating these tables.
    clusters_path = os.path.join(LANCEDB_DIR, "clusters.lance")
    centroids_path = os.path.join(LANCEDB_DIR, "centroids.lance")
    assert not os.path.exists(clusters_path), (
        f"'clusters' table should not exist before the candidate runs; found {clusters_path}."
    )
    assert not os.path.exists(centroids_path), (
        f"'centroids' table should not exist before the candidate runs; found {centroids_path}."
    )


def test_solution_and_run_files_absent():
    # Candidate must author these.
    assert not os.path.exists(os.path.join(PROJECT_DIR, "solution.py")), (
        "solution.py should not exist before the candidate runs."
    )
    assert not os.path.exists(os.path.join(PROJECT_DIR, "run.py")), (
        "run.py should not exist before the candidate runs."
    )
