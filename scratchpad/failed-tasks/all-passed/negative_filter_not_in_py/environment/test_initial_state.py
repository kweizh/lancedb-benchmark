import os


PROJECT_DIR = "/home/user/myproject"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Expected project directory {PROJECT_DIR} to exist before the task starts."
    )


def test_zealt_run_id_env_var_present():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id and run_id.strip(), (
        "Expected ZEALT_RUN_ID environment variable to be set for parallel-run isolation."
    )
