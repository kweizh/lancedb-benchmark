import base64
import os
import shutil
import subprocess
import time

import pytest

PROJECT_DIR = "/home/user/myproject"
DB_URI = "/home/user/myproject/lancedb"
TABLE_NAME = "documents"
ETCD_URL = "http://127.0.0.1:2379"


def _ensure_etcd_running():
    """Best-effort: start the local etcd server if it is not already up."""
    import requests

    for _ in range(2):
        try:
            resp = requests.post(
                f"{ETCD_URL}/v3/kv/range",
                json={"key": base64.b64encode(b"\x00").decode()},
                timeout=3,
            )
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        starter = "/usr/local/bin/start-etcd.sh"
        if os.path.isfile(starter):
            subprocess.run(["bash", starter], check=False)
            time.sleep(3)
    return False


def test_requests_importable():
    import requests  # noqa: F401


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_seed_script_exists():
    seed = os.path.join(PROJECT_DIR, "seed_dataset.py")
    assert os.path.isfile(seed), f"Seed script {seed} does not exist."


def test_etcd_binary_available():
    assert shutil.which("etcd") is not None, "etcd binary not found in PATH."


def test_start_etcd_script_exists():
    starter = "/usr/local/bin/start-etcd.sh"
    assert os.path.isfile(starter), f"etcd start script {starter} does not exist."


def test_etcd_reachable():
    assert _ensure_etcd_running(), (
        "Local etcd server is not reachable at "
        f"{ETCD_URL} via the JSON gRPC gateway."
    )


def test_seeded_table_present_with_unindexed_rows():
    import lancedb

    db = lancedb.connect(DB_URI)
    assert TABLE_NAME in db.table_names(), (
        f"Seeded table '{TABLE_NAME}' not found at {DB_URI}."
    )
    tbl = db.open_table(TABLE_NAME)
    assert tbl.count_rows() == 500, (
        f"Expected seeded table to contain 500 rows, got {tbl.count_rows()}."
    )

    indices = tbl.list_indices()
    assert len(indices) >= 1, "Seeded table is missing its vector index."
    index_name = indices[0].name
    stats = tbl.index_stats(index_name)
    assert stats is not None, f"Could not read index_stats for index '{index_name}'."
    assert stats.num_unindexed_rows == 200, (
        "Expected 200 freshly-appended unindexed rows in the seeded table, got "
        f"{stats.num_unindexed_rows}."
    )
