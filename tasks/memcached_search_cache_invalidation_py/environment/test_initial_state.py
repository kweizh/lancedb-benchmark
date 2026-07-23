import json
import os
import socket

import pytest

PROJECT_DIR = "/home/user/myproject"
FIXTURE_PATH = os.path.join(PROJECT_DIR, "fixture.json")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pymemcache_importable():
    import pymemcache  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_memcached_running():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect(("127.0.0.1", 11211))
    except OSError as exc:  # pragma: no cover - explicit failure message
        pytest.fail(f"memcached is not reachable on 127.0.0.1:11211: {exc}")
    finally:
        sock.close()


def test_fixture_file_exists_and_valid():
    assert os.path.isfile(FIXTURE_PATH), f"Fixture file {FIXTURE_PATH} does not exist."
    with open(FIXTURE_PATH) as f:
        fixture = json.load(f)
    for key in ("db_path", "table_name", "dim", "num_rows"):
        assert key in fixture, f"Fixture file is missing required key '{key}'."


def test_seeded_table_present():
    import lancedb

    with open(FIXTURE_PATH) as f:
        fixture = json.load(f)
    db = lancedb.connect(fixture["db_path"])
    assert fixture["table_name"] in db.table_names(), (
        f"Seeded table '{fixture['table_name']}' not found in LanceDB at {fixture['db_path']}."
    )
    table = db.open_table(fixture["table_name"])
    assert table.count_rows() == fixture["num_rows"], (
        "Seeded table row count does not match fixture metadata."
    )
