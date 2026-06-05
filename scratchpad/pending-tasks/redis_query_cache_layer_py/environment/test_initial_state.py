import os
import shutil
import socket
import subprocess


PROJECT_DIR = "/home/user/myproject"


def test_python3_available():
    assert shutil.which("python3") is not None, "python3 not found in PATH."


def test_redis_server_binary_available():
    assert shutil.which("redis-server") is not None, "redis-server binary not found in PATH."


def test_redis_cli_available():
    assert shutil.which("redis-cli") is not None, "redis-cli binary not found in PATH."


def test_lancedb_importable():
    result = subprocess.run(
        ["python3", "-c", "import lancedb"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to import lancedb: {result.stderr}"


def test_redis_client_importable():
    result = subprocess.run(
        ["python3", "-c", "import redis"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to import redis: {result.stderr}"


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_redis_daemon_reachable():
    # Redis daemon must already be listening on localhost:6379 at task start.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        try:
            sock.connect(("127.0.0.1", 6379))
        except OSError as exc:  # pragma: no cover - assertion message handles it
            raise AssertionError(f"Redis daemon not reachable on localhost:6379: {exc}")


def test_redis_ping():
    result = subprocess.run(
        ["redis-cli", "-h", "127.0.0.1", "-p", "6379", "PING"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"redis-cli PING failed: {result.stderr}"
    assert result.stdout.strip() == "PONG", f"Expected PONG, got {result.stdout!r}"


def test_run_id_env_set():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable is not set."


def test_seeded_table_exists():
    run_id = os.environ["ZEALT_RUN_ID"]
    script = (
        "import lancedb\n"
        "db = lancedb.connect('/home/user/myproject/lancedb_data')\n"
        f"names = db.table_names()\n"
        f"assert 'docs_{run_id}' in names, f'docs_{run_id} not in {{names}}'\n"
        f"tbl = db.open_table('docs_{run_id}')\n"
        "assert tbl.count_rows() == 200, f'expected 200 rows, got {tbl.count_rows()}'\n"
    )
    result = subprocess.run(
        ["python3", "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Seeded table check failed: {result.stderr}"
