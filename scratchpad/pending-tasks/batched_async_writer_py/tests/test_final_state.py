import asyncio
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pyarrow as pa
import pytest


PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "data", "lancedb")
FLUSH_LOG = os.path.join(PROJECT_DIR, "flush_log.txt")


def _run_id() -> str:
    return os.environ.get("ZEALT_RUN_ID", "")


def _table_name() -> str:
    return f"events_{_run_id()}"


@pytest.fixture(scope="session", autouse=True)
def execute_driver():
    """Clean any leftover state, then run the candidate's driver once."""
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR, ignore_errors=True)
    if os.path.isfile(FLUSH_LOG):
        os.remove(FLUSH_LOG)
    proc = subprocess.run(
        ["python3", "run.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ},
    )
    assert proc.returncode == 0, (
        f"`python3 run.py` failed (exit={proc.returncode}).\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    yield


def test_solution_module_exposes_batched_writer():
    sys.path.insert(0, PROJECT_DIR)
    try:
        import solution  # type: ignore
    finally:
        # keep solution importable for later tests
        pass
    assert hasattr(solution, "BatchedWriter"), (
        "solution.py must expose a `BatchedWriter` class."
    )
    sig = inspect.signature(solution.BatchedWriter.__init__)
    expected = {"self", "table", "batch_size", "max_in_flight", "flush_interval_ms"}
    assert expected.issubset(sig.parameters.keys()), (
        f"BatchedWriter.__init__ must accept {expected}; got {list(sig.parameters.keys())}"
    )
    for method in ("add", "flush", "close"):
        assert hasattr(solution.BatchedWriter, method), (
            f"BatchedWriter is missing required method `{method}`."
        )
        assert inspect.iscoroutinefunction(getattr(solution.BatchedWriter, method)), (
            f"BatchedWriter.{method} must be an `async def` coroutine method."
        )


def test_table_persisted_with_10000_rows():
    import lancedb

    db = lancedb.connect(DB_DIR)
    names = db.table_names()
    assert _table_name() in names, (
        f"Expected table `{_table_name()}` in db.table_names(); got {names}."
    )
    tbl = db.open_table(_table_name())
    df = tbl.to_pandas()
    assert len(df) == 10000, f"Expected 10000 rows, got {len(df)}."

    ids = df["id"].tolist()
    assert len(set(ids)) == 10000, "Found duplicate `id` values in the table."
    assert set(ids) == set(range(10000)), (
        "`id` column must contain every integer from 0 through 9999 exactly once."
    )

    # Vector column must be 16-dim float arrays
    first_vec = df["vector"].iloc[0]
    arr = np.asarray(first_vec)
    assert arr.shape == (16,), f"Expected 16-dim vector, got shape {arr.shape}."


def test_flush_log_has_batched_writes_not_per_row():
    assert os.path.isfile(FLUSH_LOG), (
        f"Flush log {FLUSH_LOG} does not exist; the writer must append one line per flush."
    )
    with open(FLUSH_LOG, "r") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) >= 60, (
        f"Expected at least 60 flush lines (batching enforced); got {len(lines)}."
    )
    assert len(lines) < 10000, (
        f"Found {len(lines)} flush lines — this looks like per-row writes rather than batched flushes."
    )


def _load_solution_module():
    sys.path.insert(0, PROJECT_DIR)
    spec = importlib.util.spec_from_file_location(
        "solution", os.path.join(PROJECT_DIR, "solution.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_writer_backpressure_and_fast_flush():
    """Exercise BatchedWriter directly to confirm the contract."""
    solution = _load_solution_module()

    async def scenario() -> tuple[int, float]:
        import lancedb

        tmpdir = tempfile.mkdtemp(prefix="bw-verify-")
        try:
            db = await lancedb.connect_async(tmpdir)
            schema = pa.schema(
                [
                    pa.field("id", pa.int64()),
                    pa.field("seq", pa.int64()),
                    pa.field("vector", pa.list_(pa.float32(), 16)),
                ]
            )
            tbl = await db.create_table("verify_writer", schema=schema, mode="overwrite")

            writer = solution.BatchedWriter(
                tbl, batch_size=16, max_in_flight=2, flush_interval_ms=200
            )

            rng = np.random.default_rng(0)
            for i in range(50):
                row = {
                    "id": i,
                    "seq": i,
                    "vector": rng.standard_normal(16).astype(np.float32).tolist(),
                }
                await writer.add(row)
                await asyncio.sleep(0)

            t0 = time.monotonic()
            await writer.flush()
            elapsed = time.monotonic() - t0

            await writer.close()
            count = await tbl.count_rows()
            return count, elapsed
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    count, elapsed = asyncio.run(scenario())
    assert count == 50, f"Direct BatchedWriter test expected 50 rows, got {count}."
    # 2x flush_interval_ms = 0.4s; allow generous 1.0s slack for cold starts.
    assert elapsed < 0.4 + 1.0, (
        f"Trailing flush() took {elapsed:.3f}s; expected under 2*flush_interval_ms (+1s slack)."
    )
