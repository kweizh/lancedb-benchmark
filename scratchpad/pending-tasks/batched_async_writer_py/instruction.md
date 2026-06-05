# Asynchronous Batched Writer for LanceDB

## Background
High-throughput ingestion services that feed LanceDB usually receive rows one-at-a-time from upstream producers (e.g. Kafka consumers, web sockets, event loops). Issuing one `table.add([row])` per arrival is unacceptably slow because each call rewrites a new Lance fragment. The standard cure is a back-pressured, batched writer: producers enqueue rows on an in-memory `asyncio.Queue`; one or more background worker tasks coalesce up to N rows (or wait at most T milliseconds) before issuing a single `await table.add(batch)`. This task asks you to build that writer on top of `lancedb.connect_async`.

## Requirements
Implement a reusable `BatchedWriter` class plus a demo CLI that exercises it end-to-end.

- `BatchedWriter(table, batch_size=128, max_in_flight=4, flush_interval_ms=500)` where `table` is an `AsyncTable`.
- `await writer.add(row)` — non-blocking enqueue. Must return immediately when the queue has space and must `await` (apply back-pressure) when the queue is full.
- `await writer.flush()` — wait until all rows enqueued so far have been persisted to the table.
- `await writer.close()` — flush, stop the worker tasks, and release resources.
- Internal worker tasks must coalesce up to `batch_size` rows OR wait at most `flush_interval_ms` (whichever happens first) before issuing **one** `await tbl.add(batch)` call.
- The writer must append a timestamped line to a flush log every time it actually calls `tbl.add`.
- A driver script `run.py` that connects to LanceDB, opens a table whose name is suffixed with `${ZEALT_RUN_ID}`, produces exactly 10000 rows (each with a monotonically increasing `id`, a deterministic 16-dim float32 `vector`, and a `seq` value), feeds them through the writer with random small sleeps to simulate realistic interleaving, then calls `flush()` and `close()`, and finally prints a summary.

## Implementation Hints
- Open the database with `lancedb.connect_async("data/lancedb")` and create the table with an explicit Arrow schema using `pa.list_(pa.float32(), 16)` (or `pa.float32` fixed-size list) so the vector dimension is locked.
- An `asyncio.Queue(maxsize=batch_size * max_in_flight)` gives natural back-pressure: producers awaiting `queue.put()` block when the buffer is full.
- Use a single background worker (or a small pool) that collects items via `queue.get()` with `asyncio.wait_for(..., timeout=flush_interval_ms/1000)` to implement the time-based flush.
- Track in-flight batches with a counter / `asyncio.Event` so `flush()` can wait until everything has truly been written.
- Append every flush event to `/home/user/myproject/flush_log.txt` with format `ts=<iso8601> batch_size=<int>`. Producing one line per flush makes the verifier's flush-count check robust.
- Read the run id from the `ZEALT_RUN_ID` environment variable and use it as the table-name suffix: `events_${ZEALT_RUN_ID}`. The driver must be re-runnable: drop or `mode="overwrite"` the existing table on startup.
- Avoid any sleep-based busy loops; rely on `asyncio.Queue`, `asyncio.wait_for`, `asyncio.Event`, and `asyncio.gather`.

## Acceptance Criteria
- Project path: `/home/user/myproject`
- Command: `python3 run.py`
- The script must connect to a LanceDB database at `/home/user/myproject/data/lancedb`.
- The script must read `ZEALT_RUN_ID` from the environment and write into a table named `events_${ZEALT_RUN_ID}`.
- The table schema must contain at least an `id` (int64), a `seq` (int64), and a 16-dim float32 vector column named `vector`.
- After `python3 run.py` exits successfully:
  - The table contains exactly 10000 rows.
  - There are no duplicate `id` values.
  - The `id` column contains every integer from `0` through `9999`.
  - `/home/user/myproject/flush_log.txt` exists and contains **at least 60** lines (one per flush operation). Far more than 60 is acceptable; the only failure is reverting to per-row writes which would produce 10000 lines but the writer never issues that many `add` calls because batching is enforced.
- The writer's `flush()` must return promptly: a final `flush()` call made after the producer has stopped must complete within 2× `flush_interval_ms`.
- `BatchedWriter` must be importable from `/home/user/myproject/solution.py` as `from solution import BatchedWriter` and accept the four documented constructor arguments.

