# Durable Redis Stream -> LanceDB Ingest Consumer

## Background
You are building the ingest tier of a vector-search pipeline. Producers push embedding messages onto a **local Redis Stream**, and a durable consumer must move them into a **LanceDB** table with **exactly-once** effect. Redis is already running locally inside this environment (`127.0.0.1:6379`); there is **no external network, no hosted API, and no model download** — every vector is a raw little-endian `float32` array carried inside the message itself.

The hard part is crash-safety: the consumer uses a Redis **consumer group** so that a message is only removed from the pending-entries-list (PEL) after it has been durably committed. If the process crashes mid-batch, the un-acknowledged messages stay pending and must be **re-delivered and reprocessed** on the next run without creating duplicate rows.

## Requirements
- Implement a rerunnable consumer at `run_consumer.py` that reads embedding messages from a Redis Stream using a **consumer group** (`XREADGROUP`) and writes them into a LanceDB table.
- Batch reads by a configurable size and **acknowledge (`XACK`) only after the LanceDB commit for that batch succeeds** (at-least-once delivery).
- On startup, **recover** any previously-delivered-but-un-acknowledged messages for this group/consumer (via `XAUTOCLAIM` or `XPENDING`+`XCLAIM`) and reprocess them, so a simulated crash mid-batch loses no messages.
- Guarantee **exactly-once effect** in LanceDB by performing an **idempotent upsert keyed on the business `id`** (never a blind append), so a re-delivered message overwrites rather than duplicates its row.
- **Gracefully drain**: keep consuming until the stream yields no new entries, then exit with status 0.

## Implementation Hints
- Project path: /home/user/myproject
- Command: `python3 run_consumer.py`
- All configuration is provided via environment variables; read them and do not hardcode names: `REDIS_HOST` (default `127.0.0.1`), `REDIS_PORT` (default `6379`), `STREAM_KEY`, `GROUP_NAME`, `CONSUMER_NAME`, `LANCEDB_DIR` (filesystem path for `lancedb.connect`), `TABLE_NAME`, `BATCH_SIZE` (default `50`), and `VECTOR_DIM` (default `32`).
- Each stream entry has exactly three fields: `id` (a UTF-8 business id string), `vector` (the raw bytes of a little-endian `float32` numpy array of length `VECTOR_DIM`, i.e. produced by `ndarray.astype('<f4').tobytes()`), and `text` (UTF-8 text). Decode the vector with `numpy.frombuffer(raw, dtype='<f4')`.
- The LanceDB table stores one row per message with keys `id` (string), `text` (string), and `vector` (a fixed-size `float32` list of length `VECTOR_DIM`). Create the table if it does not already exist; otherwise open and upsert into it. Use an upsert that updates matched keys and inserts unmatched keys (merge-insert on `id`).
- Ensure the consumer group exists before reading (create it with `MKSTREAM` if needed and ignore an already-exists error). Reclaim the pending list starting from `0-0` with a zero minimum-idle-time so a just-crashed batch is picked up immediately.
- When finished draining, print exactly one final line to stdout in the form `DONE ingested=<int> reclaimed=<int>`, where `ingested` is the total number of stream entries this run committed to LanceDB (including any reclaimed ones) and `reclaimed` is how many previously-delivered but un-acknowledged entries this run recovered.

