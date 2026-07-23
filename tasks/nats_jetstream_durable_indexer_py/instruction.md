# Durable Exactly-Once-Effect Indexer on NATS JetStream + LanceDB

## Background
You are building the ingestion path of a vector-search system. Documents are published as messages to a **NATS JetStream** stream. A background worker must durably consume those messages and index them into a **LanceDB** table so that search stays consistent even when the worker crashes and restarts. Because JetStream guarantees *at-least-once* delivery, the same message may be delivered more than once; your worker must therefore produce an **exactly-once effect** in LanceDB (every document id appears exactly once with the correct vector, no matter how many times its message is redelivered).

A local `nats-server` with JetStream enabled is already running inside the container at `nats://127.0.0.1:4222` (its binary is on `PATH`). No external network access is available or needed.

## Requirements
- Implement a durable indexing worker that consumes documents from a JetStream stream and upserts them into a LanceDB table.
- Use a **durable pull consumer** so consumer position (ack floor) is persisted on the server across worker restarts.
- Fetch messages in **batches**; for each batch, first commit the documents to LanceDB, then acknowledge the messages. Never acknowledge a message before its effect is durably committed.
- Make reprocessing / redelivery **idempotent**: upsert by document `id` so a redelivered or duplicate message never creates a second row and never corrupts an existing one.
- The worker must be safely **re-runnable**: after a simulated restart it resumes from the durable consumer's stored position, and running it again once the stream is drained must be a no-op.

## Implementation Hints
- Project path: `/home/user/myproject`
- Put your implementation in `/home/user/myproject/solution.py`.
- Use the async `nats-py` client (`import nats`) to talk to JetStream, and `lancedb` for storage.
- Read all configuration from environment variables (the verifier sets them): `NATS_URL`, `JS_STREAM`, `JS_SUBJECT`, `JS_DURABLE`, `LANCEDB_PATH`, `INDEX_BATCH_SIZE`, and `ZEALT_RUN_ID`.
- The LanceDB table name MUST be exactly `documents_${ZEALT_RUN_ID}` (read `ZEALT_RUN_ID` from the environment and append it). The table lives in the directory given by `LANCEDB_PATH`.
- Each JetStream message body is UTF-8 JSON with exactly the keys `id` (integer), `text` (string), and `vector` (a list of 32 floats). Store rows with an `id` column (int64), a `text` column (string), and a `vector` column that is a fixed-size list of 32 float32 values.
- Upsert semantics must be achieved with LanceDB's `merge_insert("id")` matched-update / not-matched-insert pattern so that redelivered ids update in place instead of duplicating.
- The durable pull consumer must use explicit acknowledgement. Create the stream (subject = `JS_SUBJECT`) and the durable consumer if they do not already exist; binding to an existing durable consumer is how the worker resumes after a restart.
- Expose an async coroutine `run_indexer(max_messages=None)` in `solution.py`. Each call must open its own NATS connection, drain available messages (stopping early once it has committed **and** acknowledged `max_messages` messages when that argument is not `None`, otherwise continuing until no more messages are available), close the connection, and return a dict whose `committed` key is the integer number of messages whose effect was committed and acknowledged during that call. A call made against a fully drained stream must return `{"committed": 0}` and leave the table unchanged.
- The verifier drives your worker by importing `solution` and calling `run_indexer(...)` several times (via `asyncio`); design it so repeated independent calls behave like independent worker restarts.
- LanceDB 0.25.3 can raise `SIGABRT` during interpreter teardown; if you run any standalone seeding/utility code, guard the process exit accordingly (this does not apply to the imported coroutine which simply returns).

