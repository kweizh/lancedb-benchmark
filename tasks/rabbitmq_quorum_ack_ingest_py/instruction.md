# Reliable RabbitMQ -> LanceDB Ingestion Pipeline

## Background
You are building the ingestion worker for a document search system. Documents arrive on a **RabbitMQ quorum queue** and must be embedded and stored durably in a **LanceDB** table. The upstream publisher only guarantees *at-least-once* delivery, so the same document id can arrive more than once (duplicates / redeliveries). Your worker must turn this into *effectively-once* storage: each unique document id is stored exactly once. Malformed ("poison") messages must never crash the worker or reach LanceDB; instead they must be dead-lettered.

A local RabbitMQ broker is available on `localhost:5672` (virtual host `/`, credentials `guest` / `guest`). No internet access, cloud APIs, or hosted embedding models are available — everything runs locally.

## Requirements
- Consume documents from a durable **quorum** queue using **manual acknowledgement** (no auto-ack).
- Deduplicate by document id so every unique id is written to LanceDB exactly once, even across duplicate deliveries and across repeated runs of the worker.
- Embed each document's text with a **deterministic local embedding** (defined below) and store rows in a LanceDB table, writing in **batches**.
- Only acknowledge a message **after** the batch containing it has been durably written to LanceDB.
- Route **poison messages** (see rules below) to a dead-letter queue by rejecting them without requeue; never write them to LanceDB.
- Drain all currently-available messages, then exit (the worker must not block forever).

## Implementation Hints
- Use the `pika` client to talk to RabbitMQ and the `lancedb` library for storage.
- Persist dedup state in LanceDB itself (query existing ids) plus an in-run seen-set, so duplicates within a batch and across runs are both handled.
- Reject poison messages with a negative acknowledgement and `requeue=False` so the broker dead-letters them via the configured dead-letter exchange.
- Project path: `/home/user/project`
- Command: `python3 ingest.py` (rerunnable; each invocation drains the queue then exits).

### Broker topology (declare idempotently with EXACTLY these names/arguments)
- Direct dead-letter exchange `documents.dlx`, type `fanout`, durable.
- Main queue `documents`: durable, arguments `{"x-queue-type": "quorum", "x-dead-letter-exchange": "documents.dlx"}`.
- Dead-letter queue `documents.dlq`: durable, arguments `{"x-queue-type": "quorum"}`, bound to exchange `documents.dlx`.
- Messages are published to the default exchange with routing key `documents` (i.e. directly onto the `documents` queue). Each message body is UTF-8 JSON.

### Message / document rules
- A valid document is a JSON object with a non-empty string `id` and a non-empty string `text`.
- A message is a **poison message** if its body is not valid UTF-8 JSON, OR the decoded value is not a JSON object, OR it lacks a non-empty string `id`, OR it lacks a non-empty string `text`. Poison messages must be dead-lettered (nack, `requeue=False`) and must not be written to LanceDB.

### Deterministic embedding (64-dim float32) — implement EXACTLY
1. Lowercase the `text` and extract tokens matching the regex `[a-z0-9]+`.
2. Start with a zero vector of length 64.
3. For each token compute `idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 64` and add `1.0` to `vector[idx]`.
4. L2-normalize the vector (if the norm is greater than 0, divide every component by the norm).

### LanceDB output
- Database directory: `/home/user/project/data/lancedb`.
- Table name: `documents` with columns `id` (string), `text` (string), and `vector` (64-dim float32 vector). Exactly one row per unique id.

### Batch commit log
- Batch size is read from the environment variable `INGEST_BATCH_SIZE` (default `16`).
- Buffer new (non-duplicate, valid) documents in arrival order and write them to LanceDB in batches of at most the batch size; flush any partial final batch before exiting.
- Append one JSON line per committed batch to `/home/user/project/data/commits.log`. Each line is an object with keys `batch_index` (0-based, strictly increasing across the life of the file, starting at 0 on a fresh file) and `ids` (the list of document ids written in that batch, in write order). A batch line must be written only after that batch has been durably written to LanceDB.

### Stdout summary
- On completion print exactly one line to stdout: `INGEST_DONE written=<W> skipped_duplicates=<D> dead_lettered=<P>` where `W` is the number of new rows written to LanceDB during this run, `D` is the number of messages skipped because their id already existed (duplicates), and `P` is the number of messages dead-lettered during this run.

