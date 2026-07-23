# Exactly-Once Kafka -> LanceDB Ingestion Consumer

## Background
You are building the durable ingestion tier of a document search system. A partitioned Kafka topic carries a stream of documents, and each document must land exactly once in a LanceDB table even if the consumer crashes and Kafka redelivers records. A local Kafka-compatible broker (Redpanda) is already running inside this environment and listening on `localhost:9092` (also exposed via `KAFKA_BOOTSTRAP_SERVERS`, default `localhost:9092`). LanceDB is the durable store.

The classic pitfall is that Kafka offset commits and the downstream write are two separate operations: if you commit offsets before the data is durably written, a crash loses records; if a record is redelivered after a crash, a naive writer duplicates rows. Your job is to make the end-to-end effect exactly-once by combining manual (post-write) offset management with an idempotent writer.

## Requirements
- Implement a Kafka consumer that reads every message from the input topic and writes one row per document into a LanceDB table.
- Use manual offset management: disable Kafka auto-commit and commit offsets only AFTER the corresponding rows are durably written to LanceDB.
- Make writes idempotent: replaying already-processed records (duplicate keys in the stream, or a full re-consume after an offset reset) must never create duplicate rows. Maintain an idempotency-key store or upsert on the document id.
- Handle a multi-partition topic and consumer-group rebalances: track and commit offsets per assigned partition, and record which Kafka partition/offset each row came from.
- The consumer must drain the currently-available messages and then exit cleanly with status code 0 (for example, stop once polling yields no new records for a few consecutive seconds). Re-running the command must be safe and idempotent.

## Implementation Hints
- Project path: `/home/user/myproject`
- Command: `python3 run.py` (run from the project directory). It connects to `KAFKA_BOOTSTRAP_SERVERS` (default `localhost:9092`), consumes the topic, writes to LanceDB, and exits 0 when the topic is drained.
- Use the `confluent-kafka` Python client. Set `enable.auto.commit=false` and commit offsets yourself after each durable batch write. The consumer group is a long-lived group (subscribe, do not manually assign) so that rebalances are handled by the group protocol.
- Resource names are derived from the run id. Read the run id from the file `/logs/artifacts/run-id` (strip whitespace); if that file is absent, use the literal `zrlocal`. Then use exactly these names:
  - Kafka topic: `ingest-docs-<run-id>`
  - Kafka consumer group: `ingest-group-<run-id>`
  - LanceDB table: `documents_<run-id>`
- Open the LanceDB database at the directory `/home/user/myproject/lancedb`.
- Each Kafka message has a UTF-8 string key equal to the document id, and a UTF-8 JSON value with the keys `id` (string), `text` (string), `source` (string), and `ts` (integer, unix seconds). The `id` in the value equals the message key and is the idempotency key.
- Persist each document as a LanceDB row with exactly these columns: `id` (string), `text` (string), `source` (string), `ts` (int64), `partition` (int64, the Kafka partition the record came from), `offset` (int64, the Kafka offset), and `vector` (a 32-dimensional float32 fixed-size list embedding of `text`).
- Compute the `vector` embedding deterministically and locally (no network, no hosted model): start from a zero vector of length 32; for every whitespace-separated token of the lowercased `text`, compute `bucket = int(hashlib.sha1(token.encode('utf-8')).hexdigest(), 16) % 32` and add `1.0` to `vector[bucket]`; finally L2-normalize the vector (leave it all zeros if the norm is 0). Store it as float32.
- A duplicate is any record whose `id` already exists in the table; the table must keep a single row per `id`. `merge_insert` upsert on `id` is a convenient way to achieve idempotent writes in LanceDB.

