# MongoDB Change Stream -> LanceDB CDC Synchronizer (Python)

## Background
You are building a change-data-capture (CDC) pipeline. A MongoDB single-node replica set runs locally in this environment and is the system of record. Application data lives in the MongoDB collection `cdc.documents`. You must keep a LanceDB table continuously in sync with that collection by tailing MongoDB's change stream and incrementally applying inserts, updates, replacements, and deletes into LanceDB, while keeping a vector embedding consistent with each document's text. The pipeline must survive process restarts (crash recovery) by persisting a MongoDB resume token.

LanceDB is an embedded, AI-native vector database built on the Lance columnar format. It supports upserts via the `merge_insert` API and row deletion via SQL filter predicates.

MongoDB is provided locally in this environment. Ensure it is running by executing the idempotent helper `start-mongo.sh` (available on the PATH); it starts `mongod`, initiates the single-node replica set `rs0`, and waits for a writable primary. It listens on `127.0.0.1:27017`. No internet access is available or needed.

## Requirements
- Implement a synchronizer that reads the MongoDB change stream for the `cdc.documents` collection and applies each event, in the order received, to a LanceDB table.
- Translate MongoDB CDC events into LanceDB operations:
  - `insert`, `update`, and `replace` events must become an upsert (insert new row or update the existing row) into LanceDB, keyed by the document id.
  - `delete` events must remove (tombstone) the corresponding row from LanceDB.
- Recompute and store each document's vector embedding from its current `text` whenever the row is inserted or changed, so the stored vector always matches the current text.
- Persist a MongoDB resume token so that a restart resumes exactly where the previous run stopped, without missing or duplicating events.

## Implementation Hints
- Use `pymongo` to open a change stream on the collection and request the full updated document so `update` events (which by default carry only changed fields) can be applied the same way as `replace` events.
- Use LanceDB's `merge_insert` (match on the id column) for upserts and the table's `delete` method with a SQL filter for deletes.
- Apply events strictly in the order they are received from the change stream so that, e.g., an insert-then-update-then-delete of the same id within one run yields no row.
- The command must process only the change events that are currently available and then exit. It must NOT block waiting for future events. A non-blocking drain (stop when the stream currently has no more events) is expected.
- Crash recovery is driven entirely by the resume token: on startup, if a saved resume token exists, resume the change stream from it; otherwise start a fresh stream from the current position.
- Hard requirements (these are exact and must be followed):
  - Project path: `/home/user/project`
  - Command: `python3 sync.py` (run with the working directory at the project path; takes no arguments).
  - MongoDB connection: use the URI in environment variable `MONGO_URI` if set, otherwise default to `mongodb://localhost:27017/?replicaSet=rs0`. Source data is the collection `documents` in database `cdc`. Each MongoDB document has a string `_id`, a string field `text`, and a string field `category`.
  - LanceDB database directory: `/home/user/project/lancedb`. Table name: `documents`. The table must have exactly these columns: `id` (string, equal to the MongoDB document `_id`), `text` (string), `category` (string), and `vector` (a fixed-size list of 8 float32 values).
  - Embedding definition (compute exactly this): for a given `text`, take the SHA-256 digest of its UTF-8 bytes, take the first 8 bytes of that digest, and produce the 8-dimensional float32 vector whose component `i` equals `byte[i] / 255.0`.
  - Resume token persistence: after EVERY run persist the current change-stream resume position to the JSON file `/home/user/project/resume_token.json`, even when zero events were processed, so the next run continues from exactly this point and does not miss events that occur between runs.
  - The command must be safely re-runnable: running it again when there are no new events must leave the LanceDB table unchanged and must not error.

