# Arrow Flight Proxy in Front of a LanceDB Vector Table

## Background
A local LanceDB database at `/home/user/flight_proxy/lancedb` already contains a table named `documents` with **200 rows**. Each row has the following columns:

- `id` (`string`): unique document identifier (e.g. `doc_000`, `doc_001`, ...).
- `text` (`string`): a short description of the document.
- `embedding` (`fixed_size_list<float, 32>`): a deterministic precomputed 32-dimensional `float32` vector (no model inference at runtime — vectors were seeded with a fixed `numpy` RNG when the container was built).

Your job is to expose that table through a minimal **Apache Arrow Flight** server so that remote clients can run vector top-k searches over gRPC without ever speaking the LanceDB API directly.

## Requirements
- Implement a Python script `server.py` that:
  1. Opens the existing LanceDB database at `/home/user/flight_proxy/lancedb` and the existing `documents` table (do NOT recreate or re-seed it — it is already populated).
  2. Defines a subclass of `pyarrow.flight.FlightServerBase`.
  3. Implements `do_get(self, context, ticket)` where the ticket bytes are a JSON object of the form `{"vector": [<32 floats>], "k": <int>}`. The handler MUST:
     - Parse the JSON payload from `ticket.ticket`.
     - Run a LanceDB vector search using the supplied `vector` and `k` against the `documents` table.
     - Return the top-`k` results to the client as an Arrow stream (`pyarrow.flight.RecordBatchStream`) preserving the columns `id` (string), `text` (string), `embedding` (`fixed_size_list<float, 32>`), and `_distance` (float32 — the LanceDB-reported distance).
  4. Binds to `grpc://0.0.0.0:8815` and starts serving (via `serve()` on the server instance) when run as `python3 server.py`.

## Implementation Hints
- Use `pyarrow.flight.FlightServerBase` and `pyarrow.flight.RecordBatchStream` (see the Apache Arrow Python cookbook: <https://arrow.apache.org/cookbook/py/flight.html>).
- The ticket payload is a UTF-8-encoded JSON blob — decode `ticket.ticket` and parse it.
- LanceDB's `Table.search(query_vector).limit(k)` returns a query builder you can materialize with `.to_arrow()` to get a `pyarrow.Table` ready for `RecordBatchStream`.
- Do not introduce additional authentication, TLS, or middleware — plain unencrypted gRPC on `0.0.0.0:8815` is expected.

## Acceptance Criteria
- Project path: /home/user/flight_proxy
- Start command: python3 server.py
- Port: 8815 (plain gRPC, no TLS)
- Wire protocol: Apache Arrow Flight (`grpc://0.0.0.0:8815`).
- Endpoint:
  - `do_get(ticket)`:
    - Ticket body: a UTF-8 JSON object with the shape `{"vector": [float, ... (length 32)], "k": int}`.
    - Response: an Arrow stream whose schema contains the fields `id` (string), `text` (string), `embedding` (`fixed_size_list<float, 32>`), and `_distance` (float32), in any order.
    - Response row count: equal to `min(k, 200)` rows, ordered by ascending `_distance` (i.e. nearest neighbors first), matching what a direct `table.search(vector).limit(k)` call against the `documents` table would return.
- The server MUST query the real LanceDB table at `/home/user/flight_proxy/lancedb` on every `do_get` call. No precomputed result caches, no mocked search.

