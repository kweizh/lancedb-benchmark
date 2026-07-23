# gRPC Client-Streaming Vector Ingestion + Search Microservice (LanceDB)

## Background
You are building a small gRPC microservice that ingests vector records into a LanceDB table and serves nearest-neighbor search over them. Clients push large numbers of records efficiently using a **client-streaming** RPC, and the server buffers the incoming stream into fixed-size batches before writing them to LanceDB. A separate unary RPC answers similarity-search queries.

Everything runs locally inside the container: the gRPC server listens on `127.0.0.1:50051` and LanceDB is used as an embedded (on-disk) library. No external network calls, hosted APIs, or model/dataset downloads are allowed.

## Requirements
- Define a Protocol Buffers service (proto3) with:
  - A **client-streaming** RPC `IngestVectors(stream VectorRecord) returns (IngestSummary)`. The client streams many `VectorRecord` messages; the server buffers them into fixed-size batches and writes each batch to LanceDB, then returns a single `IngestSummary`.
  - A **unary** RPC `Search(SearchRequest) returns (SearchResponse)` that returns the ranked nearest neighbors of a query vector.
- Compile the protobufs into Python stubs (use `grpcio-tools`).
- Implement the gRPC server (`server.py`) and a Python client helper (`client.py`).
- Persist all ingested records into a LanceDB table storing 16-dimensional `float32` vectors.
- Validate the vector dimension of every streamed record. If a record's vector dimension does not match the table's, abort the RPC with gRPC status `INVALID_ARGUMENT` and do **not** leave any partial write from that RPC in the table.

## Implementation Hints
- Use `grpcio` + `grpcio-tools` to define and compile the service; use `lancedb` (embedded, on-disk) for storage and `pyarrow` for the table schema (a fixed-size list of 16 `float32` values for the vector column).
- The server must accumulate streamed records and flush them to LanceDB in batches; the batch size is fixed at **100** records (flush every 100 records, then flush any remainder when the stream ends).
- Reject bad-dimension records using the gRPC context abort mechanism so the client observes a proper status code.
- Vector search with no index performs exact brute-force kNN; use the default L2 (Euclidean) distance and return the closest vectors first.
- Project path: `/home/user/myproject`
- Start command: `python3 server.py` (the server must print a line containing `gRPC server listening on 127.0.0.1:50051` once it is ready to accept connections).
- Server address: `127.0.0.1:50051`.
- Storage: connect LanceDB at the directory given by env var `LANCEDB_PATH` (default `/home/user/myproject/lance_data`). Use a table named `vectors_${ZEALT_RUN_ID}` (read `ZEALT_RUN_ID` from the environment). The table must exist and be **empty** when the server starts (create/overwrite an empty table with the correct schema on startup).
- The record message must carry an integer `id`, a repeated `float` `vector`, and a string `metadata`. `IngestSummary` must expose three integer counts: the number of records **received**, the number of rows **written** to LanceDB, and the number of **batches** flushed.
- `client.py` must expose exactly these two functions used by callers:
  - `ingest_vectors(records, address="127.0.0.1:50051")` where `records` is an iterable of dicts with keys `id` (int), `vector` (list of float), `metadata` (str). It streams them via the client-streaming RPC and returns a dict `{"received": int, "written": int, "batches": int}`. On a server-side abort it must let the underlying `grpc.RpcError` propagate to the caller (do not swallow it).
  - `search(query_vector, k, address="127.0.0.1:50051")` where `query_vector` is a list of float and `k` is an int. It returns a list (length at most `k`) of dicts each with keys `id` (int), `distance` (float), and `metadata` (str), ordered by ascending distance (nearest neighbor first).

