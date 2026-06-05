# gRPC Vector Search Microservice over LanceDB

## Background
You are building a gRPC microservice that wraps a LanceDB vector index so that other services can run nearest-neighbour search remotely. The container already has LanceDB, `grpcio`, `grpcio-tools`, and `pytest-xprocess` installed. A 500-row, 24-dimensional `documents` table has been seeded at `/home/user/myproject/data/lancedb` at build time. Each row has fields `id` (int64), `title` (string), `category` (string), and `vector` (24-dim float32). The expected fixture for verification lives at `/home/user/myproject/.expected.json` and is read-only.

Your job is to design the gRPC service contract, generate the Python stubs, and implement a server that serves search-with-optional-where requests against the pre-seeded table.

## Requirements
- Author a Protocol Buffers definition at `/home/user/myproject/proto/search.proto` declaring:
  - `package search;`
  - A `SearchService` with one unary RPC: `Search(SearchRequest) returns (SearchResponse)`.
  - `SearchRequest`:
    - `repeated float vector = 1;`
    - `int32 k = 2;`
    - `string where_clause = 3;`
  - `SearchResponse`:
    - `repeated Hit hits = 1;`
  - `Hit`:
    - `int64 id = 1;`
    - `float score = 2;`
    - `string title = 3;`
- Generate Python stubs from the proto using `grpcio-tools` so that `import search_pb2` and `import search_pb2_grpc` both succeed from the project root. The generated files MUST live in `/home/user/myproject/` (i.e., the project root must contain `search_pb2.py` and `search_pb2_grpc.py`).
- Implement the server in `/home/user/myproject/server.py`:
  - Opens the pre-seeded LanceDB table at `/home/user/myproject/data/lancedb` (table name `documents`). Do NOT recreate or overwrite the table.
  - Implements `SearchServiceServicer.Search` so it:
    - Runs a LanceDB vector search using `request.vector` as the query.
    - Applies `request.where_clause` as a LanceDB SQL filter when the string is non-empty. When it is empty, no filter is applied.
    - Uses `request.k` as the result limit.
    - Returns a `SearchResponse` whose `hits` list is in LanceDB retrieval order. Each `Hit.id` is the row's `id`, `Hit.title` is the row's `title`, and `Hit.score` is a non-negative number derived from the LanceDB distance (lower distance => smaller score is fine; the verifier only checks that the IDs and titles match ground truth).
  - Binds a `grpc.server(...)` to `0.0.0.0:50051` and serves until terminated (keep the process alive with `server.wait_for_termination()`).
- The server MUST be startable with `python3 server.py` from `/home/user/myproject`.

## Implementation Hints
- Run the stub generation with `python3 -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/search.proto` from the project root so that the import paths in `search_pb2_grpc.py` resolve correctly.
- Open the table once at server startup with `lancedb.connect("/home/user/myproject/data/lancedb").open_table("documents")` and reuse the handle inside the RPC.
- LanceDB's query builder accepts an SQL filter via `.where("<expr>")` and a list/array vector via `.search(vec).limit(k)`. Combine these to honour `where_clause` when it is non-empty.
- A simple `concurrent.futures.ThreadPoolExecutor(max_workers=4)` is sufficient for the `grpc.server` thread pool argument.
- Be careful: the proto field for the filter is `where_clause`. After stub generation, Python attribute access is `request.where_clause`.

## Acceptance Criteria
- Project path: /home/user/myproject
- Start command: python3 server.py
- Port: 50051 (plain gRPC, no TLS)
- gRPC service contract:
  - Package: `search`
  - Service: `SearchService`
  - Unary RPC: `Search(SearchRequest) returns (SearchResponse)`
  - Message fields (numbers and types are part of the contract):
    - `SearchRequest { repeated float vector = 1; int32 k = 2; string where_clause = 3; }`
    - `SearchResponse { repeated Hit hits = 1; }`
    - `Hit { int64 id = 1; float score = 2; string title = 3; }`
- The Python stubs `search_pb2.py` and `search_pb2_grpc.py` MUST be importable from the project root.
- The server MUST connect to the pre-seeded LanceDB table at `/home/user/myproject/data/lancedb` (table name `documents`) and serve real LanceDB search results (no in-memory mocks, no hard-coded hit lists).
- The server MUST honour both branches of the request:
  - When `where_clause` is empty, return the top-k nearest neighbours of `vector` from the full table.
  - When `where_clause` is non-empty, the returned hits MUST all satisfy the filter (e.g., `category = 'alpha'`), and the order MUST match LanceDB's filtered search order.

