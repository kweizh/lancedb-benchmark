# Real-Time Search WebSocket Server (LanceDB)

## Background
Build a WebSocket server that streams real-time search results from a pre-seeded LanceDB table. Clients send JSON query messages and the server streams JSON frames containing incremental top-k hits followed by a final completion frame.

A 200-row LanceDB table named `docs_${ZEALT_RUN_ID}` (32-d random float32 vectors, native Lance FTS index on `text`) is seeded at container startup under `LANCEDB_PATH` (`/app/lancedb_data`). A helper module `embed.py` providing `embed_text(text: str) -> numpy.ndarray` (32-d unit float32, deterministic SHA256-seeded RNG) is provided at `/home/user/myproject/embed.py`. The same `embed_text` function is used by the seed step to compute each row's vector from a per-row seed string and is the function your server must use to convert string queries to 32-d vectors for the `vector` and `hybrid` modes.

## Requirements
- Run a WebSocket server on `0.0.0.0:8765`.
- Accept JSON text messages of shape `{"query": <str>, "k": <int>, "mode": "vector" | "fts" | "hybrid"}` from each connection.
- For each message, run a LanceDB search against the seeded table and stream one JSON frame per top-k hit, in rank order, followed by a single completion frame.
- Each per-hit frame must be a JSON object with exactly the keys `rank` (1-based int), `id` (int), `score` (float), `text` (str).
- The completion frame must be a JSON object with exactly the keys `done` (`true`), `total` (int = number of hits streamed), `elapsed_ms` (number, server-measured wall time for the search in milliseconds).
- Mode behavior:
  - `vector`: embed the query string via `embed_text(...)` and run a pure vector search; `score` is the LanceDB `_distance` value (lower = better).
  - `fts`: run a native Lance FTS query on the `text` column (no Tantivy); `score` is the LanceDB BM25 `_score` value (higher = better).
  - `hybrid`: run a hybrid query combining vector and FTS with the default RRF reranker; `score` is the LanceDB `_relevance_score` value (higher = better).
- Debounce duplicate queries: if the same `(query, k, mode)` triple arrives on the same connection within 100 ms of the previous message of identical shape, ignore it (stream no frames and produce no `done` frame for the duplicate). After 100 ms have elapsed since the previous accepted query of the same shape, the next identical request must be served normally.

## Implementation Hints
- Use the `websockets` Python package (asyncio API) or FastAPI's WebSocket support.
- Connect with `lancedb.connect(os.environ["LANCEDB_PATH"])` and open the table named `docs_${ZEALT_RUN_ID}` where `ZEALT_RUN_ID` is the value of the environment variable.
- For each connection, track the timestamp + payload of the last accepted message to enforce the 100 ms debounce window on a per-connection basis.
- Use `table.search(qvec).limit(k)` for vector mode, `table.search(qstr, query_type="fts").limit(k)` for fts mode, and `table.search(query_type="hybrid").vector(qvec).text(qstr).rerank(RRFReranker()).limit(k)` for hybrid mode.
- Use `to_list()` (or equivalent) to materialize results then stream the frames as text JSON messages.
- Measure `elapsed_ms` around the search call only.

## Acceptance Criteria
- Project path: /home/user/myproject
- Start command: python3 server.py
- Port: 8765 (WebSocket, no TLS)
- WebSocket protocol:
  - Client sends a text message containing a JSON object `{"query": <str>, "k": <int>, "mode": "vector" | "fts" | "hybrid"}`.
  - Server replies with a stream of text messages, each a JSON object. For an accepted request, the server emits exactly `k` per-hit frames (`{"rank": <int>, "id": <int>, "score": <float>, "text": <str>}`) followed by exactly one completion frame (`{"done": true, "total": <int>, "elapsed_ms": <number>}`).
  - Per-hit frames must be emitted in `rank` order starting at 1.
  - For a duplicate `(query, k, mode)` arriving within 100 ms of the previous accepted message on the same connection, the server must emit no frames at all (no per-hit frames and no completion frame).
- Search semantics:
  - `vector` mode: per-hit `score` equals the LanceDB `_distance` value and is monotonically non-decreasing across the stream.
  - `fts` mode: per-hit `score` equals the LanceDB BM25 `_score` value and is monotonically non-increasing across the stream.
  - `hybrid` mode: per-hit `score` equals the LanceDB `_relevance_score` value and is monotonically non-increasing across the stream.
- Data: open the existing table `docs_${ZEALT_RUN_ID}` (read `ZEALT_RUN_ID` from the environment) at the LanceDB path given by the `LANCEDB_PATH` environment variable. Do not recreate, drop, or write to the table.
- Embedding: use the provided `embed_text` function from `/home/user/myproject/embed.py` to convert string queries to 32-d vectors for `vector` and `hybrid` modes.

