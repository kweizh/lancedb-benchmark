# GraphQL Search Endpoint over LanceDB

## Background
You must expose a GraphQL API in front of a LanceDB `documents` table that has been pre-seeded for you. The runtime container provides:

- A LanceDB database directory at `/home/user/myproject/data/`.
- A pre-seeded table named `documents_${ZEALT_RUN_ID}` (the run id is provided via the `ZEALT_RUN_ID` environment variable) with 300 rows. Each row has:
  - `id` (int64, monotonic starting at 0)
  - `title` (string)
  - `body` (string)
  - `tags` (list<string>)
  - `published_at` (int64, unix epoch seconds)
  - `vector` (fixed_size_list<float32, 1536>) produced from `body` with the real OpenAI `text-embedding-3-small` model.
- A native Lance FTS index on the `body` column.
- The `OPENAI_API_KEY` environment variable is set at container runtime so you can embed user queries and newly inserted documents on the fly.

## Requirements
Build a FastAPI application that mounts a single GraphQL endpoint at `POST /graphql` (a GET on the same path may return the GraphQL IDE — that is optional) backed by the existing LanceDB table. The GraphQL schema must expose the following operations:

- `vectorSearch(query: String!, k: Int! = 5): [DocResult!]!` — embed `query` with OpenAI `text-embedding-3-small` and run a pure vector search.
- `ftsSearch(query: String!, k: Int! = 5): [DocResult!]!` — run a BM25 full-text search against the FTS-indexed `body` column.
- `hybridSearch(query: String!, k: Int! = 5): [DocResult!]!` — run a hybrid (vector + FTS) search and rerank with the default RRF reranker.
- `filterDocs(tag: String!, afterTs: Int): [Doc!]!` — return every document whose `tags` array contains the given `tag` and (when `afterTs` is provided) whose `published_at >= afterTs`.
- `addDocument(title: String!, body: String!): Doc!` (mutation) — insert one new row. The server assigns a unique `id` (greater than any existing id), computes the OpenAI embedding for `body`, sets `published_at` to the current unix-epoch time in seconds, and sets `tags` to an empty list. The mutation must return the full inserted `Doc`.

GraphQL types:

```graphql
type Doc {
  id: Int!
  title: String!
  body: String!
  tags: [String!]!
  publishedAt: Int!
}

type DocResult {
  id: Int!
  score: Float!
  title: String!
  snippet: String!   # first 120 characters of body
}

type Query {
  vectorSearch(query: String!, k: Int! = 5): [DocResult!]!
  ftsSearch(query: String!, k: Int! = 5): [DocResult!]!
  hybridSearch(query: String!, k: Int! = 5): [DocResult!]!
  filterDocs(tag: String!, afterTs: Int): [Doc!]!
}

type Mutation {
  addDocument(title: String!, body: String!): Doc!
}
```

Validation: when any search resolver is called with `k <= 0` it must fail in a way that surfaces as a populated `errors` array in the GraphQL response (a Python exception inside the resolver is sufficient — GraphQL will convert it). The HTTP response itself should still be `200 OK` with `{"data": ..., "errors": [...]}`.

GraphQL schema introspection (`__schema { queryType { name } }`) must work.

## Implementation Hints
- Use `strawberry-graphql[fastapi]` and `from strawberry.fastapi import GraphQLRouter`, then `app.include_router(graphql_app, prefix="/graphql")`.
- Open the LanceDB table once at startup with `lancedb.connect("/home/user/myproject/data").open_table(f"documents_{os.environ['ZEALT_RUN_ID']}")`.
- Use the OpenAI Python SDK (`openai==1.54.5`, `client.embeddings.create(model="text-embedding-3-small", input=...)`) to embed query strings and new-document bodies. Use `numpy.float32` arrays so they match the table schema.
- For `ftsSearch`, run `table.search(query, query_type="fts").limit(k)`.
- For `hybridSearch`, run `table.search(query_type="hybrid").vector(qvec).text(query).limit(k)` (the default RRF reranker is fine).
- For `filterDocs`, use a SQL `where` clause with `array_has_any(tags, [...])` or `array_has(tags, 'tag')` and combine it with `published_at >= ?` as needed; ordering is not required.
- For `addDocument`, compute the next id from `max(id)+1` (the seed starts at 0 and runs through 299), embed the body once, and call `table.add([{...}])`.
- `snippet` is just `body[:120]`. `score` for vector search can be `1 - _distance` (or just `-_distance`); for FTS / hybrid it is the `_score` / `_relevance_score` column returned by LanceDB. Any monotonic ranking that yields a float is acceptable — the verifier only checks ordering, not absolute values.

## Acceptance Criteria
- Project path: /home/user/myproject
- Start command: uvicorn app:app --host 127.0.0.1 --port 8000
- Port: 8000
- Endpoint: `POST /graphql` accepts a JSON body of the form `{"query": "...", "variables": {...}}` and returns `{"data": ...}` (or `{"data": ..., "errors": [...]}` on resolver failure).
- Response schema:
  - `Doc` has fields `id` (Int), `title` (String), `body` (String), `tags` ([String]), `publishedAt` (Int).
  - `DocResult` has fields `id` (Int), `score` (Float), `title` (String), `snippet` (String, ≤ 120 chars).
- GraphQL operations (all reachable on the same endpoint):
  - `query { vectorSearch(query: $q, k: $k) { id score title snippet } }`
  - `query { ftsSearch(query: $q, k: $k) { id score title snippet } }`
  - `query { hybridSearch(query: $q, k: $k) { id score title snippet } }`
  - `query { filterDocs(tag: $t, afterTs: $ts) { id title tags publishedAt } }`
  - `mutation { addDocument(title: $t, body: $b) { id title body publishedAt tags } }`
- `__schema` introspection returns the query type.
- Invalid input handling: calling any search query with `k = 0` returns a response containing a non-empty `errors` array.
- After `addDocument`, the new document must be discoverable via a subsequent `vectorSearch` whose query string is the same body text (the new id appears in the top-5 results).
- LanceDB table name: `documents_${ZEALT_RUN_ID}` (read `ZEALT_RUN_ID` from the environment).

