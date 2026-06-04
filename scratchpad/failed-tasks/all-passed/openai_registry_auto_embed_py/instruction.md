# Auto-Embedding with the LanceDB OpenAI Embedding Registry

## Background
LanceDB ships an embedding-function registry that wires a hosted embedding model (e.g. OpenAI `text-embedding-3-small`) directly into a Pydantic `LanceModel` schema. Once configured, inserts are automatically vectorized server-side by the registry and string queries are embedded automatically too. You will build a small Python script that demonstrates this end-to-end against a real OpenAI endpoint.

## Requirements
- Use a fresh on-disk LanceDB at the location given by the `LANCEDB_URI` environment variable (default `/workspace/db`).
- Use the OpenAI embedding provider exposed through `lancedb.embeddings.get_registry().get("openai")` with model `text-embedding-3-small`.
- Define a `LanceModel` table schema named `docs` with three columns: `text` (string, `SourceField`), `label` (string), and `vector` (`Vector(func.ndims())`, `VectorField`).
- Insert exactly 8 rows where ONLY `text` and `label` are provided (no precomputed vectors): the registry must compute the embeddings via the real OpenAI API.
- After ingestion, run the vector search with a STRING query: `table.search("vector database with sql filtering").limit(3).to_list()` and persist the top-3 results.

## Implementation Hints
- Read `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) from the environment; the `openai` provider in the registry will pick it up.
- The `Vector(func.ndims())` field type means you do not specify a hard-coded dimensionality; let the registry report it.
- The 8 seed rows must include at least one row whose text is `"LanceDB supports SQL filtering on vector queries."` so that the chosen query has a clear top-1 match.
- Write the top-3 results to `/workspace/output/registry_results.json` with this shape: `{"top3_texts": [str, str, str], "top3_labels": [str, str, str]}`. The order MUST be the order returned by `to_list()`.

## Acceptance Criteria
- Project path: /workspace
- Ensure the script is executed and the artifacts exist.
- Log file: /workspace/output/run.log (free-form; must exist and be non-empty after the script runs)
- LanceDB database directory: `${LANCEDB_URI}` (default `/workspace/db`) must contain a table named `docs`.
- The `docs` table must have exactly 8 rows.
- The `vector` column in `docs` must be a fixed-size list of 1536 floats (the native dimension of `text-embedding-3-small`).
- Output file: `/workspace/output/registry_results.json` must exist and be valid JSON with the following shape:
  ```json
  {
    "top3_texts": [string, string, string],
    "top3_labels": [string, string, string]
  }
  ```
- `top3_texts` and `top3_labels` MUST each have length 3.
- The first element of `top3_texts` MUST contain the substring `SQL filtering` (case-sensitive), reflecting the SQL-filtering-themed seed row that is the strongest semantic match for the query.

