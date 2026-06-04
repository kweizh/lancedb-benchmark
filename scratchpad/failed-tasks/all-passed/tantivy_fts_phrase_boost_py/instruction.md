# Tantivy-backed Full-Text Search with Phrase & Field Boost

## Background
You are building a keyword search utility on top of LanceDB. The corpus is a small catalogue of technical documents (60 rows; each row has an integer `id`, a `title` string, and a `body` string) that has been pre-seeded into a JSON file. Your job is to load the docs into LanceDB, build a **Tantivy-backed** full-text search index on the text columns, and expose a single command-line entrypoint that runs raw Tantivy query strings (phrase queries, field boosts, boolean AND, etc.) and prints the top-k results as JSON.

Unlike the simpler native-Lance FTS path, the Tantivy backend exposes the full Tantivy *query parser* string syntax: quoted phrases such as `"vector database"`, field-scoped terms such as `title:lancedb`, per-field boosts such as `title:lancedb^3`, and boolean operators such as `+rust +tantivy` or `rust AND tantivy`.

## Requirements
- Implement a Python script (`search.py`) under `/home/user/myproject` that, when invoked, runs a Tantivy FTS query against a LanceDB table and prints a JSON list of the top-k hits to stdout.
- The script must be idempotent: on first invocation it should load the 60 seed documents from `/home/user/myproject/seed/docs.json` into a LanceDB table whose name is derived from the current `ZEALT_RUN_ID` environment variable, and then build the Tantivy-backed FTS index. Subsequent invocations should reuse the existing table and index.
- The FTS index **must** be Tantivy-backed (i.e. created with `use_tantivy=True`) and must cover both the `title` and `body` columns so that field-scoped queries work.
- The script must accept two CLI arguments: `--query` (the raw Tantivy query string) and `--k` (the number of results to return, an integer). The query string must be passed through to LanceDB / Tantivy unmodified.
- The JSON written to stdout must be a list of result objects, each containing at least the keys `id` (integer), `title` (string), and `body` (string), ordered by descending relevance (rank 0 = best match).

## Implementation Hints
- Read the seed corpus from the JSON file shipped in the image; do not regenerate it.
- Connect to LanceDB at a writable directory under `/home/user/myproject` (e.g. `lancedb_data/`). Append the value of `ZEALT_RUN_ID` to the table name so parallel runs do not collide on shared storage.
- Build the Tantivy FTS index with multiple text columns in a single `create_fts_index([...], use_tantivy=True, ...)` call so that field-scoped queries can target either column.
- When the candidate passes a query string to `table.search(...)`, LanceDB will route it through Tantivy's query parser; standard Tantivy syntax (quoted phrases, `field:term`, `field:term^boost`, `+term -term`, `AND`/`OR`) is supported.
- Use `.limit(k).to_list()` to materialize the hits and strip any internal score/distance columns before serializing to JSON.
- The Tantivy `tantivy-py` wheel must already be importable in the environment; the candidate does not need to install anything beyond what is provided.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 search.py --query <tantivy-query-string> --k <int>`
- The script reads the current `run-id` from the `ZEALT_RUN_ID` environment variable and uses it to scope the LanceDB table name.
- On the first invocation, the script must create a LanceDB table containing all 60 seeded documents and build a Tantivy-backed FTS index covering both `title` and `body` columns. The index must be Tantivy-backed (created with `use_tantivy=True`).
- The script must print a JSON list of result objects to stdout, in descending relevance order. Each object must include at least `id` (integer), `title` (string), and `body` (string).
- The script must accept Tantivy query parser syntax verbatim, including:
  - Quoted phrase queries (e.g. a query that contains a double-quoted multi-word expression).
  - Field-scoped terms with per-field boosts using the `field:term^weight` syntax.
  - Boolean AND queries using either `+term +term` or the `AND` operator.
- For each query, the top-1 hit must be the document that uniquely satisfies the query semantics over the seeded corpus.
- The script must succeed (exit code 0) for all three query shapes above.

