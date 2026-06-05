# Avro Schema Ingestion Pipeline for LanceDB

## Background
A data platform team is migrating an event stream that is currently serialized as [Apache Avro](https://avro.apache.org/) into [LanceDB](https://docs.lancedb.com/) so it can serve as the retrieval layer for a downstream RAG application. Every record carries a precomputed dense vector plus a nested `metadata` record containing the original document author, tags, and a score. Your job is to translate the Avro schema into Arrow, ingest the file into LanceDB, and make sure the resulting table supports both vector search and SQL filtering on the nested author field.

A deterministic Avro file is pre-baked into the image at `/app/data/records.avro` (300 records). The Avro schema is:

```json
{
  "type": "record",
  "name": "Document",
  "namespace": "zealt.lancedb",
  "fields": [
    {"name": "id", "type": "long"},
    {"name": "title", "type": "string"},
    {"name": "vector", "type": {"type": "array", "items": "float"}},
    {"name": "metadata", "type": {
      "type": "record",
      "name": "Metadata",
      "fields": [
        {"name": "author", "type": "string"},
        {"name": "tags", "type": {"type": "array", "items": "string"}},
        {"name": "score", "type": "double"}
      ]
    }}
  ]
}
```

Every `vector` in the file has exactly 32 float entries.

## Requirements
- Implement a single Python module at `/home/user/avro_project/solution.py` that exposes a callable `ingest_avro(avro_path: str, table_name: str) -> None`.
- Read records from the given Avro file using [`fastavro`](https://fastavro.readthedocs.io/).
- Translate the Avro schema into a `pyarrow.Schema` that LanceDB can store natively:
  - `id` -> `int64`
  - `title` -> `string`
  - `vector` -> `fixed_size_list<float32, 32>` (the dimension MUST be derived from / consistent with the data, not hard-coded if the same code is reused on a different Avro file)
  - `metadata.tags` -> `list<string>`
  - The nested `metadata` record may either be **preserved** as a `struct<author: string, tags: list<string>, score: float64>` column, or **flattened** into three top-level columns `metadata_author`, `metadata_tags`, `metadata_score`. Pick one design and document it in your code.
- Bulk-load every row from the Avro file into a single LanceDB table named exactly `table_name` under the database directory `/home/user/avro_project/lance_db`. If the table already exists, overwrite it.
- The table MUST contain exactly the number of records in the Avro file (300 for the bundled fixture), in the same order.
- Also expose a small CLI: `python3 solution.py <avro_path> <table_name>` MUST call `ingest_avro(avro_path, table_name)` and exit 0.

## Implementation Hints
- The Avro `array<float>` type is variable-length on disk; you must explicitly construct a `pyarrow.fixed_size_list(pa.float32(), N)` column to satisfy LanceDB's vector search requirements.
- `fastavro.reader(open(path, "rb"))` yields plain Python dicts; iterate once to collect rows and infer the vector dimension `N` from the first row.
- Use `lancedb.connect("/home/user/avro_project/lance_db")` and `db.create_table(table_name, data=..., schema=..., mode="overwrite")` to ingest in a single shot.
- If you preserve the nested `metadata` record, LanceDB SQL filters address it via dot-notation (e.g., `metadata.author = 'alice'`). If you flatten it, the equivalent filter is `metadata_author = 'alice'`. Either is acceptable; pick one and stick to it.
- The verifier inspects the table schema to figure out which design you chose, then issues the appropriate `where(...)` filter. Make sure your column names and types match one of the two contracts above.

## Acceptance Criteria
- Project path: /home/user/avro_project
- Command: `python3 solution.py <avro_path> <table_name>`
- Module: `/home/user/avro_project/solution.py` MUST define `ingest_avro(avro_path: str, table_name: str) -> None`.
- LanceDB database directory: `/home/user/avro_project/lance_db`
- The verifier passes the table name `records_${ZEALT_RUN_ID}`; read `run-id` from the `ZEALT_RUN_ID` environment variable is NOT required from your code, the verifier provides the suffix as part of `<table_name>`.
- After ingest, the LanceDB table MUST:
  - Exist under `/home/user/avro_project/lance_db` and be openable with `lancedb.connect(...).open_table(<table_name>)`.
  - Contain exactly the same number of rows as the Avro file (300 for the bundled fixture).
  - Have an `id` column of type `int64` and a `title` column of type `string`.
  - Have a vector column literally named `vector` of type `fixed_size_list<float32, 32>`.
  - Have a nested `metadata` struct column with fields `author: string`, `tags: list<string>`, `score: float64` **OR** three flat columns `metadata_author: string`, `metadata_tags: list<string>`, `metadata_score: float64`.
- After ingest, the table MUST be usable for:
  - L2 vector search via `table.search(query_vector).limit(5)` on the `vector` column, returning a list of rows ranked by ascending L2 distance.
  - SQL `where(...)` filtering on the metadata author field (using either `metadata.author = '<name>'` or `metadata_author = '<name>'`, matching the chosen schema design) combined with vector search via `table.search(query_vector).where(<predicate>).limit(5)`.
- `python3 solution.py /app/data/records.avro records_test` MUST exit 0 (the CLI is what the verifier invokes).

