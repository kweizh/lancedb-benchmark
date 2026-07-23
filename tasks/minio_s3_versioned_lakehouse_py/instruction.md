# Versioned LanceDB Lakehouse on S3-Compatible Object Storage (MinIO)

## Background
You are building a small "multimodal lakehouse" with **LanceDB OSS** where the table lives on an **S3-compatible object store** instead of the local filesystem. A **MinIO** server is already running inside this container and exposes an S3 API at `http://127.0.0.1:9000`. LanceDB talks to it through the `s3://` URI scheme plus `storage_options` (custom endpoint, region, and static credentials).

On top of this you must exercise LanceDB's **versioning / time-travel** features: every ingest, append, and delete produces a new immutable version, and you can query the table "as of" any earlier version. Because the data lives on object storage, you must also run LanceDB's manual maintenance (`optimize()`) to compact the many small data files that appending produces.

## Environment (fixed facts)
- MinIO S3 endpoint: `http://127.0.0.1:9000`
- Region: `us-east-1`
- Access key: `minioadmin`
- Secret key: `minioadmin`
- A bucket named `lancedb-lakehouse` already exists (created at container start).
- Store the LanceDB database at the URI `s3://lancedb-lakehouse/db` and name the table `documents`.
- Deterministic input fixtures are baked into the image (JSON, UTF-8):
  - `/app/fixtures/base.json` — the initial rows.
  - `/app/fixtures/added.json` — rows to append in a second write.
  - `/app/fixtures/queries.json` — an object mapping a query name to an 8-float query vector.
- Every fixture row is an object with keys `id` (int), `text` (str), `category` (str), and `vector` (list of 8 floats). Vectors are 8-dimensional and must be stored in a vector column suitable for L2 nearest-neighbor search.

## Requirements
Implement a rerunnable CLI at `/home/user/myproject/run.py` with two sub-commands:

1. `build` — connect to the MinIO-backed S3 store using `storage_options` and construct the versioned table in this exact order:
   - Create the `documents` table from `/app/fixtures/base.json` (overwrite any existing table so the command is rerunnable).
   - Append the rows from `/app/fixtures/added.json`.
   - Delete every row whose `category` equals `legacy`.
   - Run `optimize()` on the table to compact the data files produced by the appends (this creates at least one further version).
   - Record the LanceDB integer version number captured **immediately after** each of the first three steps, plus the final latest version, into `/home/user/myproject/versions.json`.

2. `query --query <name> --version <int> --k <int>` — connect, open the table **as of the given version number** (time-travel), run an L2 vector search with the query vector named `<name>` from `/app/fixtures/queries.json`, and print the result.

## Implementation Hints
- Project path: `/home/user/myproject`
- Connect with `lancedb.connect("s3://lancedb-lakehouse/db", storage_options={...})`. For a plain-HTTP MinIO endpoint you must supply the custom endpoint, the region, the access key, the secret key, and enable non-TLS access; without the correct endpoint override the data will not land on MinIO.
- LanceDB assigns integer version numbers starting at 1; read the current version after each write and use the checkout / time-travel API to read historical versions.
- `versions.json` must be a JSON object with exactly the integer-valued keys `base`, `added`, `deleted`, and `latest` (the version numbers right after the create, the append, the delete, and after `optimize()` respectively). `latest` must be strictly greater than `deleted`.
- `Command: python3 run.py build` — this must finish with **exit status 0**. Note: when LanceDB writes to object storage the Python process can abort during interpreter shutdown, so make sure the command terminates cleanly with status 0.
- `Command: python3 run.py query --query <name> --version <int> --k <int>` — print exactly one line of JSON to stdout, an object with exactly the keys `version` (the int you were given) and `ids` (a JSON array of the matching row `id` integers ordered nearest-first by ascending L2 distance, breaking ties by ascending `id`). Print nothing else on stdout.
- Time-travel correctness matters: a row appended in a later version must not appear when you query an earlier version, and a row deleted in a later version must still appear when you query a version from before the delete.

