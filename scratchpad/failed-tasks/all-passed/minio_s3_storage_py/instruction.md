# LanceDB on S3-compatible Object Storage (MinIO)

## Background

LanceDB supports any S3-compatible object store as a first-class storage backend. In this task you will run LanceDB against a local **MinIO** server (already running inside the container on port `9000`) instead of the local filesystem. This is the same pattern users follow to point LanceDB at AWS S3, Cloudflare R2, Tigris, or any other S3-compatible API.

MinIO is started automatically by the container entrypoint, with a pre-created bucket named `lance-bucket`. The MinIO credentials are exposed to your script through environment variables (`MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`).

Your script must connect LanceDB to MinIO over the S3 protocol, create a table, seed it with deterministic data, run a vector search, and write the results to a JSON file.

## Requirements

- Connect LanceDB to MinIO using the `s3://lance-bucket/` URI and the appropriate `storage_options` (custom endpoint, region, access key, secret key, and HTTP-not-HTTPS allowance).
- Create a table named `vectors_s3` with the following Arrow schema:
  - `id`: `int64`
  - `payload`: `string`
  - `vector`: `fixed_size_list<float32>[8]`
- Seed the table with **16 deterministic rows**. Each row has:
  - `id = i` for `i in 0..15`
  - `payload = f"row-{i:02d}"`
  - `vector` = `numpy.random.default_rng(11).standard_normal((16, 8)).astype("float32")[i]`
- Build a query vector using the same generator state: take the next `standard_normal(8)` sample from a *fresh* `default_rng(11)` after consuming the 16 seed rows (i.e., call `rng = numpy.random.default_rng(11); rng.standard_normal((16, 8))` and then `query = rng.standard_normal(8).astype("float32")`).
- Run a top-3 vector search (default L2 metric) and write the results to `/workspace/output/s3_results.json`.

## Implementation Hints

- Use `lancedb.connect("s3://lance-bucket/", storage_options={...})` and pass `endpoint`, `region`, `aws_access_key_id`, `aws_secret_access_key`, and `allow_http`.
- The MinIO endpoint is `http://127.0.0.1:9000`. The region can be anything S3 expects, e.g. `us-east-1`.
- Read `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` from `os.environ`.
- Use `pyarrow` to construct the schema (`pa.list_(pa.float32(), 8)` for the fixed-size vector field) or pass a list-of-dicts and let LanceDB infer the schema; either way the resulting field must be `fixed_size_list<float32>[8]`.
- Search with `table.search(query).limit(3).to_list()`.
- Write the JSON file as a list of `{"id": int, "payload": str, "_distance": float}` objects in the order returned by `to_list()` (i.e., nearest first).
- Make sure `/workspace/output/` exists before writing.

## Acceptance Criteria

- Project path: `/workspace`
- Ensure the script is executed and the artifact exists.
- Output file: `/workspace/output/s3_results.json`
- The output file is a JSON array of exactly 3 objects, each containing the keys `id` (integer), `payload` (string), and `_distance` (number).
- The table `vectors_s3` exists in the LanceDB database hosted on MinIO at `s3://lance-bucket/`, with exactly 16 rows and the schema described above.
- The top-3 ids in the output match the deterministic ground truth that can be recomputed from `numpy.random.default_rng(11)`.

