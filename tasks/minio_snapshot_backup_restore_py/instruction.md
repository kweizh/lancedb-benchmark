# LanceDB Point-in-Time Snapshot Backup & Restore to MinIO

## Background
LanceDB stores each table as a self-contained, versioned dataset directory on disk. Operations such as `add`, `update`, and `delete` append new immutable fragments and manifest files, so every historical version stays reproducible through `table.list_versions()`, `table.checkout(v)`, and `table.checkout_latest()`.

Your job is to build a point-in-time backup/restore tool that snapshots a local LanceDB table into a local **MinIO** (S3-compatible) object store, and later rebuilds a fully working local LanceDB table from that snapshot — **including the complete version history**, not just the newest rows.

A MinIO server is already running locally inside this container and an empty bucket named `lance-backup` has already been created for you. There is no access to AWS or any external network; everything is local.

## Requirements
Create a Python module at `/home/user/myproject/solution.py` that exposes exactly two functions:

- `backup(local_table_path, s3_uri)` — replicate the LanceDB table stored at `local_table_path` (a `*.lance` dataset directory on the local filesystem) to the object-store location `s3_uri`. The snapshot MUST capture the whole dataset so that **every** version is restorable, not only the latest data.
- `restore(s3_uri, new_local_table_path)` — rebuild a local LanceDB `*.lance` dataset directory at `new_local_table_path` from the snapshot previously written to `s3_uri`. After a successful restore, the table must open cleanly with LanceDB and expose the **identical latest data AND the identical version history** (same versions, same per-version row counts) as the original.
- `restore` MUST fail cleanly by raising an exception when `s3_uri` points at a prefix that holds no backup, and it MUST NOT leave behind a half-written or openable table at `new_local_table_path` in that case.

## Implementation Hints
- A LanceDB table on disk is a directory tree of manifest and fragment files (e.g. a `_versions/` directory of per-version manifests plus data fragments). A faithful snapshot copies the entire dataset tree so that all versions survive; exporting only the current rows and re-inserting them into a fresh table would silently discard the version history and fail verification.
- Think of the snapshot as a recursive object-by-object copy of the dataset directory into the bucket (each local file becomes one object under the prefix), and the restore as the reverse copy back onto the local filesystem, preserving the relative directory layout.
- Connect to the S3-compatible endpoint using the credentials provided as environment variables: `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, and `ALLOW_HTTP=true` (the endpoint is plain HTTP, not HTTPS). The `boto3` library is installed for talking to the object store.
- `s3_uri` always has the form `s3://<bucket>/<prefix>`.
- A restored table written to `<dir>/<name>.lance` must be openable via `lancedb.connect("<dir>").open_table("<name>")`, and calling `list_versions()` / `checkout(v)` on it must reproduce the original versions with their original row counts.
- Importing the module must have no side effects (do not connect or copy anything at import time). The verifier imports `solution` from `/home/user/myproject` and calls `backup(...)` and `restore(...)` directly.
- Project path: /home/user/myproject
- Interface: `backup(local_table_path: str, s3_uri: str) -> None` and `restore(s3_uri: str, new_local_table_path: str) -> None`.

