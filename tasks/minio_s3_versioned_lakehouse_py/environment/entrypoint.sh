#!/bin/bash

# Generate RUN_ID
RUN_ID="zr$(tr -dc 'a-z0-9' < /dev/urandom | head -c 8)"
mkdir -p /logs/artifacts
echo "$RUN_ID" > /logs/artifacts/run-id

# Start the local, in-container MinIO S3 server and pre-create the bucket.
export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin
mkdir -p /data/minio
if ! curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null 2>&1; then
  nohup /usr/local/bin/minio server /data/minio \
      --address 127.0.0.1:9000 --console-address 127.0.0.1:9001 \
      >/var/log/minio.log 2>&1 &
fi
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
/usr/local/bin/mc alias set local http://127.0.0.1:9000 minioadmin minioadmin >/dev/null 2>&1 || true
/usr/local/bin/mc mb local/lancedb-lakehouse >/dev/null 2>&1 || true

exec "$@"
