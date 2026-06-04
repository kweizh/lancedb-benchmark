FROM ubuntu:24.04

# Base system + python tooling + LanceDB stack + pieces needed by the verifier.
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        boto3==1.35.49 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the MinIO server and the `mc` client. Auto-detect amd64/arm64.
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        amd64) MINIO_ARCH=amd64 ;; \
        arm64) MINIO_ARCH=arm64 ;; \
        *) echo "Unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /usr/local/bin/minio "https://dl.min.io/server/minio/release/linux-${MINIO_ARCH}/minio"; \
    curl -fsSL -o /usr/local/bin/mc "https://dl.min.io/client/mc/release/linux-${MINIO_ARCH}/mc"; \
    chmod 0755 /usr/local/bin/minio /usr/local/bin/mc

# MinIO daemon credentials. NOTE: these are the credentials for the MinIO server
# baked into this image only — they are NOT production secrets. The MinIO daemon
# requires concrete credentials at startup; we are NOT mocking LanceDB or S3.
ENV MINIO_ROOT_USER=minioadmin \
    MINIO_ROOT_PASSWORD=minioadmin12345 \
    MINIO_ACCESS_KEY=minioadmin \
    MINIO_SECRET_KEY=minioadmin12345

# Default workspace layout expected by the task.
RUN mkdir -p /home/user && chmod 0755 /home/user && \
    mkdir -p /workspace/output && chmod -R 0755 /workspace && \
    mkdir -p /data

# Entrypoint script: start MinIO in the background, wait for readiness,
# pre-create the `lance-bucket` bucket via mc, then exec the container CMD.
RUN cat >/usr/local/bin/start-minio-entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -u

LOG_DIR=/var/log/minio
mkdir -p "${LOG_DIR}"

# Start MinIO server in the background.
nohup minio server /data --address ":9000" --console-address ":9001" \
    >"${LOG_DIR}/minio.log" 2>&1 &

# Wait for MinIO to become healthy (max 60s).
for i in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:9000/minio/health/ready >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Pre-create the bucket idempotently.
mc alias set localminio http://127.0.0.1:9000 "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" >/dev/null 2>&1 || true
mc mb --ignore-existing localminio/lance-bucket >/dev/null 2>&1 || true

# Hand off to the container CMD.
exec "$@"
EOF
RUN chmod 0755 /usr/local/bin/start-minio-entrypoint.sh

# Healthcheck so external orchestration can verify MinIO is up.
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=6 \
    CMD curl -fsS http://127.0.0.1:9000/minio/health/ready || exit 1

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB on Ubuntu 24.04 backed by an in-container MinIO S3 server with a pre-created 'lance-bucket' bucket."

ENTRYPOINT ["/usr/local/bin/start-minio-entrypoint.sh"]
CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3 packages: pytest-json-ctrf, lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, boto3==1.35.49
#   - /usr/local/bin/minio: MinIO server (S3-compatible object store) — auto-detected amd64/arm64 build
#   - /usr/local/bin/mc: MinIO client used to create the `lance-bucket` bucket at startup
#   - /usr/local/bin/start-minio-entrypoint.sh: ENTRYPOINT that boots MinIO, waits for readiness, creates the bucket, then execs CMD
#   - /data: MinIO data directory (single-node, non-distributed)
#   - /workspace and /workspace/output: project + output directories used by the task
#   - ENV: MINIO_ROOT_USER, MINIO_ROOT_PASSWORD (server-side daemon credentials)
#   - ENV: MINIO_ACCESS_KEY, MINIO_SECRET_KEY (mirror credentials read by the task script)
