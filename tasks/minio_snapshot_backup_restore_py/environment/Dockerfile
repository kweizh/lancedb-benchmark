FROM ubuntu:24.04

ARG POCHI_VERSION

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        boto3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install a local MinIO server + mc client (S3-compatible object storage for backups).
RUN ARCH=$(dpkg --print-architecture) && \
    curl -fsSL "https://dl.min.io/server/minio/release/linux-${ARCH}/minio" -o /usr/local/bin/minio && \
    curl -fsSL "https://dl.min.io/client/mc/release/linux-${ARCH}/mc" -o /usr/local/bin/mc && \
    chmod +x /usr/local/bin/minio /usr/local/bin/mc && \
    mkdir -p /data

# Local, non-secret dev credentials for the in-container MinIO server.
ENV AWS_ENDPOINT_URL=http://127.0.0.1:9000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV AWS_DEFAULT_REGION=us-east-1
ENV ALLOW_HTTP=true

# Helper that (idempotently) starts the MinIO server and ensures the backup bucket exists.
RUN cat >/usr/local/bin/start-minio.sh <<'EOF'
#!/usr/bin/env bash
export MINIO_ROOT_USER="${AWS_ACCESS_KEY_ID:-minioadmin}"
export MINIO_ROOT_PASSWORD="${AWS_SECRET_ACCESS_KEY:-minioadmin}"
mkdir -p /data
if ! curl -sf http://127.0.0.1:9000/minio/health/ready >/dev/null 2>&1; then
    nohup /usr/local/bin/minio server /data \
        --address 127.0.0.1:9000 --console-address 127.0.0.1:9001 \
        >/var/log/minio.log 2>&1 &
fi
for i in $(seq 1 40); do
    if curl -sf http://127.0.0.1:9000/minio/health/ready >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
/usr/local/bin/mc alias set local http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1 || true
/usr/local/bin/mc mb -p local/lance-backup >/dev/null 2>&1 || true
EOF
RUN chmod +x /usr/local/bin/start-minio.sh

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB backup/restore task with a local in-container MinIO S3 server; Python 3, lancedb 0.25.3, boto3."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: lancedb==0.25.3 (+ pyarrow, numpy, pandas) and boto3 for S3 access
#   - MinIO server + mc client installed to /usr/local/bin; data dir /data; bucket "lance-backup" created at start
#   - start-minio.sh + entrypoint.sh boot MinIO on container start and expose it at http://127.0.0.1:9000
#   - ENV AWS_* + ALLOW_HTTP configure the local S3-compatible endpoint
#   - /home/user/myproject: empty project directory for the candidate's solution.py

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
