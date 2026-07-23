#!/bin/bash

# Generate RUN_ID
RUN_ID="zr$(tr -dc 'a-z0-9' < /dev/urandom | head -c 8)"
mkdir -p /logs/artifacts
echo "$RUN_ID" > /logs/artifacts/run-id

# Start the local PostgreSQL advisory-lock coordinator.
PG_VER="$(ls /etc/postgresql)"
pg_ctlcluster "$PG_VER" main start || pg_ctlcluster "$PG_VER" main restart
for i in $(seq 1 60); do
    if pg_isready -h 127.0.0.1 -p 5432 >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Create the run-scoped empty LanceDB tables (idempotent). ZEALT_RUN_ID is
# injected into the container environment by the run framework.
if [ -n "$ZEALT_RUN_ID" ]; then
    python3 /usr/local/bin/_seed.py || true
fi

exec "$@"
