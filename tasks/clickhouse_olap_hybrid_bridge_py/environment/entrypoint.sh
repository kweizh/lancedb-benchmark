#!/bin/bash

# Generate RUN_ID
RUN_ID="zr$(tr -dc 'a-z0-9' < /dev/urandom | head -c 8)"
mkdir -p /logs/artifacts
echo "$RUN_ID" > /logs/artifacts/run-id

# Start the local ClickHouse server (data directory is pre-seeded at build time).
mkdir -p /run/clickhouse-server /var/log/clickhouse-server
chown -R clickhouse:clickhouse /run/clickhouse-server /var/log/clickhouse-server /var/lib/clickhouse 2>/dev/null || true
if ! curl -sf http://127.0.0.1:8123/ping >/dev/null 2>&1; then
    runuser -u clickhouse -- clickhouse-server \
        --config-file=/etc/clickhouse-server/config.xml \
        --daemon --pid-file=/run/clickhouse-server/clickhouse-server.pid || true
fi
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8123/ping >/dev/null 2>&1; then break; fi
    sleep 1
done

exec "$@"
