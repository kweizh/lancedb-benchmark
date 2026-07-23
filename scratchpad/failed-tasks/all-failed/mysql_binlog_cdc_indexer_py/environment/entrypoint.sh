#!/bin/bash

# Generate RUN_ID
RUN_ID="zr$(tr -dc 'a-z0-9' < /dev/urandom | head -c 8)"
mkdir -p /logs/artifacts
echo "$RUN_ID" > /logs/artifacts/run-id

# Start the local MySQL server (with row-based binlog) and provision the source
# database/table, then hand control to the container CMD.
exec /usr/local/bin/mysql-entrypoint.sh "$@"
