#!/usr/bin/env bash
set -u
RUN_ID="${ZEALT_RUN_ID:-}"
if [ -z "$RUN_ID" ]; then
    echo "[entrypoint] ERROR: ZEALT_RUN_ID is not set." >&2
    exit 1
fi
MARKER="/home/user/myproject/.seeded_${RUN_ID}"
if [ ! -f "$MARKER" ]; then
    python3 /home/user/myproject/_seed.py || {
        echo "[entrypoint] Seed failed; exiting." >&2
        exit 1
    }
fi
trap : TERM INT
sleep infinity & wait
