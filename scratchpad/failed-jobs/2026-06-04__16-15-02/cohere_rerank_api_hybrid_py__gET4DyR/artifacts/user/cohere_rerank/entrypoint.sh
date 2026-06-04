#!/usr/bin/env bash
set -u
MARKER=/home/user/cohere_rerank/lancedb/.seeded
if [ ! -f "$MARKER" ]; then
    python3 /home/user/cohere_rerank/_seed.py || {
        echo "[entrypoint] Seed failed; exiting." >&2
        exit 1
    }
fi
trap : TERM INT
sleep infinity & wait
