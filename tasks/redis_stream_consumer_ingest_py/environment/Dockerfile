FROM ubuntu:24.04

ARG POCHI_VERSION

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        redis-server redis-tools && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        redis==5.0.8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Entrypoint starts a local Redis server (no external network) and keeps the
# container alive so Harbor can exec the task and verification steps.
RUN cat >/usr/local/bin/entrypoint.sh <<'EOF'
#!/bin/bash
redis-server --daemonize yes --save "" --appendonly no --bind 127.0.0.1 --port 6379
for i in $(seq 1 40); do
    if redis-cli -h 127.0.0.1 -p 6379 ping >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done
trap : TERM INT
sleep infinity & wait
EOF
RUN chmod +x /usr/local/bin/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + local Redis Stream durable ingest-consumer task environment (Python)."

CMD ["/usr/local/bin/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - redis-server + redis-tools: local Redis service (started at container boot, bound to 127.0.0.1:6379, no external network)
#   - pip3 packages: lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3, redis==5.0.8
#   - /home/user/myproject: task project directory where the candidate writes run_consumer.py
#   - /usr/local/bin/entrypoint.sh: boots redis-server (daemonized) then holds the container open

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
