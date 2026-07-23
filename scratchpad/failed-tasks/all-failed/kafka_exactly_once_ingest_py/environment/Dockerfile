FROM ubuntu:24.04

ARG POCHI_VERSION

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep gnupg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Redpanda (Kafka-compatible broker + rpk CLI) from the official apt repo.
RUN curl -1sLf 'https://dl.redpanda.com/nzc4ZYQK3WRGd9sy/redpanda/cfg/setup/bash.deb.sh' | bash && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y redpanda && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python stack: LanceDB + Arrow/pandas + the Kafka client + pytest reporting.
RUN pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        confluent-kafka==2.6.1

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Entrypoint helper that boots a single-node Redpanda broker (dev-container mode)
# and blocks until the Kafka API on 127.0.0.1:9092 is accepting connections.
RUN cat >/usr/local/bin/start-redpanda.sh <<'EOF'
#!/bin/bash
mkdir -p /var/lib/redpanda/data
if bash -c "</dev/tcp/127.0.0.1/9092" 2>/dev/null; then
    echo "redpanda already running"
    exit 0
fi
nohup rpk redpanda start \
    --mode dev-container \
    --smp 1 \
    --kafka-addr PLAINTEXT://0.0.0.0:9092 \
    --advertise-kafka-addr PLAINTEXT://127.0.0.1:9092 \
    >/var/log/redpanda.log 2>&1 &
for i in $(seq 1 90); do
    if bash -c "</dev/tcp/127.0.0.1/9092" 2>/dev/null; then
        echo "redpanda ready on 127.0.0.1:9092"
        exit 0
    fi
    sleep 1
done
echo "redpanda failed to start; log follows:" >&2
cat /var/log/redpanda.log >&2 || true
exit 1
EOF
RUN chmod +x /usr/local/bin/start-redpanda.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + single-node Redpanda (Kafka) environment for an exactly-once Kafka->LanceDB ingestion task (Python)."

CMD ["/bin/bash", "-c", "/usr/local/bin/start-redpanda.sh; trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep gnupg
#   - always required pip3 packages: pytest-json-ctrf
#   - Redpanda broker + rpk CLI installed via apt (Kafka-compatible, runs fully locally, no external network)
#   - Python libs: lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3, confluent-kafka==2.6.1
#   - /usr/local/bin/start-redpanda.sh: boots a single-node dev-container Redpanda on 127.0.0.1:9092 at container start
#   - /home/user/myproject: project directory for the ingestion consumer (candidate builds run.py here)

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
