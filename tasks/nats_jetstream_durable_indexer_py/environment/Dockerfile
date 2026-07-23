FROM ubuntu:24.04

ARG POCHI_VERSION
ARG NATS_VERSION=2.11.6

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
        nats-py==2.9.1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the NATS server (single static binary with JetStream), auto-detecting CPU arch.
RUN ARCH="$(dpkg --print-architecture)" && \
    case "$ARCH" in \
        amd64) NATS_ARCH="amd64" ;; \
        arm64) NATS_ARCH="arm64" ;; \
        *) echo "unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    curl -fsSL "https://github.com/nats-io/nats-server/releases/download/v${NATS_VERSION}/nats-server-v${NATS_VERSION}-linux-${NATS_ARCH}.tar.gz" -o /tmp/nats-server.tar.gz && \
    tar -xzf /tmp/nats-server.tar.gz -C /tmp && \
    cp "/tmp/nats-server-v${NATS_VERSION}-linux-${NATS_ARCH}/nats-server" /usr/local/bin/nats-server && \
    chmod +x /usr/local/bin/nats-server && \
    rm -rf /tmp/nats-server.tar.gz "/tmp/nats-server-v${NATS_VERSION}-linux-${NATS_ARCH}" && \
    nats-server --version

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + NATS JetStream durable exactly-once indexer task; local nats-server with JetStream runs inside the container."

# Start a local NATS server with JetStream in the background, then keep the container alive.
CMD ["/bin/bash", "-c", "mkdir -p /tmp/nats-store && nats-server -js -a 127.0.0.1 -p 4222 -sd /tmp/nats-store > /tmp/nats-server.log 2>&1 & trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: lancedb==0.25.3 (+ pyarrow, numpy, pandas) for the vector store sink
#   - nats-py==2.9.1 Python JetStream client
#   - nats-server v2.11.6 static binary (JetStream) at /usr/local/bin/nats-server, started by CMD on 127.0.0.1:4222
#   - /home/user/myproject: empty project directory where the agent writes solution.py

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
