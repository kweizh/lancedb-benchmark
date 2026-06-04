FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

RUN mkdir -p /workspace && \
    mkdir -p /workspace/db && \
    mkdir -p /workspace/output && \
    chmod -R 0755 /workspace

ENV LANCEDB_URI=/workspace/db

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB versioning/time-travel task environment with lancedb 0.25.3 and a /workspace project directory."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3 for LanceDB versioning APIs
#   - /workspace: project root
#   - /workspace/db: default LanceDB on-disk URI (also exported via LANCEDB_URI env var)
#   - /workspace/output: directory where the candidate writes version_state.json
