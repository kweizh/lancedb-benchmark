FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject
RUN mkdir -p /workspace/db && chmod 0755 /workspace/db
RUN mkdir -p /workspace/output && chmod 0755 /workspace/output

ENV LANCEDB_URI=/workspace/db

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 environment for building an IVF_PQ vector index over a 32-d embeddings table."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4 (pinned for stable wheels on Ubuntu 24.04)
#   - /home/user/myproject: candidate working directory
#   - /workspace/db: empty LanceDB directory (LANCEDB_URI default)
#   - /workspace/output: directory where candidate writes ivf_pq.json
