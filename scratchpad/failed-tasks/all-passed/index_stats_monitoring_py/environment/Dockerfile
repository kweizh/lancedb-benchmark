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
RUN mkdir -p /home/user/project && \
    mkdir -p /workspace/db && \
    mkdir -p /workspace/output

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 environment for monitoring IVF_PQ index health via table.index_stats()."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 pinned (verified to import cleanly on ubuntu:24.04)
#   - pyarrow==17.0.0 and numpy==1.26.4 pinned for stable wheels
#   - /home/user/project: executor working directory for run.py
#   - /workspace/db: empty LanceDB URI directory (default LANCEDB_URI value)
#   - /workspace/output: directory for the final index_stats.json artifact
