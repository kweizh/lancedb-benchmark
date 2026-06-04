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

# Workspace skeleton expected by the task and verifier.
RUN mkdir -p /workspace/project /workspace/output && \
    chmod 0755 /workspace /workspace/project /workspace/output

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB Python environment for the count_rows_with_filter_py task."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - LanceDB Python SDK pinned to 0.25.3, with pyarrow==17.0.0 and numpy==1.26.4
#   - /workspace/project: empty project directory the executor uses
#   - /workspace/output: empty output directory where counts.json will be written
#   - LanceDB database is created on demand at LANCEDB_URI (default /workspace/db)
