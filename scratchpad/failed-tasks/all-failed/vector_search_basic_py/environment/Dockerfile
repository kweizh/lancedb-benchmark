FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        numpy==1.26.4 \
        pyarrow==17.0.0 \
        lancedb==0.25.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

# Task workspace: the candidate writes solution.py here, and /workspace/db
# and /workspace/output are populated at runtime.
RUN mkdir -p /workspace/myproject /workspace/db /workspace/output && \
    chmod -R 0755 /workspace

LABEL Author="generate-harbor-tasks"
LABEL Description="Basic LanceDB vector-search task: lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4 on ubuntu:24.04."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 + pyarrow==17.0.0 + numpy==1.26.4 for the vector_search_basic_py task
#   - /workspace/myproject: project path where the candidate writes solution.py
#   - /workspace/db: default LanceDB store (LANCEDB_URI default)
#   - /workspace/output: directory where the candidate writes results.json
