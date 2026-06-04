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

# Pre-create the workspace layout. /workspace/db must NOT be pre-populated;
# the candidate solution is responsible for creating the LanceDB dataset.
RUN mkdir -p /workspace && \
    mkdir -p /workspace/output && \
    chmod 0755 /workspace /workspace/output

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 native (non-Tantivy) FTS task environment on Ubuntu 24.04."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4 for the LanceDB native FTS task
#   - /workspace: project root (LANCEDB_URI default base)
#   - /workspace/output: directory for fts_results.json output artifact
