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
RUN mkdir -p /home/user/myproject && \
    mkdir -p /workspace/db && \
    mkdir -p /workspace/output

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB optimize/cleanup task: candidate must compact fragments and prune old versions with table.optimize(cleanup_older_than=...)."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - LanceDB stack: lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4
#   - /home/user/myproject: workspace for the candidate's Python script
#   - /workspace/db: empty directory that will host the LanceDB store (LANCEDB_URI default)
#   - /workspace/output: empty directory where the candidate writes optimize_state.json
