FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        "lancedb==0.25.3" \
        "pyarrow==17.0.0" \
        "numpy==1.26.4" \
        "pydantic>=2.7,<3" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

RUN mkdir -p /workspace /workspace/db /workspace/output && \
    chmod -R 0755 /workspace

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 environment for defining a Pydantic LanceModel schema, ingesting deterministic rows, and running a vector search."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 (Python SDK), pyarrow==17.0.0, numpy==1.26.4, pydantic v2 for the LanceModel schema task
#   - /workspace: project root
#   - /workspace/db: empty LanceDB directory (used via $LANCEDB_URI)
#   - /workspace/output: directory where the candidate writes movies_top5.json and run.log
