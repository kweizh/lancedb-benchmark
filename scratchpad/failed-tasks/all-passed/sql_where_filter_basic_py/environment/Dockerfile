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

RUN mkdir -p /workspace/solution /workspace/db /workspace/output && \
    chmod -R 0755 /workspace

ENV LANCEDB_URI=/workspace/db

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 environment for basic SQL where-clause filtering on a local /workspace/db table."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - LanceDB stack: lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /workspace/solution: project path for the candidate's run.py
#   - /workspace/db: LanceDB database directory (LANCEDB_URI default)
#   - /workspace/output: directory for the filter_results.json artifact
