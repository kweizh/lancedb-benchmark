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
RUN mkdir -p /home/user/myproject
RUN mkdir -p /workspace/db && mkdir -p /workspace/output

ENV LANCEDB_URI=/workspace/db

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB distance-metric selection task environment with lancedb 0.25.3, pyarrow 17.0.0, and numpy 1.26.4."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4 for the candidate solution and verifier
#   - /home/user/myproject: candidate working directory
#   - /workspace/db: default LanceDB URI location (overridable via LANCEDB_URI)
#   - /workspace/output: where the candidate writes distances.json
