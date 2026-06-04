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
RUN mkdir -p /home/user/rrf_hybrid && chmod 0755 /home/user/rrf_hybrid
RUN mkdir -p /workspace && chmod 0777 /workspace

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 hybrid (vector + native FTS) search with RRF reranker on Ubuntu 24.04."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4 for the hybrid + RRF reranker task
#   - /home/user/rrf_hybrid: project directory the candidate works in
#   - /workspace: writable workspace for the LanceDB store (/workspace/db) and the
#     output artifact (/workspace/output/hybrid_rrf.json)
