FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        "lancedb==0.25.3" \
        "pyarrow==17.0.0" \
        "numpy==1.26.4" \
        "pandas==2.2.3" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject
RUN mkdir -p /app && chmod 0755 /app

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 environment for the mmr_diversity_reranking_py task (cosine vector search + MMR re-ranker)."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 pinned per the project's lancedb-task-knowledge memory:
#     reliably imports on ubuntu:24.04 and exposes the table.search().distance_type("cosine") API
#     used by the candidate's MMR candidate-pool step.
#   - pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3 are the standard lancedb-task pin set used by
#     both the candidate's solution.py and the verifier's test_final_state.py (which calls
#     table.to_pandas() to load all 120 rows into numpy for the MMR ground-truth checks).
#   - /home/user/myproject: candidate workspace for solution.py
#   - /app and /app/db: LanceDB database directory written by the candidate's build_dataset()
