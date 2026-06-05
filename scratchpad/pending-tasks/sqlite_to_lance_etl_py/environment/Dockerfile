FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        openai==1.54.5 \
        httpx==0.27.2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

LABEL Author="generate-harbor-tasks"
LABEL Description="Incremental SQLite -> LanceDB ETL task. Solver implements sync() that bridges a SQLite articles table and a LanceDB table using real OpenAI text-embedding-3-small embeddings."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - LanceDB stack pinned to lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3 for stable wheels on ubuntu:24.04
#   - openai==1.54.5 for real text-embedding-3-small calls at runtime
#   - sqlite3 is part of the Python standard library
#   - /home/user/myproject: empty project directory where the candidate places solution.py and the verifier seeds source SQLite + lancedb_data/
