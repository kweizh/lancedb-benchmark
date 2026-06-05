FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pylance==0.21.0 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        "duckdb>=1.0,<2.0" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + DuckDB interop task: DuckDB SQL aggregations over the underlying Lance dataset of a LanceDB table."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - LanceDB stack: lancedb==0.25.3, pylance==0.21.0 (needed for tbl.to_lance() fragment/scanner APIs),
#     pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - DuckDB (>=1.0,<2.0) for SQL aggregations over the Lance dataset
#   - /home/user/myproject: empty project directory for the candidate's solution
