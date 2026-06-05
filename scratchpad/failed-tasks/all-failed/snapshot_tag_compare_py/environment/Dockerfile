FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        "lancedb==0.33.0" \
        "pyarrow==17.0.0" \
        "numpy==1.26.4" \
        "pandas==2.2.3" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject
RUN mkdir -p /app && chmod 0755 /app

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.33.0 environment for the snapshot_tag_compare_py task (table tags + time-travel diff)."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.33.0 chosen because the table.tags.* CRUD API is fully exposed in that release
#     (confirmed working in the project's lancedb-task-knowledge memory). The default 0.25.3 pin
#     is skipped here because tag-API verification on 0.25.3 is not memoised as safe.
#   - pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3 are the standard lancedb-task pin set
#   - /home/user/myproject: candidate workspace for solution.py
#   - /app and /app/db: LanceDB database directory written by the candidate's solution
