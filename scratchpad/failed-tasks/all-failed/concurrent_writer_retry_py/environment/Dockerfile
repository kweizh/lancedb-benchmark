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
RUN mkdir -p /home/user/myproject && mkdir -p /home/user/lance_db

ENV LANCEDB_URI=/home/user/lance_db
ENV LANCEDB_TABLE=items

# Seed the target LanceDB table with 200 deterministic rows so the candidate
# can focus exclusively on writing the SafeWriter + retry logic.
RUN python3 -c "import os, sys, lancedb, pyarrow as pa; \
uri = '/home/user/lance_db'; \
db = lancedb.connect(uri); \
schema = pa.schema([pa.field('id', pa.int64()), pa.field('value', pa.int64()), pa.field('ts', pa.int64())]); \
tbl = db.create_table('items', schema=schema, mode='overwrite'); \
tbl.add([{'id': i, 'value': 0, 'ts': 0} for i in range(200)]); \
assert tbl.count_rows() == 200, 'seed failed'; \
print('seeded items with', tbl.count_rows(), 'rows'); \
os._exit(0)"

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 concurrent SafeWriter task: pre-seeded 200-row items table at /home/user/lance_db with table name 'items'."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3 for the LanceDB task
#   - /home/user/myproject: empty project directory (candidate writes solution.py here)
#   - /home/user/lance_db: LanceDB database directory, pre-seeded with 200 rows under table 'items'
#   - ENV LANCEDB_URI=/home/user/lance_db, LANCEDB_TABLE=items
