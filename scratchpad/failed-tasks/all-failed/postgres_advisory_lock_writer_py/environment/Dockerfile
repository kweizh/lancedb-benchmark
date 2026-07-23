FROM ubuntu:24.04

ARG POCHI_VERSION

ENV LANG=C.UTF-8
ENV LANCEDB_URI=/home/user/myproject/lancedb_data
ENV PG_LOCK_KEY=913571
ENV PGHOST=127.0.0.1
ENV PGPORT=5432
ENV PGUSER=postgres
ENV PGDATABASE=postgres

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        postgresql postgresql-client && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 psycopg2-binary && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure PostgreSQL for password-less local (trust) connections so the
# advisory-lock coordinator is reachable at 127.0.0.1:5432 as user "postgres".
RUN PG_VER="$(ls /etc/postgresql)" && \
    HBA="/etc/postgresql/${PG_VER}/main/pg_hba.conf" && \
    printf '%s\n' \
        "local   all all trust" \
        "host    all all 127.0.0.1/32 trust" \
        "host    all all ::1/128 trust" > "$HBA" && \
    echo "listen_addresses = 'localhost'" >> "/etc/postgresql/${PG_VER}/main/postgresql.conf"

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Build-time seed script: creates the run-scoped empty LanceDB tables.
RUN cat > /usr/local/bin/_seed.py <<'EOF'
import os
import lancedb
import pyarrow as pa

run_id = os.environ["ZEALT_RUN_ID"]
uri = os.environ["LANCEDB_URI"]

db = lancedb.connect(uri)

data_schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("value", pa.int64()),
    pa.field("writer_id", pa.utf8()),
    pa.field("seq", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), 8)),
])
audit_schema = pa.schema([
    pa.field("seq", pa.int64()),
    pa.field("writer_id", pa.utf8()),
    pa.field("value", pa.int64()),
    pa.field("ids_json", pa.utf8()),
])

names = set(db.table_names())
data_name = f"records_{run_id}"
audit_name = f"audit_{run_id}"

if data_name not in names:
    db.create_table(data_name, schema=data_schema)
if audit_name not in names:
    db.create_table(audit_name, schema=audit_schema)

print(f"seeded tables: {data_name}, {audit_name}", flush=True)
# lancedb 0.25.3 can SIGABRT during interpreter teardown after writes; exit hard.
os._exit(0)
EOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + local PostgreSQL advisory-lock coordinator for a serialized single-writer guard task."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - postgresql (local advisory-lock coordinator, trust auth on 127.0.0.1:5432, user "postgres")
#   - pip3 packages: lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 psycopg2-binary
#   - /home/user/myproject: project dir; LanceDB stored at /home/user/myproject/lancedb_data
#   - entrypoint starts PostgreSQL and creates empty run-scoped tables records_<ZEALT_RUN_ID> / audit_<ZEALT_RUN_ID>
#   - ENV: LANCEDB_URI, PG_LOCK_KEY, PGHOST, PGPORT, PGUSER, PGDATABASE

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
