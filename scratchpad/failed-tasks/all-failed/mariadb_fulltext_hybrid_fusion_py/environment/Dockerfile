FROM ubuntu:24.04

ARG POCHI_VERSION

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        mariadb-server mariadb-client && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 pymysql==1.1.1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && mkdir -p /app /app/lancedb

ENV MARIADB_HOST=127.0.0.1 \
    MARIADB_PORT=3306 \
    MARIADB_USER=appuser \
    MARIADB_PASSWORD=apppass \
    MARIADB_DATABASE=hybrid \
    LANCEDB_PATH=/app/lancedb

# Seed script: builds a corpus rigged so one doc is keyword-only, one is
# vector-only, and one ranks high in BOTH engines. Vectors are deterministic
# via a seeded numpy RNG. Writes ground-truth to /app/expected.json.
RUN cat >/app/_seed.py <<'SEEDEOF'
import json
import os

import lancedb
import numpy as np
import pyarrow as pa
import pymysql

RUN_ID = os.environ["ZEALT_RUN_ID"]
TABLE = f"docs_{RUN_ID}"
LANCEDB_PATH = os.environ.get("LANCEDB_PATH", "/app/lancedb")
DB = os.environ.get("MARIADB_DATABASE", "hybrid")
HOST = os.environ.get("MARIADB_HOST", "127.0.0.1")
PORT = int(os.environ.get("MARIADB_PORT", "3306"))
USER = os.environ.get("MARIADB_USER", "appuser")
PASSWORD = os.environ.get("MARIADB_PASSWORD", "apppass")

DIM = 32
SEED = 2026
rng = np.random.default_rng(SEED)

BOTH_ID = 1
KEYWORD_ONLY_ID = 2
VECTOR_ONLY_ID = 3
FILLER_START = 100
N_FILLERS = 60

QUERY_TEXT = "quantum photonic lattice calibration"


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


u = np.zeros(DIM)
u[0] = 1.0

# both: essentially identical direction to the query
nc = np.zeros(DIM)
nc[1:] = rng.standard_normal(DIM - 1)
both_vec = unit(u + 0.01 * unit(nc))

# vector-only: close but clearly behind "both"
nb = np.zeros(DIM)
nb[1:] = rng.standard_normal(DIM - 1)
vec_only_vec = unit(u + 0.05 * unit(nb))

# keyword-only: orthogonal to the query direction -> farthest in cosine space
keyword_only_vec = np.zeros(DIM)
keyword_only_vec[1] = 1.0

# fillers: partial alignment in dims >=2 so cosine(u, filler) in [0.10, 0.70]
alphas = np.linspace(0.10, 0.70, N_FILLERS)
filler_ids = []
filler_vecs = []
for i in range(N_FILLERS):
    a = float(alphas[i])
    noise = np.zeros(DIM)
    noise[2:] = rng.standard_normal(DIM - 2)
    noise = unit(noise)
    v = unit(a * u + np.sqrt(max(1.0 - a * a, 1e-9)) * noise)
    filler_vecs.append(v)
    filler_ids.append(FILLER_START + i)

# Fail-fast geometry invariants
c_both = cos(both_vec, u)
c_vec = cos(vec_only_vec, u)
c_kw = cos(keyword_only_vec, u)
max_filler = max(cos(v, u) for v in filler_vecs)
assert c_both > c_vec > max_filler, (c_both, c_vec, max_filler)
assert c_kw < min(cos(v, u) for v in filler_vecs), (c_kw, "keyword-only not farthest")

# Text bodies: only BOTH and KEYWORD_ONLY contain query tokens.
both_body = (
    "Quantum photonic lattice calibration protocol. The photonic lattice "
    "enables quantum calibration, and this quantum photonic lattice supports "
    "precise calibration of the lattice."
)
keyword_only_body = (
    "Routine hardware calibration notes for the assembly line, including "
    "maintenance schedules and inspection checklists for the shop floor."
)
vector_only_body = (
    "Optical resonator tuning guide describing waveguide interference, beam "
    "alignment, and mode coupling for integrated circuits."
)
filler_vocab = [
    "logistics inventory shipping warehouse routing schedule",
    "finance budget invoice ledger revenue forecast",
    "weather climate rainfall temperature humidity forecast",
    "cooking recipe kitchen ingredient flavor seasoning",
    "travel itinerary passport airport luggage hotel",
    "sports tournament league scoreboard training stadium",
    "history archive manuscript timeline dynasty museum",
    "biology cell organism ecosystem enzyme membrane",
]

rows_mysql = [(BOTH_ID, both_body), (KEYWORD_ONLY_ID, keyword_only_body),
              (VECTOR_ONLY_ID, vector_only_body)]
for idx, fid in enumerate(filler_ids):
    text = f"General reference entry {fid} covering {filler_vocab[idx % len(filler_vocab)]}."
    rows_mysql.append((fid, text))

# ---- MariaDB: text + FULLTEXT index ----
conn = pymysql.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, database=DB)
try:
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS `{TABLE}`")
        cur.execute(
            f"CREATE TABLE `{TABLE}` ("
            f"  id INT PRIMARY KEY,"
            f"  body TEXT,"
            f"  FULLTEXT KEY ft_body (body)"
            f") ENGINE=InnoDB"
        )
        cur.executemany(f"INSERT INTO `{TABLE}` (id, body) VALUES (%s, %s)", rows_mysql)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id FROM `{TABLE}` "
            f"WHERE MATCH(body) AGAINST (%s IN NATURAL LANGUAGE MODE) "
            f"ORDER BY MATCH(body) AGAINST (%s IN NATURAL LANGUAGE MODE) DESC, id ASC",
            (QUERY_TEXT, QUERY_TEXT),
        )
        fts_ids = [int(r[0]) for r in cur.fetchall()]
    assert set(fts_ids) == {BOTH_ID, KEYWORD_ONLY_ID}, fts_ids
    assert fts_ids[0] == BOTH_ID, fts_ids
finally:
    conn.close()

# ---- LanceDB: precomputed vectors ----
all_ids = [BOTH_ID, KEYWORD_ONLY_ID, VECTOR_ONLY_ID] + filler_ids
all_vecs = [both_vec, keyword_only_vec, vec_only_vec] + filler_vecs
data = [{"id": int(i), "vector": [float(x) for x in v]}
        for i, v in zip(all_ids, all_vecs)]

db = lancedb.connect(LANCEDB_PATH)
if TABLE in db.table_names():
    db.drop_table(TABLE)
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])
tbl = db.create_table(TABLE, data=data, schema=schema)

vres = tbl.search([float(x) for x in u]).distance_type("cosine").limit(1000).to_list()
vec_ids = [int(r["id"]) for r in vres]
assert vec_ids[0] == BOTH_ID, vec_ids[:5]
assert vec_ids[1] == VECTOR_ONLY_ID, vec_ids[:5]
assert KEYWORD_ONLY_ID not in vec_ids[:10], vec_ids[:10]

expected = {
    "both_id": BOTH_ID,
    "keyword_only_id": KEYWORD_ONLY_ID,
    "vector_only_id": VECTOR_ONLY_ID,
    "query_text": QUERY_TEXT,
    "query_vector": [float(x) for x in u],
}
with open("/app/expected.json", "w") as f:
    json.dump(expected, f)

print("SEED OK", TABLE, "fts=", fts_ids, "vec_top3=", vec_ids[:3])
# lancedb 0.25.3 can SIGABRT during interpreter teardown; exit hard after writes.
os._exit(0)
SEEDEOF

# Entrypoint: start MariaDB, provision user/db, seed once per run-id, then block.
RUN cat >/usr/local/bin/entrypoint.sh <<'ENTRYEOF'
#!/bin/bash
if [ -z "${ZEALT_RUN_ID}" ] && [ -f /logs/artifacts/run-id ]; then
  export ZEALT_RUN_ID="$(cat /logs/artifacts/run-id)"
fi
mkdir -p /run/mysqld /app/lancedb
chown -R mysql:mysql /run/mysqld /var/lib/mysql
if [ ! -d /var/lib/mysql/mysql ]; then
  mariadb-install-db --user=mysql --datadir=/var/lib/mysql --skip-test-db >/dev/null 2>&1
fi
mariadbd --user=mysql --datadir=/var/lib/mysql \
         --socket=/run/mysqld/mysqld.sock --bind-address=127.0.0.1 --port=3306 \
         >/var/log/mariadb.log 2>&1 &
for i in $(seq 1 60); do
  if mariadb-admin --socket=/run/mysqld/mysqld.sock ping >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
mariadb --socket=/run/mysqld/mysqld.sock -u root <<SQL
CREATE DATABASE IF NOT EXISTS hybrid;
CREATE USER IF NOT EXISTS 'appuser'@'127.0.0.1' IDENTIFIED BY 'apppass';
CREATE USER IF NOT EXISTS 'appuser'@'localhost' IDENTIFIED BY 'apppass';
GRANT ALL PRIVILEGES ON hybrid.* TO 'appuser'@'127.0.0.1';
GRANT ALL PRIVILEGES ON hybrid.* TO 'appuser'@'localhost';
FLUSH PRIVILEGES;
SQL
if [ -n "${ZEALT_RUN_ID}" ]; then
  MARKER="/app/.seeded_${ZEALT_RUN_ID}"
  if [ ! -f "${MARKER}" ]; then
    if python3 /app/_seed.py; then
      touch "${MARKER}"
    fi
  fi
fi
trap : TERM INT; sleep infinity & wait
ENTRYEOF
RUN chmod +x /usr/local/bin/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="Local MariaDB FULLTEXT + LanceDB vector store for RRF hybrid-fusion task."

CMD ["/bin/bash", "-c", "/usr/local/bin/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - MariaDB server (local, 127.0.0.1:3306) with a FULLTEXT-indexed corpus table
#   - LanceDB 0.25.3 (+ pyarrow, numpy, pandas) on-disk vector store at /app/lancedb
#   - pymysql client for MariaDB access
#   - /app/_seed.py: deterministic corpus seeder (rigged keyword-only / vector-only / both docs)
#   - /usr/local/bin/entrypoint.sh: starts MariaDB, provisions user/db, seeds per run-id
#   - /app/expected.json: ground-truth ids + query written by the seeder
#   - /home/user/myproject: project directory for the candidate's solution.py

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
