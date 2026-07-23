FROM ubuntu:24.04

ARG POCHI_VERSION

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Fail the build early if the system SQLite build lacks the FTS5 extension.
RUN python3 -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE _p USING fts5(title, body)'); print('FTS5 available')"

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && mkdir -p /app/data

# --- Seed script: builds the rigged dual-store corpus (SQLite FTS5 + LanceDB). ---
RUN cat > /app/_seed.py <<'PYEOF'
import json
import math
import os
import re
import sqlite3

import lancedb
import numpy as np
import pyarrow as pa

DIM = 32
RRF_K = 60
DATA_DIR = "/app/data"
SQLITE_PATH = os.path.join(DATA_DIR, "corpus.db")
LANCEDB_URI = os.path.join(DATA_DIR, "lancedb")
DISTANCE_METRIC = "cosine"
QUERY_TERM = "photon"

FILLER = [
    "lattice", "tensor", "gauge", "boson", "fermion", "entropy", "plasma",
    "dielectric", "crystal", "phonon", "neutrino", "hadron", "meson", "quark",
    "lepton", "spinor", "fusion", "fission", "isotope", "nucleus", "orbital",
    "valence", "magnet", "current", "voltage", "charge", "spin", "momentum",
    "velocity", "gradient", "vector", "scalar", "matrix", "kernel", "signal",
]


def get_run_id():
    rid = os.environ.get("ZEALT_RUN_ID", "").strip()
    if not rid:
        try:
            with open("/logs/artifacts/run-id") as f:
                rid = f.read().strip()
        except Exception:
            rid = ""
    rid = re.sub(r"[^a-z0-9]", "", rid.lower())
    return rid or "local"


def filler_tokens(n, offset):
    return [FILLER[(offset + i) % len(FILLER)] for i in range(n)]


def make_vec(alpha, slot):
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = float(alpha)
    v[1 + (slot % (DIM - 1))] = math.sqrt(max(0.0, 1.0 - float(alpha) * float(alpha)))
    return v


def build_corpus():
    # (id, title_tokens, body_tokens, alpha)
    photon_docs = [
        (0, ["photon", "photon", "photon"],
            ["photon", "photon", "photon", "photon", "photon"], 0.05),
        (1, ["photon", "dynamics"],
            ["photon", "photon", "field"], 0.90),
        (2, ["quantum", "optics", "review", "article", "introduction"],
            ["photon"] + filler_tokens(60, 3), 0.98),
        (3, ["wave", "particle", "theory"],
            ["photon", "photon", "photon", "photon", "medium"], 0.15),
        (4, ["photon", "spectroscopy", "laboratory"],
            ["calibration", "mirror", "lens", "bench", "optics"], 0.45),
        (5, ["calibration", "laboratory", "bench"],
            ["photon", "mirror", "lens", "optics", "setup", "detector"], 0.35),
        (6, ["emission", "absorption"],
            ["photon", "photon"] + filler_tokens(8, 6), 0.55),
        (7, ["detector", "readout"],
            ["photon"] + filler_tokens(8, 7), 0.25),
    ]

    docs = []
    for doc_id, title, body, alpha in photon_docs:
        docs.append({
            "id": doc_id,
            "title": " ".join(title),
            "body": " ".join(body),
            "alpha": alpha,
        })

    # Filler documents (no "photon" token). id 8 is the vector rank-2 anchor.
    filler_alphas = [0.95] + [round(0.10 + i * 0.02, 2) for i in range(31)]
    for idx, alpha in enumerate(filler_alphas):
        doc_id = 8 + idx
        title = " ".join(filler_tokens(3, doc_id * 2))
        body = " ".join(filler_tokens(12, doc_id * 3 + 1))
        docs.append({"id": doc_id, "title": title, "body": body, "alpha": alpha})

    # Distinct-alpha invariant (guarantees a deterministic vector ordering).
    alphas = [d["alpha"] for d in docs]
    assert len(set(alphas)) == len(alphas), "Non-unique vector alphas would break determinism."

    for d in docs:
        d["vector"] = make_vec(d["alpha"], d["id"]).tolist()
    return docs


def write_sqlite(table, docs):
    con = sqlite3.connect(SQLITE_PATH)
    try:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(f"CREATE VIRTUAL TABLE {table} USING fts5(title, body)")
        con.executemany(
            f"INSERT INTO {table}(rowid, title, body) VALUES (?, ?, ?)",
            [(d["id"], d["title"], d["body"]) for d in docs],
        )
        con.commit()
    finally:
        con.close()


def write_lancedb(table, docs):
    db = lancedb.connect(LANCEDB_URI)
    if table in db.table_names():
        db.drop_table(table)
    schema = pa.schema([
        ("id", pa.int64()),
        ("title", pa.string()),
        ("body", pa.string()),
        ("vector", pa.list_(pa.float32(), DIM)),
    ])
    rows = [{"id": d["id"], "title": d["title"], "body": d["body"], "vector": d["vector"]}
            for d in docs]
    db.create_table(table, data=rows, schema=schema)


def bm25_order(table, tw, bw, limit=50):
    con = sqlite3.connect(SQLITE_PATH)
    try:
        sql = (
            f"SELECT rowid FROM {table} WHERE {table} MATCH ? "
            f"ORDER BY bm25({table}, {float(tw)}, {float(bw)}), rowid LIMIT {int(limit)}"
        )
        rows = con.execute(sql, (QUERY_TERM,)).fetchall()
        scores = con.execute(
            f"SELECT bm25({table}, {float(tw)}, {float(bw)}) FROM {table} "
            f"WHERE {table} MATCH ?", (QUERY_TERM,)
        ).fetchall()
    finally:
        con.close()
    return [int(r[0]) for r in rows], [s[0] for s in scores]


def vector_order(table, qvec, limit=50):
    db = lancedb.connect(LANCEDB_URI)
    tbl = db.open_table(table)
    res = tbl.search(qvec).distance_type(DISTANCE_METRIC).limit(limit).to_list()
    res.sort(key=lambda r: (r["_distance"], int(r["id"])))
    return [int(r["id"]) for r in res]


def weighted_rrf(bm25_ids, vector_ids, k, keyword_weight):
    br = {d: i + 1 for i, d in enumerate(bm25_ids)}
    vr = {d: i + 1 for i, d in enumerate(vector_ids)}
    scored = []
    for d in set(bm25_ids) | set(vector_ids):
        s = 0.0
        if d in br:
            s += keyword_weight * (1.0 / (RRF_K + br[d]))
        if d in vr:
            s += (1.0 - keyword_weight) * (1.0 / (RRF_K + vr[d]))
        scored.append((d, s))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [d for d, _ in scored[:k]]


def self_check(table, qvec):
    bm25_ids, bm25_scores = bm25_order(table, 1.0, 1.0)
    vec_ids = vector_order(table, qvec)

    assert len(bm25_ids) >= 5, f"Need >=5 keyword matches, got {len(bm25_ids)}."
    rounded = [round(float(s), 9) for s in bm25_scores]
    assert len(set(rounded)) == len(rounded), "BM25 scores must be distinct (no ties)."

    assert bm25_ids[0] == 0, f"Expected kw-champion (id 0) as BM25 rank-1, got {bm25_ids[0]}."
    assert vec_ids[0] == 2, f"Expected vec-champion (id 2) as vector rank-1, got {vec_ids[0]}."

    mid = weighted_rrf(bm25_ids, vec_ids, 5, 0.5)
    assert mid[0] == 1, f"Expected balanced doc (id 1) to win at keyword_weight=0.5, got {mid[0]}."
    assert len({bm25_ids[0], vec_ids[0], mid[0]}) == 3, "Extreme/intermediate winners must differ."

    title_ids, _ = bm25_order(table, 8.0, 1.0)
    body_ids, _ = bm25_order(table, 1.0, 8.0)
    assert title_ids != body_ids, "Column weighting must change BM25 order."
    assert title_ids.index(4) < title_ids.index(5), "Title weighting must lift the title-only doc."
    assert body_ids.index(5) < body_ids.index(4), "Body weighting must lift the body-only doc."


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    run_id = get_run_id()
    table = f"docs_{run_id}"
    docs = build_corpus()

    write_sqlite(table, docs)
    write_lancedb(table, docs)

    qvec = make_vec(1.0, 0).tolist()  # canonical query vector == e0
    self_check(table, qvec)

    with open(os.path.join(DATA_DIR, "config.json"), "w") as f:
        json.dump({
            "sqlite_path": SQLITE_PATH,
            "lancedb_uri": LANCEDB_URI,
            "table_name": table,
            "vector_dim": DIM,
            "distance_metric": DISTANCE_METRIC,
            "rrf_k": RRF_K,
        }, f, indent=2)

    with open(os.path.join(DATA_DIR, "queries.json"), "w") as f:
        json.dump([{"query": QUERY_TERM, "vector": qvec}], f)

    print(f"Seeded dual-store corpus into table '{table}' ({len(docs)} docs).")
    os._exit(0)


main()
PYEOF

# Build-time smoke test: verify the rigged invariants hold, then clear the data
# so the real corpus is (re)seeded at container start with the runtime run id.
RUN ZEALT_RUN_ID=zrbuildcheck python3 /app/_seed.py && rm -rf /app/data/*

# --- Entrypoint: idempotently seed the run-id-scoped corpus, then stay alive. ---
RUN cat > /app/entrypoint.sh <<'SHEOF'
#!/bin/bash
set -u
mkdir -p /app/data
MARK="/app/data/.seeded_${ZEALT_RUN_ID:-local}"
if [ ! -f "$MARK" ]; then
  if python3 /app/_seed.py; then
    touch "$MARK"
  else
    echo "WARNING: corpus seeding failed" >&2
  fi
fi
trap : TERM INT
sleep infinity & wait
SHEOF
RUN chmod +x /app/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="Dual-store search task: SQLite FTS5 BM25 + LanceDB vectors fused with weighted RRF."

CMD ["/app/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - target/task pip3 packages: lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#   - build-time check that the system SQLite ships the FTS5 extension
#   - /app/_seed.py: deterministic rigged dual-store corpus seeder (SQLite FTS5 + LanceDB)
#   - /app/entrypoint.sh: seeds tables named docs_${ZEALT_RUN_ID} at startup, writes /app/data/config.json and /app/data/queries.json
#   - /home/user/myproject: empty project directory where the agent writes solution.py

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
