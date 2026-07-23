FROM ubuntu:24.04

ARG POCHI_VERSION

# ---------------------------------------------------------------------------
# Base tooling, Python, ClickHouse server/client, and Python libraries.
# ClickHouse is installed from the official APT repository. All network access
# happens at BUILD time only; the resulting environment runs fully locally.
# ---------------------------------------------------------------------------
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        gnupg apt-transport-https util-linux procps tzdata && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg && \
    ARCH=$(dpkg --print-architecture) && \
    echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] https://packages.clickhouse.com/deb stable main" > /etc/apt/sources.list.d/clickhouse.list && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        clickhouse-server clickhouse-client && \
    pip3 install --break-system-packages \
        pytest-json-ctrf lancedb clickhouse-connect clickhouse-driver numpy pandas && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/hybrid_bridge/data

# ---------------------------------------------------------------------------
# Deterministic data seeding script for both LanceDB and ClickHouse.
# ---------------------------------------------------------------------------
RUN cat > /tmp/seed_data.py <<'EOF'
import hashlib
import random
import re
from datetime import datetime, timedelta

import pyarrow as pa
import lancedb
import clickhouse_connect

LANCEDB_DIR = "/home/user/hybrid_bridge/data/lancedb"
EMBED_DIM = 32
TOKEN_RE = re.compile(r"[a-z0-9]+")

random.seed(1234)


def embed(text):
    vec = [0.0] * EMBED_DIM
    for token in TOKEN_RE.findall(text.lower()):
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % EMBED_DIM
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return [float(v) for v in vec]


# Vocabulary designed so common audio/device terms match many documents.
COMMON = [
    "wireless", "bluetooth", "audio", "device", "sound", "headphones",
    "speaker", "noise", "cancelling", "battery", "portable",
]
EXTRA = [
    "review", "charging", "premium", "budget", "compact", "studio",
    "gaming", "microphone", "earbuds", "surround", "waterproof", "durable",
    "lightweight", "comfort", "bass", "treble", "connector", "firmware",
    "warranty", "package",
]
CATEGORIES = ["audio", "wearable", "accessory", "home", "office"]
REGIONS = ["us-east", "us-west", "eu-central", "ap-south"]
EVENT_TYPES = ["view", "click", "purchase"]

N_DOCS = 400

rows = []
for doc_id in range(1, N_DOCS + 1):
    n_words = random.randint(8, 16)
    words = []
    # Bias toward common terms so full-text recall sets are sizeable.
    for _ in range(n_words):
        if random.random() < 0.6:
            words.append(random.choice(COMMON))
        else:
            words.append(random.choice(EXTRA))
    text = " ".join(words)
    rows.append(
        {
            "id": doc_id,
            "text": text,
            "category": random.choice(CATEGORIES),
            "vector": embed(text),
        }
    )

schema = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("category", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
    ]
)

db = lancedb.connect(LANCEDB_DIR)
if "documents" in db.table_names():
    db.drop_table("documents")
table = db.create_table("documents", data=rows, schema=schema)
table.create_fts_index("text", replace=True)
print("LanceDB documents rows:", table.count_rows())

# ---------------------------------------------------------------------------
# ClickHouse dimension + event tables.
# ---------------------------------------------------------------------------
client = clickhouse_connect.get_client(
    host="127.0.0.1", port=8123, username="default", password="", database="default"
)

client.command("DROP TABLE IF EXISTS events")
client.command("DROP TABLE IF EXISTS users")
client.command(
    "CREATE TABLE users (user_id UInt32, tier String, region String) "
    "ENGINE = MergeTree ORDER BY user_id"
)
client.command(
    "CREATE TABLE events (event_id UInt64, doc_id Int64, user_id UInt32, "
    "event_type String, ts DateTime, value Float64) "
    "ENGINE = MergeTree ORDER BY (doc_id, ts)"
)

N_USERS = 200
user_rows = []
for uid in range(1, N_USERS + 1):
    tier = "premium" if uid % 3 == 0 else "free"
    user_rows.append([uid, tier, random.choice(REGIONS)])
client.insert("users", user_rows, column_names=["user_id", "tier", "region"])

WINDOW_START = datetime(2024, 1, 1, 0, 0, 0)
WINDOW_SPAN_SECONDS = 30 * 24 * 3600  # events spread across January 2024

event_rows = []
event_id = 0
for doc_id in range(1, N_DOCS + 1):
    if random.random() < 0.15:
        n_events = 0
    else:
        n_events = random.randint(1, 40)
    for _ in range(n_events):
        event_id += 1
        uid = random.randint(1, N_USERS)
        ts = WINDOW_START + timedelta(seconds=random.randint(0, WINDOW_SPAN_SECONDS - 1))
        value = round(random.uniform(1.0, 100.0), 4)
        event_rows.append(
            [event_id, doc_id, uid, random.choice(EVENT_TYPES), ts, value]
        )

for i in range(0, len(event_rows), 5000):
    client.insert(
        "events",
        event_rows[i : i + 5000],
        column_names=["event_id", "doc_id", "user_id", "event_type", "ts", "value"],
    )

print("ClickHouse users rows:", client.query("SELECT count() FROM users").result_rows[0][0])
print("ClickHouse events rows:", client.query("SELECT count() FROM events").result_rows[0][0])
EOF

RUN mkdir -p /run/clickhouse-server /var/log/clickhouse-server && \
    chown -R clickhouse:clickhouse /run/clickhouse-server /var/log/clickhouse-server /var/lib/clickhouse && \
    runuser -u clickhouse -- clickhouse-server \
        --config-file=/etc/clickhouse-server/config.xml \
        --daemon --pid-file=/run/clickhouse-server/clickhouse-server.pid && \
    (for i in $(seq 1 60); do curl -sf http://127.0.0.1:8123/ping >/dev/null 2>&1 && break; sleep 1; done) && \
    python3 /tmp/seed_data.py && \
    (test -f /run/clickhouse-server/clickhouse-server.pid && kill "$(cat /run/clickhouse-server/clickhouse-server.pid)" 2>/dev/null || pkill -f clickhouse-server || true) && \
    sleep 5

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + local ClickHouse two-tier retrieval/analytics bridge task environment (Python)."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - ClickHouse server + client installed from the official APT repo (started locally by the entrypoint)
#   - pip3 packages: lancedb, clickhouse-connect, clickhouse-driver, numpy, pandas
#   - /home/user/hybrid_bridge: task project directory
#   - /home/user/hybrid_bridge/data/lancedb: pre-seeded LanceDB 'documents' table (FTS index on 'text')
#   - ClickHouse pre-seeded 'events' and 'users' tables in database 'default'
#   - /entrypoint.sh: sets RUN_ID and starts the local ClickHouse server before the container CMD

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
