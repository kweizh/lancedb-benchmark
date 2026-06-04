FROM ubuntu:24.04

ARG OPENAI_API_KEY

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        langchain==0.3.27 \
        langchain-core==0.3.78 \
        langchain-community==0.3.30 \
        langchain-openai==0.3.35 \
        langchain-text-splitters==0.3.11 \
        "openai>=1.104.2,<2.0" \
        tiktoken && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject /home/user/myproject/db

# Seed the internal_kb table with real OpenAI text-embedding-3-small vectors at build time.
RUN cat >/tmp/seed_internal_kb.py <<'PYEOF'
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa
from openai import OpenAI

DB_DIR = "/home/user/myproject/db"
TABLE = "internal_kb"
MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

DOCS = [
    # Helios database (13 docs)
    ("helios-01", "Helios", "The Helios database supports full ACID transactions across all read and write operations."),
    ("helios-02", "Helios", "Helios is a distributed SQL database optimized for high-throughput OLTP workloads."),
    ("helios-03", "Helios", "Helios uses the Raft consensus protocol to replicate writes across nodes."),
    ("helios-04", "Helios", "Helios offers strong consistency guarantees and serializable isolation by default."),
    ("helios-05", "Helios", "Helios provides built-in encryption at rest using AES-256 across all storage layers."),
    ("helios-06", "Helios", "The Helios database was first released to the public in March 2021."),
    ("helios-07", "Helios", "The official Helios command-line interface is named heliosctl."),
    ("helios-08", "Helios", "A single Helios cluster can scale to a maximum of 64 nodes."),
    ("helios-09", "Helios", "Helios queries can be profiled with the EXPLAIN ANALYZE command to inspect execution plans."),
    ("helios-10", "Helios", "Helios offers a free community edition that supports clusters of up to 5 nodes."),
    ("helios-11", "Helios", "Helios stores its tables on disk using a columnar format for analytical scans."),
    ("helios-12", "Helios", "Helios enterprise pricing is based on the total managed data volume in terabytes."),
    ("helios-13", "Helios", "Official Helios documentation is hosted at helios-db.io and includes API references."),

    # Aurora analytics (13 docs)
    ("aurora-01", "Aurora", "Aurora analytics enforces a maximum data retention period of 365 days for all time-series metrics."),
    ("aurora-02", "Aurora", "Aurora supports streaming ingestion via first-class Kafka and Kinesis connectors."),
    ("aurora-03", "Aurora", "Aurora dashboards are defined declaratively using Vega-Lite specifications."),
    ("aurora-04", "Aurora", "Aurora analytics was first released in November 2020."),
    ("aurora-05", "Aurora", "A single Aurora node can ingest up to 2 million events per second under typical workloads."),
    ("aurora-06", "Aurora", "Aurora's primary query language is called AQL, short for Aurora Query Language."),
    ("aurora-07", "Aurora", "Aurora provides automatic anomaly detection on numeric metrics out of the box."),
    ("aurora-08", "Aurora", "Aurora can export scheduled reports as either PDF documents or CSV files."),
    ("aurora-09", "Aurora", "Aurora integrates natively with Slack to deliver alerts to specific channels."),
    ("aurora-10", "Aurora", "The smallest supported production deployment of Aurora is a 3-node cluster."),
    ("aurora-11", "Aurora", "Aurora supports row-level security through configurable RBAC policies."),
    ("aurora-12", "Aurora", "Aurora data is sharded by tenant_id by default to provide multi-tenant isolation."),
    ("aurora-13", "Aurora", "Aurora pricing is based on monthly ingestion volume measured in gigabytes."),

    # Borealis CRM (14 docs)
    ("borealis-01", "Borealis", "Borealis CRM is priced at $99 per seat per month on the standard plan."),
    ("borealis-02", "Borealis", "Borealis offers a 30-day free trial with no credit card required to sign up."),
    ("borealis-03", "Borealis", "Borealis supports custom workflows through its visual builder called Flowstone."),
    ("borealis-04", "Borealis", "Borealis offers data migration assistants for Salesforce, HubSpot, and Pipedrive."),
    ("borealis-05", "Borealis", "Borealis CRM was launched in January 2022."),
    ("borealis-06", "Borealis", "The Borealis enterprise plan includes single sign-on (SSO) via SAML 2.0."),
    ("borealis-07", "Borealis", "The Borealis mobile app is available for both iOS and Android devices."),
    ("borealis-08", "Borealis", "On the starter plan, a Borealis workspace supports up to 10,000 contacts."),
    ("borealis-09", "Borealis", "Borealis's lead scoring engine uses gradient-boosted decision tree models."),
    ("borealis-10", "Borealis", "Borealis offers built-in calendar integrations with both Google Calendar and Microsoft Outlook."),
    ("borealis-11", "Borealis", "Borealis is GDPR compliant and holds a SOC 2 Type II attestation."),
    ("borealis-12", "Borealis", "The Borealis public API uses REST conventions and OAuth 2.0 for authentication."),
    ("borealis-13", "Borealis", "Outbound emails from Borealis are sent through a managed SendGrid relay."),
    ("borealis-14", "Borealis", "Borealis ships with a Kanban-style pipeline view for tracking deals across stages."),
]


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required at build time to seed the internal_kb table.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    texts = [d[2] for d in DOCS]

    # Embed in a single batch — 40 short docs is well within model limits.
    resp = client.embeddings.create(model=MODEL, input=texts)
    vectors = [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]
    assert len(vectors) == len(DOCS), "embedding count mismatch"
    for v in vectors:
        assert v.shape == (EMBED_DIM,), f"unexpected embedding shape {v.shape}"

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("product", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
    ])

    rows = []
    for (doc_id, product, text), vec in zip(DOCS, vectors):
        rows.append({
            "id": doc_id,
            "product": product,
            "text": text,
            "vector": vec.tolist(),
        })

    table_pa = pa.Table.from_pylist(rows, schema=schema)

    db = lancedb.connect(DB_DIR)
    if TABLE in db.table_names():
        db.drop_table(TABLE)
    tbl = db.create_table(TABLE, data=table_pa, mode="create")

    n = tbl.count_rows()
    print(f"Seeded {TABLE} with {n} docs.")
    # Sanity: verify cosine-search returns the ACID doc for a Helios transaction query.
    qresp = client.embeddings.create(model=MODEL, input=["Does Helios support ACID transactions?"])
    qvec = np.asarray(qresp.data[0].embedding, dtype=np.float32)
    top = tbl.search(qvec).distance_type("cosine").limit(1).to_list()
    assert top, "search returned no results"
    assert "ACID" in top[0]["text"], f"unexpected top doc: {top[0]['text']!r}"
    print("Sanity search OK.")


if __name__ == "__main__":
    main()
    # lancedb 0.25.3 occasionally SIGABRTs during interpreter shutdown after writes.
    os._exit(0)
PYEOF

RUN OPENAI_API_KEY="${OPENAI_API_KEY}" python3 /tmp/seed_internal_kb.py && \
    rm -f /tmp/seed_internal_kb.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LangChain agent task with a pre-seeded LanceDB internal_kb table (40 docs, text-embedding-3-small)."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip packages: pytest-json-ctrf, lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4,
#                   pandas==2.2.3, langchain==0.3.27, langchain-core==0.3.78,
#                   langchain-community==0.3.30, langchain-openai==0.3.35,
#                   langchain-text-splitters==0.3.11, openai==1.54.5, httpx==0.27.2, tiktoken
#   - /home/user/myproject: project directory
#   - /home/user/myproject/db: LanceDB database with seeded internal_kb table (40 docs about
#     fictional products Helios database, Aurora analytics, Borealis CRM) using real
#     OpenAI text-embedding-3-small embeddings.
