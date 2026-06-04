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
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject

ENV LANCE_TABLE=articles

# Seed the LanceDB fixture: 50 articles (200+ words each), rigged unique tokens
# inserted at deterministic positions for ids 7, 23, 41, plus a native FTS
# index on `body` built with positional information.
RUN cat >/tmp/seed_articles.py <<'PYEOF'
import os
import sys

import numpy as np
import pyarrow as pa
import lancedb


WORDBANK = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "river", "mountain", "forest", "ocean",
    "valley", "ridge", "meadow", "harbor", "bridge", "atlas", "compass",
    "lantern", "kettle", "scroll", "garden", "tunnel", "ember", "willow",
    "thunder", "harvest", "marble", "saffron", "indigo", "crimson", "amber",
    "silver", "copper", "bronze", "granite", "quartz", "linen", "velvet",
    "porcelain", "ivory", "obsidian", "feather", "satellite", "telescope",
    "calendar", "manifest", "voyage", "expedition", "research", "study",
    "engineer", "architect", "scholar", "merchant", "captain", "navigator",
    "wandered", "drifted", "calculated", "measured", "observed", "recorded",
    "compared", "analyzed", "described", "documented", "narrated", "considered",
    "the", "a", "of", "in", "and", "for", "with", "across", "beneath",
    "between", "around", "through", "beyond", "within", "while", "until",
    "before", "after", "above", "below",
]

RIGGED = {
    7: "Chromaspectrum",
    23: "Hyperloomic",
    41: "Zephyrglyph",
}


def make_body(rng, rigged_token=None):
    words = list(rng.choice(WORDBANK, size=240, replace=True))
    if rigged_token is not None:
        # Deterministic insertion point at word index 110 (middle-ish of doc).
        # This guarantees the rigged token appears exactly once and at a
        # position roughly in the middle of a 240-word body.
        words[110] = rigged_token
    body = " ".join(words)
    # Ensure body is well over 200 words.
    assert len(body.split()) >= 220
    return body


def main():
    rng = np.random.default_rng(2026)
    rows = []
    for i in range(50):
        title = f"Article {i:02d}: " + " ".join(rng.choice(WORDBANK, size=4, replace=True))
        body = make_body(rng, rigged_token=RIGGED.get(i))
        rows.append({"id": i, "title": title, "body": body})

    # Sanity: every rigged token must appear exactly once in its target row
    # and nowhere else.
    for rid, token in RIGGED.items():
        for row in rows:
            count = row["body"].lower().count(token.lower())
            if row["id"] == rid:
                assert count == 1, (
                    f"Rigged token {token!r} expected exactly once in id={rid}, got {count}"
                )
            else:
                assert count == 0, (
                    f"Rigged token {token!r} leaked into id={row['id']}, count={count}"
                )

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("body", pa.string()),
    ])
    table_data = pa.Table.from_pylist(rows, schema=schema)

    db_path = "/home/user/myproject/data"
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)
    table_name = os.environ.get("LANCE_TABLE", "articles")
    if table_name in db.table_names():
        db.drop_table(table_name)
    tbl = db.create_table(table_name, data=table_data, mode="overwrite")

    # Native FTS (no Tantivy) with positional info enabled.
    tbl.create_fts_index(
        "body",
        use_tantivy=False,
        with_position=True,
        remove_stop_words=False,
    )

    # Sanity-check FTS surface
    hits = tbl.search("Chromaspectrum", query_type="fts").limit(1).to_list()
    assert hits and hits[0]["id"] == 7, f"FTS sanity check failed: {hits}"

    print(f"Seeded table {table_name!r} with {tbl.count_rows()} rows.")
    # lancedb 0.25.3 occasionally SIGABRTs during interpreter shutdown after
    # write-heavy seeding; exit hard to avoid masking the successful build.
    sys.stdout.flush()
    os._exit(0)


main()
PYEOF
RUN python3 /tmp/seed_articles.py && rm /tmp/seed_articles.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB FTS snippet-highlight task: 50 pre-seeded articles with native positional FTS index in /home/user/myproject/data."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3 packages: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#   - /home/user/myproject: candidate working directory
#   - /home/user/myproject/data: pre-seeded LanceDB database
#   - LANCE_TABLE=articles: name of the fixture table
#   - 50 rigged articles, FTS index on body built with use_tantivy=False, with_position=True
