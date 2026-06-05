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
RUN mkdir -p /app /app/lancedb_data && chmod -R 0755 /app

# Seed script. Runs at container start (entrypoint) once ZEALT_RUN_ID is known.
RUN cat >/usr/local/bin/seed_lancedb.py <<'PYEOF'
"""Seed the LanceDB table phrase_docs_<ZEALT_RUN_ID> and persist ground-truth ids.

Documents are partitioned into three groups:
  - ids 0..9   adjacent ("machine learning" consecutive)
  - ids 10..19 within-3 token gap between "machine" and "learning"
  - ids 20..49 far gap (>= 4 intervening tokens) so slop=3 must NOT match
A subset (deep_ids) contains the word "deep" for the boolean MUST_NOT test.
"""
import json
import os
import sys

import lancedb
import pyarrow as pa

LANCEDB_DIR = "/app/lancedb_data"
GT_PATH = "/app/ground_truth.json"
DONE_FLAG = "/app/.seed_done"


def main() -> int:
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        print("ZEALT_RUN_ID not set; skipping seed.", file=sys.stderr)
        return 0
    if os.path.exists(DONE_FLAG):
        return 0

    adjacent = [
        "the field of machine learning is rapidly evolving across many industries",
        "modern machine learning models require substantial training data to perform",
        "a solid foundation in machine learning enables data driven decisions today",
        "engineers apply machine learning to predict trends and outcomes daily",
        "researchers publish about machine learning at top international conferences yearly",
        "tutorials introducing machine learning are widely available online today everywhere",
        "many startups build products powered by machine learning algorithms now",
        "practical machine learning courses cover regression and classification methods often",
        "open source libraries enable accessible machine learning prototyping for everyone",
        "universities now offer specialized machine learning degrees worldwide regularly",
    ]
    assert len(adjacent) == 10

    within3 = [
        "we combine machine and learning approaches to solve real world problems",
        "effective machine driven learning systems improve over time",
        "this machine of learning is an old metaphor used in mathematics",
        "a machine plus learning equals progress in the data era",
        "modern machine guided learning helps students grasp difficult concepts",
        "the machine assisted active learning combines automation and human insight",
        "the machine intensive online active learning algorithms scale well",
        "old machine first principles aided learning by exposing core dynamics",
        "a machine based supervised statistical learning model underpins applications",
        "the machine assisted online supervised learning workflow operates well",
    ]
    assert len(within3) == 10

    far = [
        "the machine vision team works on segmentation and supervised learning every week",
        "this machine reads the document corpus and trains its learning model nightly",
        "their machine processed the dataset overnight and reported improved learning curves",
        "the machine compiles statistics for the team and forms the learning loop",
        "this machine helps managers run the report and improves their learning pipeline",
        "deep observations on the machine workflow have informed the new learning roadmap",
        "deep insight makes the machine robust before any new learning step occurs",
        "deep changes to the machine architecture happen before the long learning sprint",
        "scientists use the machine to test theories and run extended learning trials",
        "students bring the machine outdoors and document their results during learning weeks",
        "the machine sat idle while teams analyzed results between rounds of learning here",
        "the machine answered every question and was rebooted before the next learning cycle",
        "the machine accelerated quickly and helped the engineers complete the learning module",
        "modern machine instructions help operators reduce delays during long learning sessions",
        "the machine seemed busy throughout the entire afternoon during a long learning shift",
        "deep questions about the machine surface every day during the long learning period",
        "deep impressions about the machine influence the curriculum and the long learning calendar",
        "deep critiques about the machine appeared in journals and triggered new learning trials",
        "deep readings about the machine were considered before initiating the long learning project",
        "the machine gathered samples carefully and produced reports for offline batch learning workflows",
        "the machine triggered alarms briefly and then resumed normal operations during long learning trials",
        "the machine recorded telemetry constantly and submitted detailed metrics across each learning window",
        "the machine handled requests very smoothly even when the queue grew during learning peaks",
        "the machine processed inputs immediately upon arrival without buffering during the busy learning period",
        "the machine reset its counters cleanly and resumed steady operation across the entire learning session",
        "deep speculation about the machine continued well into the night before the long learning summit",
        "the machine paused briefly to recalibrate sensors before resuming a long structured learning routine",
        "the machine emitted a soft hum continuously throughout the night to support nightly learning passes",
        "the machine signaled completion clearly via the dashboard once it finished the daily learning task",
        "the machine performed splendidly during the trial and continued to support extended learning sessions",
    ]
    assert len(far) == 30

    contents = adjacent + within3 + far
    assert len(contents) == 50

    # Verify invariants before writing the table.
    adjacent_ids, within3_ids, far_ids, deep_ids = [], [], [], []
    for i, text in enumerate(contents):
        toks = text.split()
        assert "machine" in toks and "learning" in toks, (
            f"Seed invariant broken: doc {i} missing machine/learning: {text}"
        )
        m_idx = toks.index("machine")
        l_idx = toks.index("learning")
        gap = l_idx - m_idx - 1
        assert l_idx > m_idx, f"machine must precede learning in doc {i}: {text}"
        if i < 10:
            assert gap == 0, f"adjacent doc {i} has gap={gap}: {text}"
            adjacent_ids.append(i)
        elif i < 20:
            assert 1 <= gap <= 3, f"within3 doc {i} has gap={gap}: {text}"
            within3_ids.append(i)
        else:
            assert gap >= 4, f"far doc {i} has gap={gap}: {text}"
            far_ids.append(i)
        if "deep" in toks:
            deep_ids.append(i)
    assert len(deep_ids) >= 5, "Need at least 5 deep-tagged docs"
    # No deep in groups A/B by construction
    for i in adjacent_ids + within3_ids:
        assert "deep" not in contents[i].split(), f"unexpected deep in doc {i}"

    db = lancedb.connect(LANCEDB_DIR)
    table_name = f"phrase_docs_{run_id}"
    if table_name in db.table_names():
        db.drop_table(table_name)

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("content", pa.string()),
    ])
    rows = [{"id": i, "content": contents[i]} for i in range(50)]
    tbl = db.create_table(table_name, data=rows, schema=schema)

    # Native (Lance) FTS index with positional info and stop-word retention.
    tbl.create_fts_index(
        "content",
        use_tantivy=False,
        with_position=True,
        remove_stop_words=False,
    )

    gt = {
        "adjacent_ids": adjacent_ids,
        "within3_ids": within3_ids,
        "far_ids": far_ids,
        "deep_ids": deep_ids,
        "table_name": table_name,
    }
    with open(GT_PATH, "w") as f:
        json.dump(gt, f, indent=2, sort_keys=True)

    with open(DONE_FLAG, "w") as f:
        f.write("ok")

    print(
        f"[seed] table={table_name} rows=50 adjacent={len(adjacent_ids)} "
        f"within3={len(within3_ids)} far={len(far_ids)} deep={len(deep_ids)}"
    )
    return 0


if __name__ == "__main__":
    rc = main()
    # lancedb 0.25.3 can SIGABRT during interpreter teardown; bail out hard.
    os._exit(rc)
PYEOF

RUN cat >/usr/local/bin/entrypoint.sh <<'EOF'
#!/bin/bash
# Seed the LanceDB fixture if ZEALT_RUN_ID is set and we have not yet run.
if [ -n "${ZEALT_RUN_ID}" ] && [ ! -f /app/.seed_done ]; then
    python3 /usr/local/bin/seed_lancedb.py || echo "[entrypoint] seed failed" 1>&2
fi
exec "$@"
EOF
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/seed_lancedb.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB native FTS phrase-proximity & boolean MUST/MUST_NOT task with 50-doc fixture seeded at container start."

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /home/user/myproject: empty project directory for the candidate's solution.py
#   - /app/lancedb_data: LanceDB connection directory
#   - /app/ground_truth.json: ground-truth id partitions written at entrypoint
#   - /usr/local/bin/seed_lancedb.py: idempotent seed script (writes table phrase_docs_${ZEALT_RUN_ID})
#   - /usr/local/bin/entrypoint.sh: runs seed once then execs the CMD sleep loop
