FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        openai==1.54.5 \
        tiktoken \
        langchain==0.3.27 \
        langchain-core==0.3.78 \
        langchain-text-splitters==0.3.11 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject
RUN mkdir -p /app/docs

# ---- Bake fixture: 3 source docs (3000-5000 chars each) + 10 anchor queries.
# Two of the queries are rigged so that overlap=0 misses them: the entity name
# sits at the tail of one chunk and the answer value sits at the head of the
# next chunk; only an overlapping chunk can contain both.
RUN cat >/app/_build_fixture.py <<'PYEOF'
"""Deterministic fixture builder.

Generates three source documents and the 10-query anchor list. Two anchor
queries are rigged: a unique entity name sits at the END of one o=0 chunk
and the corresponding answer phrase sits at the START of the next o=0
chunk. The ground-truth span covers only the answer phrase, so the chunk
that contains the entity (which scores well in vector search) has zero
span overlap with the ground truth, and the chunk that contains the
answer phrase lacks the entity name so it scores poorly. At overlap=150
a single chunk contains both entity and answer, yielding a hit.

The script aligns rigged anchors by padding filler tokens until the
splitter places a boundary at the desired position, then asserts the
rigging invariants before writing files.
"""
import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = "/app/docs"
ANCHORS_PATH = "/app/anchors.json"
os.makedirs(DOCS_DIR, exist_ok=True)

FILLER_WORDS = [
    "ocean", "coral", "shore", "tides", "algae", "kelps", "reefs", "whale",
    "shoal", "delta", "river", "fjord", "marsh", "depth", "shelf", "wreck",
    "crust", "earth", "crest", "valve", "grain", "wheel", "stone", "amber",
    "flint", "stars", "comet", "novas", "quark", "orbit", "lunar", "solar",
    "ridge", "delta", "epoch", "sands", "dunes", "verge", "plume", "stove",
]


def filler(n_chars, salt=0):
    """Deterministic filler text of EXACTLY n_chars chars, ending on a clean
    word boundary (no padding chars). May return up to n_chars; falls short
    by < 6 chars if n_chars is not exactly word-aligned.
    """
    out = []
    cur = 0
    i = 0
    while True:
        w = FILLER_WORDS[(i + salt) % len(FILLER_WORDS)]
        add = (1 if out else 0) + len(w)
        if cur + add > n_chars:
            break
        out.append(w)
        cur += add
        i += 1
    text = " ".join(out)
    # Pad up to n_chars with a single-letter token, separated by space.
    while len(text) + 2 <= n_chars:
        text = text + " z"
    return text


SPLITTER_O0 = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=0, add_start_index=True
)
SPLITTER_O75 = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=75, add_start_index=True
)
SPLITTER_O150 = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=150, add_start_index=True
)


def chunks_with_spans(splitter, text):
    docs = splitter.create_documents([text])
    out = []
    for d in docs:
        s = d.metadata.get("start_index", -1)
        out.append((s, s + len(d.page_content), d.page_content))
    return out


def build_rigged_prefix(entity_phrase, next_first_word):
    """Return filler text such that filler + ' ' + entity_phrase ends exactly
    at the first o=0 chunk boundary (chunk 0 ends with entity_phrase).
    Probes use ``next_first_word`` as the first token after the boundary so
    the alignment is identical to the assembled doc.
    """
    for pad_chars in range(220, 310):
        head = filler(pad_chars, salt=7) + " " + entity_phrase
        probe = head + " " + next_first_word + " " + filler(200, salt=11)
        chunks = chunks_with_spans(SPLITTER_O0, probe)
        if len(chunks) < 2:
            continue
        end0 = chunks[0][1]
        target = len(head)
        if end0 == target:
            return head
    raise RuntimeError(
        f"Could not align chunk boundary for entity phrase {entity_phrase!r}."
    )


def assemble_doc(rigged_anchors, normal_anchors, tail_filler_len, tail_salt):
    """Build a doc with optional rigged regions then normal anchor sentences then filler."""
    text = ""
    rigged_spans = []  # list of (entity_phrase, value_phrase, value_start, value_end)
    for entity_phrase, value_phrase, between_separator in rigged_anchors:
        next_first_word = value_phrase.split(" ", 1)[0]
        head = build_rigged_prefix(entity_phrase, next_first_word)
        if text:
            text = text + " " + head
        else:
            text = head
        # Place the value phrase immediately after the chunk boundary.
        # The boundary is at len(text) (a space). The next chunk starts at len(text)+1.
        boundary_pos = len(text)
        text = text + between_separator + value_phrase
        v_start = boundary_pos + len(between_separator)
        v_end = v_start + len(value_phrase)
        rigged_spans.append((entity_phrase, value_phrase, v_start, v_end))
        # Add cushion filler so the next region starts fresh.
        text = text + " " + filler(120, salt=hash(entity_phrase) & 0xFF)
    # Append normal anchor sentences (each unique value phrase placed in the
    # middle of a filler block, well away from chunk boundaries).
    normal_spans = []
    for value_phrase in normal_anchors:
        text = text + " " + filler(80, salt=hash(value_phrase) & 0xFF) + " " + value_phrase
        v_start = len(text) - len(value_phrase)
        v_end = len(text)
        normal_spans.append((value_phrase, v_start, v_end))
        text = text + " " + filler(80, salt=(hash(value_phrase) >> 8) & 0xFF)
    # Pad to desired length.
    if tail_filler_len > 0:
        text = text + " " + filler(tail_filler_len, salt=tail_salt)
    return text, rigged_spans, normal_spans


# -------------------------------------------------------------------
# Anchor catalogue. Each entry: (query, value_phrase).
# The two RIGGED entries come first within their doc, with (entity_phrase, value_phrase).

# Doc 1 (marine biology): 1 rigged + 3 normal
DOC1_RIGGED = [
    (
        "Mesoplodon perrini",  # unique entity
        "grows to 4.1 meters in length",  # value (ground-truth answer span)
        " ",  # separator between entity tail and value start: single space at boundary
    ),
]
DOC1_NORMAL = [
    "the species Loxosceles reclusa lives in temperate shoreline caves",
    "an enzyme labelled Coralase-9 catalyses chitin breakdown at 37 degrees Celsius",
    "the protected nursery zone covers exactly 218 square kilometers of seabed",
]
DOC1_QUERIES = [
    "How long does Mesoplodon perrini grow to be?",  # rigged
    "Where does the species Loxosceles reclusa live?",
    "At what temperature does the enzyme Coralase-9 catalyse chitin breakdown?",
    "What is the area of the protected nursery zone in square kilometers?",
]

# Doc 2 (astronomy): 1 rigged + 2 normal
DOC2_RIGGED = [
    (
        "pulsar HJ-9817",
        "emits radio bursts every 4.31 milliseconds",
        " ",
    ),
]
DOC2_NORMAL = [
    "the rogue exoplanet WD-447b orbits its host every 612 standard days",
    "the Marsden gamma-ray monitor recorded 88 distinct flares during winter 2021",
]
DOC2_QUERIES = [
    "What is the radio burst period of pulsar HJ-9817?",  # rigged
    "How long does the rogue exoplanet WD-447b take to orbit its host?",
    "How many flares did the Marsden gamma-ray monitor record during winter 2021?",
]

# Doc 3 (archaeology): 0 rigged + 3 normal
DOC3_NORMAL = [
    "the Bronze Age site at Tell Qarnayim yielded 1742 fragments of imported tin ingot",
    "the inscription on Stela K from Yaxchilan dates the accession event to 752 CE",
    "the Halaf phase pottery kiln stayed operational for approximately 380 years",
]
DOC3_QUERIES = [
    "How many fragments of imported tin ingot were yielded by Tell Qarnayim?",
    "What year does the inscription on Stela K from Yaxchilan date the accession event to?",
    "How many years did the Halaf phase pottery kiln stay operational?",
]


def main():
    docs = {}
    all_anchors = []

    # ---- DOC 1
    text, rigged, normal = assemble_doc(
        DOC1_RIGGED, DOC1_NORMAL, tail_filler_len=2400, tail_salt=3
    )
    docs["doc_1"] = text
    # Build anchors for doc 1, in DOC1_QUERIES order
    # rigged first (Q index 0), then 3 normals (1,2,3)
    e_phrase_1, v_phrase_1, vs1, ve1 = rigged[0]
    all_anchors.append({
        "query": DOC1_QUERIES[0], "doc_id": "doc_1",
        "start": vs1, "end": ve1, "_rigged": True, "_entity": e_phrase_1,
    })
    for i in range(3):
        vp, vs, ve = normal[i]
        all_anchors.append({
            "query": DOC1_QUERIES[i + 1], "doc_id": "doc_1",
            "start": vs, "end": ve, "_rigged": False, "_entity": "",
        })

    # ---- DOC 2
    text, rigged, normal = assemble_doc(
        DOC2_RIGGED, DOC2_NORMAL, tail_filler_len=2600, tail_salt=13
    )
    docs["doc_2"] = text
    e_phrase_2, v_phrase_2, vs2, ve2 = rigged[0]
    all_anchors.append({
        "query": DOC2_QUERIES[0], "doc_id": "doc_2",
        "start": vs2, "end": ve2, "_rigged": True, "_entity": e_phrase_2,
    })
    for i in range(2):
        vp, vs, ve = normal[i]
        all_anchors.append({
            "query": DOC2_QUERIES[i + 1], "doc_id": "doc_2",
            "start": vs, "end": ve, "_rigged": False, "_entity": "",
        })

    # ---- DOC 3
    text, rigged, normal = assemble_doc(
        [], DOC3_NORMAL, tail_filler_len=2800, tail_salt=23
    )
    docs["doc_3"] = text
    for i in range(3):
        vp, vs, ve = normal[i]
        all_anchors.append({
            "query": DOC3_QUERIES[i], "doc_id": "doc_3",
            "start": vs, "end": ve, "_rigged": False, "_entity": "",
        })

    assert len(all_anchors) == 10, f"Expected 10 anchors, got {len(all_anchors)}."

    # Validate doc lengths and write
    for did, t in docs.items():
        assert 3000 <= len(t) <= 5000, (
            f"{did} length {len(t)} not in [3000, 5000]."
        )
        # Sanity: every anchor's value substring exists at the recorded offset
        for a in all_anchors:
            if a["doc_id"] != did:
                continue
            sub = t[a["start"] : a["end"]]
            # For rigged we know exact phrase via _entity/value placement; for normal we
            # find the phrase via FILLER+space+value, so substring is value_phrase.
            # We don't have value_phrase in the dict; we just spot-check the slice is non-empty.
            assert sub.strip(), (
                f"Empty slice at anchor {a['query']!r} for {did}."
            )
        with open(os.path.join(DOCS_DIR, f"{did}.txt"), "w", encoding="utf-8") as f:
            f.write(t)

    # ---- Validate splitter behaviour and rigging invariants
    rigged_anchors = [a for a in all_anchors if a["_rigged"]]
    assert len(rigged_anchors) == 2, "Exactly 2 rigged anchors expected."

    for did, t in docs.items():
        c0 = chunks_with_spans(SPLITTER_O0, t)
        c150 = chunks_with_spans(SPLITTER_O150, t)
        assert len(c150) > len(c0), (
            f"{did}: o150 chunk count {len(c150)} not > o0 chunk count {len(c0)}."
        )

    for a in rigged_anchors:
        did = a["doc_id"]
        t = docs[did]
        v_start, v_end = a["start"], a["end"]
        entity = a["_entity"]
        c0 = chunks_with_spans(SPLITTER_O0, t)
        c150 = chunks_with_spans(SPLITTER_O150, t)

        # 1) No o=0 chunk contains BOTH entity and value.
        bad_o0 = [
            (s, e, txt) for (s, e, txt) in c0 if entity in txt and t[v_start:v_end] in txt
        ]
        assert not bad_o0, (
            f"Rigging broken for {did} entity {entity!r}: an o=0 chunk has both "
            f"entity and value: {bad_o0}"
        )

        # 2) The o=0 chunk that contains the entity does NOT span overlap GT [v_start, v_end].
        chunk_with_entity = [
            (s, e) for (s, e, txt) in c0 if entity in txt
        ]
        assert chunk_with_entity, (
            f"Rigging broken for {did}: no o=0 chunk contains entity {entity!r}."
        )
        for s, e in chunk_with_entity:
            assert not (s < v_end and v_start < e), (
                f"Rigging broken for {did}: o=0 entity-chunk [{s},{e}) overlaps GT "
                f"[{v_start},{v_end}). Boundary alignment failed."
            )

        # 3) At least one o=150 chunk contains BOTH entity AND value.
        good_o150 = [
            (s, e) for (s, e, txt) in c150 if entity in txt and t[v_start:v_end] in txt
        ]
        assert good_o150, (
            f"Rigging broken for {did}: no o=150 chunk contains both entity "
            f"{entity!r} and value at [{v_start},{v_end})."
        )

    # ---- Strip private fields and write anchors.json
    public_anchors = [
        {"query": a["query"], "doc_id": a["doc_id"], "start": a["start"], "end": a["end"]}
        for a in all_anchors
    ]
    with open(ANCHORS_PATH, "w", encoding="utf-8") as f:
        json.dump(public_anchors, f, indent=2)

    print(
        "fixture OK:",
        {did: len(t) for did, t in docs.items()},
        "anchors:", len(public_anchors),
    )


main()
PYEOF
RUN python3 /app/_build_fixture.py && \
    ls -la /app/docs/ && \
    wc -c /app/docs/*.txt && \
    cat /app/anchors.json | head -40 && \
    rm /app/_build_fixture.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB chunk-overlap retrieval evaluation harness: 3 baked source docs + 10 anchor queries, candidate chunks at overlap 0/75/150 with LangChain and embeds with OpenAI text-embedding-3-small."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#           openai==1.54.5 tiktoken langchain==0.3.27 langchain-core==0.3.78
#           langchain-text-splitters==0.3.11
#   - /home/user/myproject: candidate project directory (empty)
#   - /app/docs/{doc_1,doc_2,doc_3}.txt: 3 source documents (~3000-5000 chars each)
#   - /app/anchors.json: 10 anchor queries with (doc_id, start, end) ground-truth spans
#   - rigging: 2 of 10 anchors fall at o=0 chunk boundaries (entity tail vs value head)
