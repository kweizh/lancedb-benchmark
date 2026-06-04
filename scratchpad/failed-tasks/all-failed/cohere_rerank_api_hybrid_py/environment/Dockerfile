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
        httpx==0.27.2 \
        cohere \
        tantivy==0.22.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/cohere_rerank

# Bake the seed script and entrypoint into the image.
RUN cat >/home/user/cohere_rerank/_seed.py <<'PYEOF'
"""Seed the LanceDB `docs` table with 200 deterministic short documents,
embed each `content` with real OpenAI `text-embedding-3-small`, and create
a BM25 FTS index. Runs once at container startup before the task begins."""
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa
from openai import OpenAI

DB_DIR = "/home/user/cohere_rerank/lancedb"
TABLE_NAME = "docs"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
SEEDED_MARKER = "/home/user/cohere_rerank/lancedb/.seeded"


# Rigged English ground-truth document. Must match tests/test_final_state.py exactly.
RIGGED_DOC = {
    "id": "install-linux-guide",
    "language": "en",
    "content": (
        "A complete step-by-step guide to install our package on Linux: "
        "use apt-get on Ubuntu and Debian, or dnf on Fedora. "
        "This walks through the full Linux installation workflow."
    ),
}

# A handful of English distractor docs that share some keywords but are NOT the rigged doc.
EN_DISTRACTORS = [
    {
        "id": "install-mac-brew",
        "content": (
            "How to install our application on macOS with Homebrew. "
            "Run brew install ourapp on your Mac terminal and you are done."
        ),
    },
    {
        "id": "install-windows-msi",
        "content": (
            "Windows installation guide for our application using the MSI installer package. "
            "Double-click the installer and follow the wizard."
        ),
    },
    {
        "id": "linux-news-kernel",
        "content": (
            "Linux kernel 6.5 release notes: scheduler improvements and new networking features. "
            "Read the changelog for security patches."
        ),
    },
    {
        "id": "linux-jobs-posting",
        "content": (
            "We are hiring a Linux systems administrator. Remote welcome. "
            "Experience with Ubuntu server fleets is a plus."
        ),
    },
    {
        "id": "linux-gaming-2024",
        "content": (
            "Top Linux games of 2024: gaming on Linux has never been better thanks to Proton. "
            "Steam Deck owners rejoice."
        ),
    },
]


# Twenty short English filler topics. We repeat each four times with a number to reach 74 fillers.
EN_FILLER_TEMPLATES = [
    "A classic Italian tomato pasta recipe with fresh basil and garlic, perfect for dinner. Variant {n}.",
    "Quarterly earnings report highlights record revenue growth in the cloud division. Variant {n}.",
    "Travel guide to Kyoto: cherry blossoms, temples, and tea ceremonies in spring. Variant {n}.",
    "Beginners guide to home gardening: how to grow tomatoes and herbs on a balcony. Variant {n}.",
    "Stock market commentary: tech sector rebounds after a long correction. Variant {n}.",
    "Tips for training a puppy to sit, stay, and walk on a leash without pulling. Variant {n}.",
    "Photography composition basics: rule of thirds, leading lines, and negative space. Variant {n}.",
    "Beginner yoga routine for office workers with tight hips and stiff shoulders. Variant {n}.",
    "Watercolor painting techniques for landscapes: wet-on-wet washes and dry brush details. Variant {n}.",
    "How to brew a perfect pour-over coffee using a Hario V60 and a goose-neck kettle. Variant {n}.",
    "Marathon training plan covering long runs, tempo workouts, and rest week recovery. Variant {n}.",
    "Astronomy basics: identifying planets and constellations in the northern sky. Variant {n}.",
    "Cooking class focused on classic French sauces: bechamel, veloute, and hollandaise. Variant {n}.",
    "Personal finance basics: how to build an emergency fund and pay off credit card debt. Variant {n}.",
    "Beginners electric guitar lesson: power chords, palm muting, and basic blues licks. Variant {n}.",
    "Sourdough bread starter maintenance: feeding schedule, hydration, and discard recipes. Variant {n}.",
    "Hiking the Pacific Crest Trail: gear list, water sources, and resupply strategy. Variant {n}.",
    "Birdwatching journal entry: a morning of warblers in the spring migration. Variant {n}.",
    "Pottery wheel throwing tips: centering clay, opening, and pulling even walls. Variant {n}.",
    "Cycling commute tips: pannier setup, rain gear, and visible night lighting. Variant {n}.",
]

# Spanish filler templates (10 topics, repeated to reach 60).
ES_FILLER_TEMPLATES = [
    "Receta tradicional de paella valenciana con pollo, conejo y judias verdes. Variante {n}.",
    "Guia de viaje por Barcelona: la Sagrada Familia, las Ramblas y el barrio gotico. Variante {n}.",
    "Como cultivar tomates en macetas pequenas en el balcon de un apartamento. Variante {n}.",
    "Consejos para entrenar a un cachorro a sentarse, quedarse y caminar con correa. Variante {n}.",
    "Curso de fotografia: regla de los tercios, lineas de composicion y espacio negativo. Variante {n}.",
    "Rutina de yoga matutina para oficinistas con dolor de espalda y cuello. Variante {n}.",
    "Como preparar un cafe de filtro perfecto con una cafetera V60 y agua a 92 grados. Variante {n}.",
    "Reporte trimestral de la bolsa: el sector tecnologico se recupera tras la correccion. Variante {n}.",
    "Plan de entrenamiento para correr una media maraton en doce semanas. Variante {n}.",
    "Astronomia basica: como identificar planetas y constelaciones en el hemisferio norte. Variante {n}.",
]

# French filler templates (10 topics, repeated to reach 60).
FR_FILLER_TEMPLATES = [
    "Recette traditionnelle de boeuf bourguignon avec vin rouge, lardons et champignons. Variante {n}.",
    "Guide de voyage a Paris: la Tour Eiffel, le Louvre et le quartier de Montmartre. Variante {n}.",
    "Comment cultiver des tomates en pots sur un petit balcon parisien. Variante {n}.",
    "Conseils pour dresser un chiot a s'asseoir, rester et marcher en laisse. Variante {n}.",
    "Cours de photographie: regle des tiers, lignes directrices et espace negatif. Variante {n}.",
    "Routine de yoga matinale pour les employes de bureau avec mal de dos. Variante {n}.",
    "Comment preparer un cafe filtre parfait avec une V60 et de l'eau a 92 degres. Variante {n}.",
    "Rapport trimestriel boursier: le secteur technologique rebondit apres la correction. Variante {n}.",
    "Programme d'entrainement pour courir un semi-marathon en douze semaines. Variante {n}.",
    "Astronomie de base: identifier les planetes et les constellations dans l'hemisphere nord. Variante {n}.",
]


def build_rows():
    rows = []

    # 1 rigged + 5 distractors + 74 fillers = 80 English rows.
    rows.append({"id": RIGGED_DOC["id"], "content": RIGGED_DOC["content"], "language": "en"})
    for d in EN_DISTRACTORS:
        rows.append({"id": d["id"], "content": d["content"], "language": "en"})

    en_filler_count = 0
    for n in range(1, 5):  # 4 variants
        for i, tpl in enumerate(EN_FILLER_TEMPLATES):
            if en_filler_count >= 74:
                break
            rows.append({
                "id": f"en-filler-{i:02d}-{n}",
                "content": tpl.format(n=n),
                "language": "en",
            })
            en_filler_count += 1
        if en_filler_count >= 74:
            break
    assert en_filler_count == 74, en_filler_count

    # 60 Spanish rows: 10 templates x 6 variants
    es_count = 0
    for n in range(1, 7):
        for i, tpl in enumerate(ES_FILLER_TEMPLATES):
            if es_count >= 60:
                break
            rows.append({
                "id": f"es-filler-{i:02d}-{n}",
                "content": tpl.format(n=n),
                "language": "es",
            })
            es_count += 1
        if es_count >= 60:
            break
    assert es_count == 60, es_count

    # 60 French rows: 10 templates x 6 variants
    fr_count = 0
    for n in range(1, 7):
        for i, tpl in enumerate(FR_FILLER_TEMPLATES):
            if fr_count >= 60:
                break
            rows.append({
                "id": f"fr-filler-{i:02d}-{n}",
                "content": tpl.format(n=n),
                "language": "fr",
            })
            fr_count += 1
        if fr_count >= 60:
            break
    assert fr_count == 60, fr_count

    assert len(rows) == 200, len(rows)
    return rows


def embed_batch(client, texts, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        for item in resp.data:
            out.append(item.embedding)
    return out


def main():
    if os.path.exists(SEEDED_MARKER):
        print(f"[seed] Already seeded ({SEEDED_MARKER} exists), skipping.", flush=True)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("[seed] ERROR: OPENAI_API_KEY is required to seed embeddings.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(DB_DIR, exist_ok=True)

    rows = build_rows()
    texts = [r["content"] for r in rows]

    client = OpenAI()
    print(f"[seed] Embedding {len(texts)} documents with {EMBED_MODEL}...", flush=True)
    embeddings = embed_batch(client, texts)
    assert len(embeddings) == len(rows), (len(embeddings), len(rows))
    for r, vec in zip(rows, embeddings):
        assert len(vec) == EMBED_DIM, (r["id"], len(vec))
        r["embedding"] = vec

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), EMBED_DIM)),
        pa.field("language", pa.string()),
    ])

    arrays = {
        "id": [r["id"] for r in rows],
        "content": [r["content"] for r in rows],
        "embedding": [np.asarray(r["embedding"], dtype=np.float32) for r in rows],
        "language": [r["language"] for r in rows],
    }
    table_arrow = pa.table(arrays, schema=schema)

    db = lancedb.connect(DB_DIR)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    tbl = db.create_table(TABLE_NAME, data=table_arrow, mode="create")

    print("[seed] Creating native (no-Tantivy) FTS index on 'content'...", flush=True)
    tbl.create_fts_index("content", use_tantivy=False, replace=True)

    n = tbl.count_rows()
    print(f"[seed] Table 'docs' now has {n} rows.", flush=True)

    with open(SEEDED_MARKER, "w") as f:
        f.write("ok\n")
    print(f"[seed] Wrote marker {SEEDED_MARKER}.", flush=True)
    # Avoid lancedb 0.25.3 SIGABRT at interpreter teardown after heavy writes.
    os._exit(0)


main()
PYEOF

RUN cat >/home/user/cohere_rerank/entrypoint.sh <<'SHEOF'
#!/usr/bin/env bash
set -u
MARKER=/home/user/cohere_rerank/lancedb/.seeded
if [ ! -f "$MARKER" ]; then
    python3 /home/user/cohere_rerank/_seed.py || {
        echo "[entrypoint] Seed failed; exiting." >&2
        exit 1
    }
fi
trap : TERM INT
sleep infinity & wait
SHEOF

RUN chmod +x /home/user/cohere_rerank/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB Cohere rerank hybrid search task: 200 docs (80 en / 60 es / 60 fr) with real OpenAI 1536-d embeddings, seeded at container start; candidate builds a Cohere-reranked hybrid search CLI."

CMD ["/home/user/cohere_rerank/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 + pyarrow==17.0.0 + numpy==1.26.4 + pandas==2.2.3 + openai==1.54.5 + cohere + tantivy==0.22.0
#   - /home/user/cohere_rerank/_seed.py: seeds the 200-row `docs` table at first container start using OPENAI_API_KEY
#   - /home/user/cohere_rerank/entrypoint.sh: runs the seed script then sleeps; idempotent via .seeded marker
