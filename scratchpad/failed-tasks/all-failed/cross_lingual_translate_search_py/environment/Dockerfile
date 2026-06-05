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
        openai==1.54.5 \
        httpx==0.27.2 \
        tiktoken && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Bake the English corpus + OpenAI embeddings into a JSON snapshot at build time.
# The actual LanceDB table is materialized at container start so we can use the
# per-trial ZEALT_RUN_ID suffix (required by the parallel-run-safety rule).
RUN mkdir -p /opt/seed && \
    cat >/opt/seed/build_seed.py <<'PYEOF'
import json
import os
import time

from openai import OpenAI

DOCS = [
    # ---------------- machine_learning (anchor for Chinese query) ----------------
    ("machine_learning", "Machine learning is the study of computer algorithms that improve automatically through experience and the use of data. Supervised learning algorithms build a model from labeled training data to make predictions on unseen inputs."),
    ("machine_learning", "Common machine learning algorithms include linear regression, logistic regression, decision trees, random forests, gradient boosted trees, support vector machines, k-nearest neighbors, and deep neural networks for classification and regression tasks."),
    ("machine_learning", "Training a machine learning model usually involves splitting data into training and validation sets, optimizing a loss function via stochastic gradient descent, and evaluating predictive accuracy with metrics such as precision, recall, F1, and AUC."),
    ("machine_learning", "Unsupervised machine learning algorithms such as k-means clustering, principal component analysis, and self-organizing maps discover structure in unlabeled data and are widely used for customer segmentation and anomaly detection."),

    # ---------------- sushi_recipes (anchor for Japanese query) ----------------
    ("sushi_recipes", "Sushi is a traditional Japanese dish made with vinegared short-grain rice combined with fresh seafood, vegetables, and toasted nori seaweed. A basic sushi recipe begins with rinsing the rice until the water runs clear and seasoning it with rice vinegar, sugar, and salt."),
    ("sushi_recipes", "A salmon nigiri sushi recipe requires sashimi-grade salmon sliced thinly across the grain and hand-pressed onto a small mound of seasoned sushi rice, brushed with a touch of soy sauce and served with pickled ginger and wasabi."),
    ("sushi_recipes", "To prepare a California roll sushi at home, spread sushi rice over a sheet of nori on a bamboo mat, lay strips of crab stick, avocado, and cucumber across the rice, then roll tightly and slice into bite-sized pieces."),
    ("sushi_recipes", "Tamago sushi is a sweet Japanese rolled omelet recipe where eggs, dashi stock, mirin, sugar, and soy sauce are folded layer by layer in a rectangular pan, sliced, and placed atop seasoned sushi rice with a thin band of nori."),

    # ---------------- sahara_desert (anchor for Arabic query) ----------------
    ("sahara_desert", "The Sahara is the largest hot desert in the world, covering roughly 9.2 million square kilometers across North Africa, including parts of Algeria, Chad, Egypt, Libya, Mali, Mauritania, Morocco, Niger, Sudan, and Tunisia."),
    ("sahara_desert", "The Sahara desert features vast sand seas known as ergs, rocky plateaus called hamadas, gravel plains called regs, and dramatic dune fields such as the Grand Erg Oriental and the Erg Chebbi, with summer temperatures often exceeding 45 degrees Celsius."),
    ("sahara_desert", "Bordered by the Atlas Mountains to the northwest, the Red Sea to the east, the Mediterranean Sea to the north, and the semi-arid Sahel to the south, the Sahara desert receives less than 100 millimeters of rainfall per year on average."),
    ("sahara_desert", "The nomadic Tuareg and Bedouin peoples have crossed the Sahara desert for centuries on caravan routes carrying salt, gold, and textiles between sub-Saharan Africa and the Mediterranean coast through oases such as Siwa and Ghadames."),

    # ---------------- distractor: quantum_computing ----------------
    ("quantum_computing", "Quantum computing exploits quantum mechanical phenomena such as superposition and entanglement to perform calculations that are intractable for classical computers, using qubits implemented with trapped ions, superconducting circuits, or photonic systems."),
    ("quantum_computing", "Shor's algorithm demonstrates how a sufficiently large quantum computer could factor large integers in polynomial time, threatening modern public-key cryptography based on RSA and forcing migration to post-quantum cryptographic schemes."),
    ("quantum_computing", "Quantum error correction codes such as the surface code use many physical qubits to encode a single logical qubit, suppressing decoherence and gate errors that otherwise limit the depth of useful quantum circuits."),
    ("quantum_computing", "Variational quantum eigensolvers combine a parameterized quantum circuit with a classical optimizer to estimate ground-state energies of molecules and materials on noisy intermediate-scale quantum hardware."),

    # ---------------- distractor: renaissance_art ----------------
    ("renaissance_art", "Renaissance art emerged in Florence during the fourteenth century, blending classical Greek and Roman influences with new techniques such as linear perspective, chiaroscuro shading, and detailed anatomical study of the human form."),
    ("renaissance_art", "Leonardo da Vinci's Mona Lisa exemplifies High Renaissance painting through its sfumato modeling, atmospheric landscape background, and psychological depth that has fascinated viewers for over five centuries."),
    ("renaissance_art", "Michelangelo Buonarroti sculpted David from a single block of Carrara marble between 1501 and 1504, depicting the biblical hero in heroic nude contrapposto as a symbol of Florentine civic virtue."),
    ("renaissance_art", "Raphael's School of Athens fresco in the Vatican Stanze gathers ancient philosophers including Plato and Aristotle in an idealized classical architecture, representing the Renaissance synthesis of philosophy, science, and art."),

    # ---------------- distractor: roman_architecture ----------------
    ("roman_architecture", "Roman architecture is famous for its monumental civic structures including the Colosseum, the Pantheon, and vast aqueducts that delivered water across hundreds of kilometers using gentle gradients and arched stone channels."),
    ("roman_architecture", "The Romans pioneered the use of opus caementicium, a volcanic-ash concrete that allowed them to construct enormous unreinforced domes such as the Pantheon's coffered rotunda, which remained the world's largest dome for over a millennium."),
    ("roman_architecture", "Roman basilicas were rectangular civic halls with apses at one end, central naves flanked by aisles, and clerestory windows, a layout later adapted as the floor plan for early Christian churches throughout the Mediterranean world."),
    ("roman_architecture", "Triumphal arches such as the Arch of Titus and the Arch of Constantine in Rome commemorated military victories with relief sculpture, inscriptions, and a single or triple opening framed by engaged columns."),

    # ---------------- distractor: beekeeping ----------------
    ("beekeeping", "Beekeeping, also known as apiculture, is the maintenance of honey bee colonies in wooden hives for honey, beeswax, propolis, royal jelly, and crop pollination services on commercial farms and backyard apiaries."),
    ("beekeeping", "A standard Langstroth beehive consists of stacked boxes containing removable frames of beeswax foundation on which worker bees draw out hexagonal cells for brood rearing and honey storage."),
    ("beekeeping", "Beekeepers manage colonies by inspecting brood patterns, monitoring varroa mite infestations, replacing aging queens, and supplementing winter food stores with sugar syrup or fondant to ensure colony survival."),
    ("beekeeping", "Honey is harvested by removing capped honeycomb frames, uncapping the wax seals with a hot knife, and spinning the frames in a centrifugal extractor before straining and bottling the liquid honey."),

    # ---------------- distractor: mountain_climbing ----------------
    ("mountain_climbing", "Mountain climbing, or mountaineering, involves ascending peaks using a combination of hiking, rock climbing, ice climbing, and glacier travel, with safety techniques such as roping up, crevasse rescue, and avalanche awareness."),
    ("mountain_climbing", "High-altitude climbers acclimatize gradually to reduce the risk of acute mountain sickness, high-altitude pulmonary edema, and cerebral edema, often spending nights at intermediate camps before summit pushes above 7000 meters."),
    ("mountain_climbing", "Modern alpine climbing emphasizes fast and light ascents in a single push from base camp, contrasting with the heavy expedition style that uses fixed ropes, supplemental oxygen, and multiple high camps."),
    ("mountain_climbing", "The Seven Summits comprise the highest peak on each continent, including Mount Everest in Asia, Aconcagua in South America, Denali in North America, and Mount Vinson in Antarctica."),

    # ---------------- distractor: symphony_orchestra ----------------
    ("symphony_orchestra", "A modern symphony orchestra typically includes about a hundred musicians divided into strings, woodwinds, brass, and percussion sections, led by a conductor who shapes tempo, dynamics, and overall interpretation."),
    ("symphony_orchestra", "Beethoven's Ninth Symphony, premiered in 1824, introduced vocal soloists and a chorus into the symphonic form, setting Friedrich Schiller's Ode to Joy in its triumphant final movement."),
    ("symphony_orchestra", "The string section of a symphony orchestra is arranged in five subsections: first violins, second violins, violas, cellos, and double basses, each playing distinct lines of harmony and counterpoint."),
    ("symphony_orchestra", "Symphonic repertoire ranges from the classical period of Haydn and Mozart through the romantic giants Mahler and Bruckner to twentieth-century composers such as Shostakovich, Stravinsky, and Sibelius."),

    # ---------------- distractor: coral_reef_ecology ----------------
    ("coral_reef_ecology", "Coral reefs are biodiverse marine ecosystems built by colonies of tiny animals called polyps that secrete calcium carbonate skeletons, hosting symbiotic zooxanthellae algae that provide energy through photosynthesis."),
    ("coral_reef_ecology", "The Great Barrier Reef off the coast of Queensland, Australia, is the largest coral reef system on Earth, stretching more than 2300 kilometers and supporting over 1500 species of fish."),
    ("coral_reef_ecology", "Rising sea temperatures cause coral bleaching as polyps expel their zooxanthellae, leaving white skeletons that can recover if conditions return to normal or die if heat stress is prolonged."),
    ("coral_reef_ecology", "Coral reef conservation programs include marine protected areas, coral nurseries that out-plant heat-resistant strains, and reduction of land-based runoff that brings nutrients and sediment onto fragile reef flats."),
]

assert len(DOCS) == 40, f"Expected 40 docs, got {len(DOCS)}"

api_key = os.environ.get("OPENAI_API_KEY")
assert api_key, "OPENAI_API_KEY must be passed as --build-arg during docker build."

client = OpenAI(api_key=api_key)
model = "text-embedding-3-small"

rows = []
for idx, (topic, content) in enumerate(DOCS):
    last_err = None
    for attempt in range(5):
        try:
            resp = client.embeddings.create(model=model, input=content)
            vec = resp.data[0].embedding
            assert len(vec) == 1536, f"Unexpected embedding dim {len(vec)}"
            rows.append({"id": idx, "topic": topic, "content": content, "embedding": vec})
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to embed doc {idx}: {last_err}")

out_path = "/opt/seed/corpus_seed.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(rows, f)
print(f"WROTE {len(rows)} rows to {out_path}")
PYEOF

RUN OPENAI_API_KEY="${OPENAI_API_KEY}" python3 /opt/seed/build_seed.py

# Entrypoint script: materialize the per-trial LanceDB table from the seed JSON.
RUN cat >/usr/local/bin/init-corpus.sh <<'SHEOF'
#!/usr/bin/env bash
set -euo pipefail
python3 /opt/seed/init_corpus.py
exec "$@"
SHEOF
RUN chmod +x /usr/local/bin/init-corpus.sh && \
    cat >/opt/seed/init_corpus.py <<'PYEOF'
import json
import os
import sys

import lancedb
import pyarrow as pa

URI = os.environ.get("LANCEDB_URI", "/home/user/lancedb_data")
PREFIX = os.environ.get("LANCEDB_TABLE_PREFIX", "corpus_")
RUN_ID = os.environ.get("ZEALT_RUN_ID", "default")
SEED = "/opt/seed/corpus_seed.json"

os.makedirs(URI, exist_ok=True)
with open(SEED, "r", encoding="utf-8") as f:
    rows = json.load(f)

table_name = f"{PREFIX}{RUN_ID}"
db = lancedb.connect(URI)
existing = set(db.table_names())
if table_name in existing:
    print(f"Table {table_name} already present; skipping seed.")
    sys.exit(0)

dim = len(rows[0]["embedding"])
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("topic", pa.string()),
    pa.field("content", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), dim)),
])
arrow_rows = [
    {"id": r["id"], "topic": r["topic"], "content": r["content"], "embedding": r["embedding"]}
    for r in rows
]
tbl = db.create_table(table_name, data=arrow_rows, schema=schema)
print(f"Seeded {tbl.count_rows()} rows into LanceDB table {table_name} at {URI}.")
sys.stdout.flush()
os._exit(0)
PYEOF

ENV LANCEDB_URI=/home/user/lancedb_data
ENV LANCEDB_TABLE_PREFIX=corpus_
ENV OPENAI_CHAT_MODEL=gpt-4o-mini
ENV OPENAI_EMBED_MODEL=text-embedding-3-small

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB cross-lingual translate-then-search task: 40-row English corpus pre-embedded with OpenAI text-embedding-3-small, entrypoint seeds a per-trial table."

ENTRYPOINT ["/usr/local/bin/init-corpus.sh"]
CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 openai==1.54.5 httpx==0.27.2 tiktoken
#   - /home/user/myproject: candidate workspace
#   - /opt/seed/corpus_seed.json: 40 pre-embedded English docs (text-embedding-3-small) baked at build
#   - /usr/local/bin/init-corpus.sh + /opt/seed/init_corpus.py: entrypoint materializes
#     LanceDB table ${LANCEDB_TABLE_PREFIX}${ZEALT_RUN_ID} under ${LANCEDB_URI}
