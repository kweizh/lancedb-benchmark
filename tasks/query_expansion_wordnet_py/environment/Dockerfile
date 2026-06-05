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
        "nltk>=3.8" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Pre-download the WordNet + OMW-1.4 corpora at build time so the task runs
# entirely offline. WordNet is a static lexical database (not a model).
ENV NLTK_DATA=/usr/share/nltk_data
RUN mkdir -p /usr/share/nltk_data && \
    python3 -c "import nltk; nltk.download('wordnet', download_dir='/usr/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/share/nltk_data')" && \
    python3 -c "from nltk.corpus import wordnet as wn; assert len(wn.synsets('car')) > 0, 'wordnet missing'; print('wordnet ok:', len(wn.synsets('car')))"

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject
RUN mkdir -p /app/lancedb_data

ENV LANCEDB_URI=/app/lancedb_data
ENV LANCEDB_TABLE=docs

# Seed the 100-row English corpus at build time. The fixture has:
#   * 6 docs containing the word "car" but never "automobile"/"vehicle"
#   * 8 docs containing "automobile" but never "car" (synonym-only targets)
#   * 86 unrelated docs containing none of {car, automobile, vehicle, auto,
#     machine, motorcar, railcar, ...}
# The candidate must build a native FTS index on `content` themselves; the
# seed script only creates the table.
RUN cat >/app/seed.py <<'PYEOF'
import os
import re
import sys

import pyarrow as pa
import lancedb

LANCEDB_DIR = os.environ.get("LANCEDB_URI", "/app/lancedb_data")
TABLE_NAME = os.environ.get("LANCEDB_TABLE", "docs")

# Words that must NOT appear (as standalone tokens) in unrelated docs, so the
# verifier's `automobile`/`vehicle` set and the candidate's WordNet expansion
# (`auto`, `automobile`, `machine`, `motorcar`, ...) are uncontaminated.
FORBIDDEN_IN_UNRELATED = {
    "car", "cars",
    "automobile", "automobiles",
    "vehicle", "vehicles",
    "auto", "autos",
    "machine", "machines", "machinery",
    "motorcar", "motorcars",
    "railcar", "railcars",
    "gondola", "gondolas",
}

car_docs = [
    "She drove her car to the airport very early on Monday morning.",
    "My friend's car broke down on the busy highway last weekend evening.",
    "He parked his car in front of the small bakery near the corner.",
    "The blue car raced past the finish line in absolute record time today.",
    "Repairing a car engine often requires expensive replacement parts and labour hours.",
    "Her first car was a small hatchback she loved dearly for years.",
]

automobile_docs = [
    "The automobile industry has rapidly transformed thanks to electric powertrains worldwide.",
    "Restoring a vintage automobile takes patience, specialized tools and steady hands.",
    "An automobile assembly line is a marvel of modern industrial engineering today.",
    "Insurance for a luxury automobile can be quite expensive every single calendar year.",
    "The local museum proudly displays an antique automobile from the roaring twenties era.",
    "Modern automobile safety standards include airbags, crumple zones and antilock braking.",
    "Buying a new automobile is a significant financial commitment for most growing families.",
    "An automobile race across the dusty desert attracted thousands of cheering spectators.",
]

unrelated_docs = [
    "Birds migrate south every winter to find warmer climates and abundant food sources.",
    "The library opens at nine in the morning on weekdays and stays open until eight.",
    "Baking sourdough bread requires a patient hand and a well-fed starter culture daily.",
    "Saturn has fascinated astronomers for centuries with its bright icy ring system.",
    "Ancient Rome built impressive aqueducts to deliver fresh water across the wide empire.",
    "The pianist performed a beautiful concerto with the local orchestra last spring evening.",
    "Penguins waddle gracefully across the ice floes despite their short stubby legs.",
    "A sudden thunderstorm rolled across the prairie and soaked the ripening wheat fields.",
    "Shakespeare wrote tragedies and comedies that still resonate with modern audiences today.",
    "Quantum physicists puzzle over the strange behaviour of entangled particles every day.",
    "Hiking the alpine ridge offered breathtaking views of glacial valleys far below.",
    "Knitting a thick winter sweater takes many evenings of careful concentrated patient work.",
    "Bees diligently pollinate wildflowers across the sun-drenched meadow each summer afternoon.",
    "The chef garnished the soup with fresh herbs and a swirl of olive oil.",
    "Lighthouses once guided sailors safely past treacherous rocky coastlines on stormy nights.",
    "Children laughed gleefully while sliding down the snowy hillside on plastic sleds.",
    "Coral reefs teem with colourful fish that flit among the branching pink corals.",
    "Translating ancient Greek philosophy requires deep knowledge of classical languages and history.",
    "The artist sketched the bustling marketplace with quick deft strokes of dark charcoal.",
    "Marathon runners train for months to build endurance and mental toughness gradually.",
    "Honeybees produce wax combs that hold both larvae and sweet golden harvest honey.",
    "The novelist crafted a delicate plot full of subtle twists and rich symbolism.",
    "Geologists drilled deep core samples to study layers of ancient sediment carefully.",
    "A small fishing boat bobbed gently in the harbour as the orange sun set.",
    "Volunteers planted native shrubs along the riverbank to prevent rapid soil erosion.",
    "The blacksmith hammered glowing iron into a graceful curved horseshoe shape patiently.",
    "Olympic divers leap from the platform and twist gracefully into the still water.",
    "Mountain goats nimbly navigate the sheer cliffs in search of tender alpine mosses.",
    "Bookbinders carefully stitch each signature before gluing on the supple leather cover.",
    "The teacher led her students through a lively discussion of medieval literature today.",
    "Salt marshes filter pollutants and provide habitat for migratory wading birds annually.",
    "Old wooden barns dotted the rolling green countryside under a brilliant clear sky.",
    "The pastry chef piped delicate rosettes of buttercream onto the tall tiered cake.",
    "Acoustic guitar strings vibrated softly as she fingerpicked an old folk tune tonight.",
    "Beekeepers carefully inspected each hive frame for signs of unwelcome mite infestation.",
    "Photographers captured stunning aurora displays high above the frozen arctic circle skies.",
    "The cobbler repaired worn boots with sturdy stitches and fresh tough leather soles.",
    "Sloths move slowly through the rainforest canopy, eating leaves and napping often.",
    "Knit scarves and woolen mittens hung from pegs in the cozy little entryway.",
    "Historians debated the causes of the bronze age collapse for decades after publication.",
    "Fresh basil leaves bring a fragrant lift to a simple summer tomato salad bowl.",
    "The astronomer focused the telescope on a distant nebula glowing softly red tonight.",
    "Squirrels frantically gathered acorns and walnuts before the first heavy fluffy snowfall.",
    "Watercolour painters love capturing the soft gradients of an autumn sunset over hills.",
    "The blacksmith's apprentice fanned the forge until the iron glowed bright orange-white.",
    "Trail runners weaved between boulders along the steep alpine ridge near sunrise daily.",
    "Cookbooks lined the kitchen shelf, dog-eared and stained with cheerful family use.",
    "Coastal sand dunes shift slowly with the prevailing winds and seasonal tides nightly.",
    "The cellist coaxed a haunting melody from her instrument under bright warm stage lights.",
    "A flock of starlings wheeled and dove in a mesmerizing aerial dance above.",
    "Apple orchards burst into pink and white blossom every single spring without fail.",
    "Researchers carefully tagged migrating monarch butterflies before the long flight south.",
    "Bagpipers led the wedding procession across the wide green sunlit hilltop slowly.",
    "Fresh mountain streams tumble noisily over moss-covered boulders down the steep valley.",
    "The bakery sold out of crusty sourdough loaves before noon on a busy Saturday.",
    "Glassblowers spun molten glass into delicate vases shaped like blooming spring tulips.",
    "Wildflower meadows hum with the gentle activity of bees and small fluttering butterflies.",
    "The history teacher displayed ancient pottery shards on a long polished oak table.",
    "Sailors hoisted the colourful spinnaker and surged ahead in the gusty afternoon breeze.",
    "Beekeepers harvest honey in late summer once the comb is fully sealed and capped.",
    "Painters mixed cobalt blue with titanium white to capture the cloudy evening sky.",
    "A herd of bison thundered across the open prairie at golden hour sunset.",
    "Stargazers gathered on the meadow to watch the annual perseid meteor shower together.",
    "Fresh baguettes filled the small bakery with an irresistible warm welcoming scent.",
    "The poet recited her latest verses at the small candlelit cafe downstairs near midnight.",
    "Wool sweaters dried slowly on a wooden rack near the crackling stone fireplace.",
    "Long shadows stretched across the snowy field as twilight slowly settled in quietly.",
    "Pottery wheels spun while students shaped pale clay into tall slender elegant vases.",
    "Dolphins leaped joyfully alongside the tour boat as it crossed the calm wide bay.",
    "Beach grasses bent gently in the salty breeze along the long dune crest line.",
    "The choir rehearsed a soaring anthem in the echoing stone cathedral on Tuesday.",
    "Beekeepers checked their hives for the first early signs of spring growth and bloom.",
    "Foxes prowled the hedgerow at dusk in search of small unwary scampering rodents.",
    "Lavender bloomed in tidy purple rows across the gentle hillside farm every summer.",
    "Sourdough starters bubble happily on the counter when kept warm and regularly fed.",
    "Composers sketched musical phrases on staff paper beside a steaming mug of green tea.",
    "Glaciers carved deep U-shaped valleys across the northern hemisphere a long time ago.",
    "Birdwatchers waited patiently in the marsh for the elusive bittern to finally appear.",
    "Wooden footbridges arched gracefully across the quiet stream in the old hillside garden.",
    "Tea kettles hissed cheerfully on the stove as guests arrived for a hearty breakfast.",
    "The watchmaker patiently assembled tiny gears under a bright glowing magnifying lamp.",
    "Climbers carefully rappelled down the granite face under a cloudless azure morning sky.",
    "Fresh tomatoes ripened on the vine in the kitchen garden beside the wooden porch.",
    "Snowy owls hunted silently across the wide tundra in the fading pale polar twilight.",
    "Lullabies drifted softly from the nursery as evening settled over the quiet old house.",
    "The cobbler hung freshly polished boots in a tidy row beside the front door.",
]

assert len(car_docs) == 6, len(car_docs)
assert len(automobile_docs) == 8, len(automobile_docs)
assert len(unrelated_docs) == 86, len(unrelated_docs)


def _tokens(s):
    return set(re.findall(r"\b\w+\b", s.lower()))


# Fixture invariants (fail fast at build time if a hand-written sentence is contaminated).
for i, d in enumerate(car_docs):
    toks = _tokens(d)
    assert "car" in toks, f"car_docs[{i}] missing 'car' token: {d!r}"
    assert "automobile" not in toks, f"car_docs[{i}] contains 'automobile': {d!r}"
    assert "vehicle" not in toks, f"car_docs[{i}] contains 'vehicle': {d!r}"

for i, d in enumerate(automobile_docs):
    toks = _tokens(d)
    assert "automobile" in toks, f"automobile_docs[{i}] missing 'automobile' token: {d!r}"
    assert "car" not in toks, f"automobile_docs[{i}] contains 'car': {d!r}"

for i, d in enumerate(unrelated_docs):
    toks = _tokens(d)
    leak = toks & FORBIDDEN_IN_UNRELATED
    assert not leak, f"unrelated_docs[{i}] leaks forbidden tokens {leak}: {d!r}"

# Layout: ids 0..5 car docs, 6..13 automobile docs, 14..99 unrelated docs.
rows = []
rows.extend({"id": i, "content": c} for i, c in enumerate(car_docs))
rows.extend({"id": 6 + i, "content": c} for i, c in enumerate(automobile_docs))
rows.extend({"id": 14 + i, "content": c} for i, c in enumerate(unrelated_docs))
assert len(rows) == 100, f"expected 100 rows, got {len(rows)}"
assert {r["id"] for r in rows} == set(range(100)), "ids must cover 0..99"

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("content", pa.string()),
])

db = lancedb.connect(LANCEDB_DIR)
if TABLE_NAME in db.table_names():
    db.drop_table(TABLE_NAME)
arrow_tbl = pa.Table.from_pylist(rows, schema=schema)
tbl = db.create_table(TABLE_NAME, data=arrow_tbl)
assert tbl.count_rows() == 100, f"expected 100 rows after create, got {tbl.count_rows()}"

# Synonym-only invariant: ≥3 docs that contain 'automobile' or 'vehicle' but no 'car'.
synonym_only = 0
for r in rows:
    t = _tokens(r["content"])
    if ("automobile" in t or "vehicle" in t) and "car" not in t:
        synonym_only += 1
assert synonym_only >= 3, f"fixture invariant: expected ≥3 synonym-only docs, found {synonym_only}"

print(f"seed: wrote {tbl.count_rows()} rows to {LANCEDB_DIR}/{TABLE_NAME}; synonym_only={synonym_only}")

# Avoid the lancedb 0.25.3 interpreter-teardown SIGABRT observed with write-heavy seed scripts.
os._exit(0)
PYEOF

RUN python3 /app/seed.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB native FTS + NLTK WordNet query-expansion task with a 100-row hand-rigged English corpus."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 nltk>=3.8
#   - NLTK_DATA=/usr/share/nltk_data with the WordNet + OMW-1.4 corpora downloaded
#   - /app/seed.py seeds the 100-row `docs` table in /app/lancedb_data (6 'car' docs,
#     8 'automobile' docs, 86 unrelated docs)
#   - LANCEDB_URI=/app/lancedb_data, LANCEDB_TABLE=docs
#   - /home/user/myproject is the empty project directory where the candidate writes solution.py
