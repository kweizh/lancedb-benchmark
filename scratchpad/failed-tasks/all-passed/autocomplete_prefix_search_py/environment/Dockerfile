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

ENV LANCE_DB_URI=/app/lance_db
ENV MOVIES_TABLE=movies
ENV PREFIX_TABLE=prefix_vectors

# Seed the LanceDB instance at build time so the container starts in the
# state required by test_initial_state.py. Two tables are created:
#   - movies (500 rows): id, title, title_lower, popularity, vector(32)
#   - prefix_vectors (50 rows): prefix, vector(32)
# Engineered guarantees:
#   - "crystal" prefix yields >= 10 movies
#   - "zephyr" prefix yields exactly 2 movies
#   - "orbital" prefix yields 0 movies and is present in prefix_vectors
RUN mkdir -p /app && python3 - <<'PYEOF'
import os
import numpy as np
import pyarrow as pa
import lancedb

ADJECTIVES = [
    "Crystal", "Phantom", "Silent", "Hidden", "Frozen",
    "Burning", "Eternal", "Sacred", "Forgotten", "Wild",
    "Quiet", "Dark", "Bright", "Ancient", "Mystic",
    "Cosmic", "Quantum", "Solar", "Lunar", "Stellar",
    "Iron", "Golden", "Silver", "Ruby", "Emerald",
    "Diamond", "Velvet", "Steel", "Ivory", "Onyx",
]
assert len(ADJECTIVES) == 30 and len(set(ADJECTIVES)) == 30

NOUNS = [
    "Forest", "River", "Mountain", "Valley", "Castle",
    "Temple", "Garden", "Ocean", "Desert", "Storm",
    "Dream", "Memory", "Shadow", "Light", "Echo",
    "Voice", "Heart", "Path", "Bridge", "Tower",
    "Mirror", "Crown", "Mask", "Ring", "Sword",
    "Symphony", "Whisper", "Flame", "Tide", "Compass",
    "Cathedral", "Lantern", "Carousel", "Meadow", "Harbor",
    "Glacier", "Canyon", "Reef", "Atrium", "Pavilion",
    "Lighthouse", "Library", "Museum", "Sanctuary", "Observatory",
    "Marketplace", "Foundry", "Workshop", "Bazaar", "Cloister",
]
assert len(NOUNS) == 50 and len(set(NOUNS)) == 50

# Generate 498 unique deterministic titles from a controlled enumeration.
seen = set()
records = []
i = 0
while len(records) < 498:
    adj = ADJECTIVES[i % 30]
    noun = NOUNS[(i // 30) % 50]
    year = 1990 + ((i // 1500) % 36)
    title = f"{adj} {noun} {year}"
    if title not in seen:
        seen.add(title)
        records.append(title)
    i += 1
    if i > 200000:
        raise RuntimeError("Could not enumerate 498 unique titles.")

# Sanity check the engineered prefix counts so the verifier will pass.
crystal_count = sum(1 for t in records if t.lower().startswith("crystal"))
assert crystal_count >= 10, f"Expected >=10 'crystal' titles, got {crystal_count}"
zephyr_pre = sum(1 for t in records if t.lower().startswith("zephyr"))
assert zephyr_pre == 0, f"Pre-injection 'zephyr' count must be 0, got {zephyr_pre}"
orbital_pre = sum(1 for t in records if t.lower().startswith("orbital"))
assert orbital_pre == 0, f"'orbital' count must be 0, got {orbital_pre}"

# Inject exactly two "Zephyr ..." titles.
records.append("Zephyr Symphony 2024")
records.append("Zephyr Tide 1999")
assert len(records) == 500
assert len(set(records)) == 500, "Titles must be unique."

# Deterministic 32-d float32 embeddings for movies.
rng = np.random.default_rng(2026)
movie_vectors = rng.standard_normal((500, 32)).astype(np.float32)

# Deterministic, unique popularity scores via a permutation.
pop_rng = np.random.default_rng(20262)
popularity = pop_rng.permutation(500).astype(np.float64) / 500.0
assert len(set(popularity.tolist())) == 500, "Popularity values must be unique."

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("title", pa.string()),
    pa.field("title_lower", pa.string()),
    pa.field("popularity", pa.float64()),
    pa.field("vector", pa.list_(pa.float32(), 32)),
])

ids = list(range(500))
titles = records
titles_lower = [t.lower() for t in titles]
vector_col = [v.tolist() for v in movie_vectors]

movies_table = pa.table(
    {
        "id": ids,
        "title": titles,
        "title_lower": titles_lower,
        "popularity": popularity.tolist(),
        "vector": vector_col,
    },
    schema=schema,
)

# Prefix vectors: 30 adjective prefixes (lowercase) + zephyr + 19 phantom prefixes.
PHANTOMS = [
    "orbital", "neutron", "tachyon", "synapse", "vortex",
    "plasma", "fjord", "obsidian", "twilight", "ember",
    "horizon", "polaris", "andromeda", "celestial", "luminous",
    "radiant", "spectral", "nebula", "meridian",
]
assert len(PHANTOMS) == 19 and len(set(PHANTOMS)) == 19

prefix_strings = [a.lower() for a in ADJECTIVES] + ["zephyr"] + PHANTOMS
assert len(prefix_strings) == 50, f"Got {len(prefix_strings)} prefixes."
assert len(set(prefix_strings)) == 50, "Prefix strings must be unique."

# Confirm "orbital" yields zero prefix matches over the full title set.
for p in PHANTOMS:
    cnt = sum(1 for t in titles if t.lower().startswith(p))
    assert cnt == 0, f"Phantom prefix '{p}' unexpectedly matched {cnt} titles"

pre_rng = np.random.default_rng(123456)
prefix_vectors = pre_rng.standard_normal((50, 32)).astype(np.float32)

prefix_schema = pa.schema([
    pa.field("prefix", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 32)),
])

prefix_table = pa.table(
    {
        "prefix": prefix_strings,
        "vector": [v.tolist() for v in prefix_vectors],
    },
    schema=prefix_schema,
)

db_uri = "/app/lance_db"
os.makedirs(db_uri, exist_ok=True)
db = lancedb.connect(db_uri)
if "movies" in db.table_names():
    db.drop_table("movies")
if "prefix_vectors" in db.table_names():
    db.drop_table("prefix_vectors")
db.create_table("movies", movies_table)
db.create_table("prefix_vectors", prefix_table)

# Final post-seed sanity assertions.
mt = db.open_table("movies")
pt = db.open_table("prefix_vectors")
assert mt.count_rows() == 500
assert pt.count_rows() == 50

print("Seed complete: movies=500, prefix_vectors=50")
# Avoid lancedb 0.25.3 SIGABRT during interpreter teardown.
os._exit(0)
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB autocomplete service task: 500-movie + 50-prefix tables seeded for SQL prefix + semantic fallback verification."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf, lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /home/user/myproject: candidate project directory (empty; candidate creates solution.py)
#   - /app/lance_db: seeded LanceDB instance with two tables (movies, prefix_vectors)
#   - ENV LANCE_DB_URI, MOVIES_TABLE, PREFIX_TABLE: configuration the candidate reads at runtime
