FROM ubuntu:24.04

ARG POCHI_VERSION

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    pip3 install --break-system-packages pytest-json-ctrf lancedb duckdb pyarrow numpy pandas && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

# Generate the deterministic input data (documents.jsonl + categories.csv).
RUN mkdir -p /home/user/project/data && python3 - <<'PYEOF'
import csv
import json
import os
import random

OUT_DIR = "/home/user/project/data"
os.makedirs(OUT_DIR, exist_ok=True)

DIM = 8

CATEGORIES = [
    ("Women's Apparel", "Fashion", 0.08),
    ("Men's Shoes", "Fashion", 0.08),
    ("Kids' Toys", "Toys & Games", 0.05),
    ("Board Games", "Toys & Games", 0.05),
    ("Kitchen", "Home", 0.07),
    ("Garden", "Home", 0.07),
]

TITLE_WORDS = [
    "Deluxe", "Classic", "Premium", "Eco", "Compact", "Vintage", "Modern",
    "Ultra", "Handmade", "Portable", "Cozy", "Rustic", "Smart", "Bright",
]
NOUNS = ["Set", "Kit", "Bundle", "Collection", "Pack", "Edition", "Model"]

rng = random.Random(20240715)

docs = []
doc_id = 1
for cat, dept, tax in CATEGORIES:
    for _ in range(8):
        vec = [round(rng.uniform(0.0, 1.0), 5) for _ in range(DIM)]
        price = round(rng.uniform(5.0, 150.0), 2)
        in_stock = rng.random() < 0.8
        title = f"{rng.choice(TITLE_WORDS)} {cat.split()[0].strip(chr(39))} {rng.choice(NOUNS)} #{doc_id}"
        docs.append(
            {
                "id": doc_id,
                "title": title,
                "category": cat,
                "price": price,
                "in_stock": in_stock,
                "vector": vec,
            }
        )
        doc_id += 1

rng.shuffle(docs)

with open(os.path.join(OUT_DIR, "documents.jsonl"), "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d) + "\n")

with open(os.path.join(OUT_DIR, "categories.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["category", "department", "tax_rate"])
    for cat, dept, tax in CATEGORIES:
        w.writerow([cat, dept, tax])

print(f"Wrote {len(docs)} documents to {OUT_DIR}")
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="Local LanceDB + embedded DuckDB hybrid search/analytics environment with a deterministic document + category dataset in /home/user/project/data."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - target/task pip3 packages: lancedb (LanceDB Python SDK), duckdb (embedded SQL engine), pyarrow, numpy, pandas
#   - /home/user/project: task working directory
#   - /home/user/project/data/documents.jsonl: deterministic 48-row document corpus (id/title/category/price/in_stock/8-dim vector)
#   - /home/user/project/data/categories.csv: auxiliary category -> department/tax_rate table (some category names contain apostrophes)
#   - No network/cloud services required; everything runs locally.

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
