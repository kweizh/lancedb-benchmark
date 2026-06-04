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
        click && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/loader_project && chmod 0755 /home/user/loader_project
RUN mkdir -p /app/data

# Build-time fixture: generate /app/data/articles.csv with 5000 rows.
# Two marker rows (id=4242 photosynthesis, id=1337 Apollo 11) are surrounded
# by 4998 filler rows using a strictly neutral office vocabulary, so the
# semantic distance from the two test queries to the marker rows is huge.
RUN cat >/tmp/generate_articles_csv.py <<'PY'
import csv
import os
import random

OUT_PATH = "/app/data/articles.csv"
SEED = 2026
N_ROWS = 5000

OFFICE_WORDS = [
    "meeting", "agenda", "quarterly", "budget", "fiscal", "report",
    "deadline", "project", "stakeholder", "deliverable", "milestone",
    "schedule", "calendar", "attendee", "minutes", "action", "item",
    "review", "summary", "update", "status", "synchronization", "alignment",
    "team", "department", "office", "headquarters", "branch", "manager",
    "director", "executive", "policy", "procedure", "workflow", "operation",
    "logistics", "compliance", "audit", "training", "session", "workshop",
    "briefing", "conference", "presentation", "memo", "draft", "approval",
    "signoff", "checklist", "rollout", "implementation", "evaluation",
    "vendor", "supplier", "contract", "procurement", "renewal", "expense",
    "reimbursement", "invoice", "ledger", "spreadsheet", "dashboard",
    "metrics", "indicator", "objective", "roadmap", "phase", "iteration",
    "sprint", "retrospective", "feedback", "annual", "monthly", "weekly",
    "headcount", "onboarding", "offboarding", "appraisal", "promotion",
    "merger", "acquisition", "restructuring", "outsourcing", "consultancy",
]

CATEGORIES = ["operations", "finance", "logistics", "hr", "compliance"]


def gen_filler_body(rng):
    n = rng.randint(25, 45)
    words = rng.choices(OFFICE_WORDS, k=n)
    return (" ".join(words).capitalize() + ".")


def gen_filler_title(rng):
    n = rng.randint(3, 6)
    return " ".join(w.capitalize() for w in rng.choices(OFFICE_WORDS, k=n))


def date_for_id(idx):
    year = 2024 + (idx % 2)
    month = ((idx * 7) % 12) + 1
    day = ((idx * 13) % 28) + 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    rows = []
    for idx in range(1, N_ROWS + 1):
        if idx == 4242:
            title = "Photosynthesis: chloroplasts, sunlight, and glucose production in plants"
            body = (
                "Photosynthesis is the biochemical process by which green plants and "
                "certain organisms use chlorophyll-bearing chloroplasts to convert "
                "sunlight, carbon dioxide, and water into glucose and oxygen, sustaining "
                "nearly all life on Earth's terrestrial and aquatic ecosystems."
            )
            category = "biology"
            published = "2024-04-22"
        elif idx == 1337:
            title = "Apollo 11: Armstrong, Aldrin, and the first crewed Moon landing"
            body = (
                "The Apollo 11 mission was the first crewed flight to land humans on the "
                "Moon. On July 20, 1969, astronauts Neil Armstrong and Buzz Aldrin walked "
                "on the lunar surface while Michael Collins orbited above in the command "
                "module, completing one of the defining achievements of human spaceflight."
            )
            category = "space"
            published = "2024-07-20"
        else:
            local_rng = random.Random(SEED * 1000003 + idx)
            title = gen_filler_title(local_rng)
            body = gen_filler_body(local_rng)
            category = CATEGORIES[idx % len(CATEGORIES)]
            published = date_for_id(idx)
        rows.append({
            "id": idx,
            "title": title,
            "body": body,
            "category": category,
            "published": published,
        })

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "title", "body", "category", "published"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
PY
RUN python3 /tmp/generate_articles_csv.py && \
    rm -f /tmp/generate_articles_csv.py && \
    chmod 0644 /app/data/articles.csv

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB CSV bulk loader CLI task: a pre-baked 5000-row articles.csv lives at /app/data, the executor implements /home/user/loader_project/loader.py with ingest/search subcommands backed by real OpenAI text-embedding-3-small."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - LanceDB stack: lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - OpenAI client: openai==1.54.5, tiktoken, click
#   - /home/user/loader_project: empty project directory where the executor writes loader.py
#   - /app/data/articles.csv: deterministic 5000-row CSV (seed=2026) with two marker rows (id=4242 photosynthesis, id=1337 Apollo 11)
