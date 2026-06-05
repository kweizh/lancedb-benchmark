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
RUN mkdir -p /home/user/myproject

# Pre-generate the Hive-partitioned Parquet dataset at /home/user/myproject/parquet_dataset/
# with 3 year partitions (2022, 2023, 2024), 200 rows each, 24-d float32 embeddings.
# The `year` column is encoded ONLY in the partition directory name, NOT in the file payload.
RUN python3 <<'PYEOF'
import os
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

OUT_DIR = "/home/user/myproject/parquet_dataset"
EMBED_DIM = 24
ROWS_PER_YEAR = 200
YEARS = [2022, 2023, 2024]

# Schema of the per-partition parquet files. The `year` column is intentionally
# OMITTED from the file payload so that it is encoded ONLY in the Hive partition
# directory name. Candidates must use `pyarrow.dataset.dataset(..., partitioning="hive")`
# to materialize `year` as a real Arrow column at read time.
file_schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("title", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), EMBED_DIM)),
])

if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

for year in YEARS:
    rng = np.random.default_rng(2026 + (year - 2022))
    base_id = (year - 2022) * ROWS_PER_YEAR
    ids = []
    titles = []
    embeddings = []
    for i in range(ROWS_PER_YEAR):
        emb = rng.standard_normal(EMBED_DIM).astype(np.float32)
        ids.append(base_id + i)
        titles.append(f"doc-{year}-{i:03d}")
        embeddings.append(emb.tolist())
    part_dir = os.path.join(OUT_DIR, f"year={year}")
    os.makedirs(part_dir, exist_ok=True)
    table = pa.table({
        "id": pa.array(ids, type=pa.int64()),
        "title": pa.array(titles, type=pa.string()),
        "embedding": pa.array(embeddings, type=pa.list_(pa.float32(), EMBED_DIM)),
    }, schema=file_schema)
    pq.write_table(table, os.path.join(part_dir, "part-0.parquet"))

# Sanity: per-partition files must NOT contain `year` as a physical column.
# Use ParquetFile (no partition discovery) instead of pq.read_table (auto-discovers
# Hive partition columns from the directory name in pyarrow >= 11).
for year in YEARS:
    pdir = os.path.join(OUT_DIR, f"year={year}")
    assert os.path.isdir(pdir), f"missing partition dir {pdir}"
    files = [f for f in os.listdir(pdir) if f.endswith(".parquet")]
    assert len(files) >= 1, f"no parquet file in {pdir}"
    pf = pq.ParquetFile(os.path.join(pdir, files[0]))
    file_cols = pf.schema_arrow.names
    assert "year" not in file_cols, (
        f"{pdir} parquet file unexpectedly has `year` as physical column: {file_cols}"
    )
    for col in ("id", "title", "embedding"):
        assert col in file_cols, f"{pdir} missing column {col}: {file_cols}"
    assert pf.metadata.num_rows == ROWS_PER_YEAR, (
        f"{pdir} expected {ROWS_PER_YEAR} rows, got {pf.metadata.num_rows}"
    )

# Confirm the open-side dataset surfaces `year` as a real column via Hive partitioning.
opened = ds.dataset(OUT_DIR, partitioning="hive")
opened_names = set(opened.schema.names)
assert {"id", "title", "embedding", "year"}.issubset(opened_names), \
    f"opened dataset missing required columns; got {opened_names}"
total = opened.to_table().num_rows
assert total == 600, f"expected 600 rows total, got {total}"

# Confirm that NOT passing `partitioning="hive"` would lose the `year` column entirely
# (this is the trap the candidate must avoid).
no_part = ds.dataset(OUT_DIR)
assert "year" not in no_part.schema.names, (
    "Dataset opened WITHOUT hive partitioning should NOT have `year` column; "
    "fixture is broken."
)

print(f"Wrote Hive-partitioned parquet dataset to {OUT_DIR} with {total} rows.")
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB task: ingest Hive-partitioned parquet dataset preserving year column and expose year-filtered vector search."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#   - /home/user/myproject/parquet_dataset: Hive-partitioned dataset
#       year=2022/, year=2023/, year=2024/ — 200 rows each, columns: id, title, embedding[24]
#   - /home/user/myproject/lancedb: empty; candidate's solution.py creates `articles_${ZEALT_RUN_ID}`
