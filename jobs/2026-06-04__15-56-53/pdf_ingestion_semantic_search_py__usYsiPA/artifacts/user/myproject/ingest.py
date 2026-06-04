"""
ingest.py — PDF Ingestion & Semantic Search Pipeline
Reads PDFs from /app/corpus/, chunks text, embeds with OpenAI, stores in LanceDB.
"""

import os
import re
import pathlib
import lancedb
import pyarrow as pa
from pypdf import PdfReader
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORPUS_DIR = pathlib.Path("/app/corpus")
LANCEDB_DIR = "/home/user/myproject/lancedb"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 400          # characters per chunk
CHUNK_OVERLAP = 80        # character overlap between consecutive chunks

ZEALT_RUN_ID = os.environ.get("ZEALT_RUN_ID", "default")
TABLE_NAME = f"pdf_chunks_{ZEALT_RUN_ID}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_openai_client() -> OpenAI:
    """Return an OpenAI client, honouring OPENAI_BASE_URL if set."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping chunks of roughly *chunk_size* characters.
    Splits prefer sentence/word boundaries where possible.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # If not at the very end, try to break at a whitespace boundary
        if end < len(text):
            # Look back up to 40 chars for a space or newline
            boundary = text.rfind(" ", start + chunk_size - 40, end)
            if boundary == -1:
                boundary = text.rfind("\n", start + chunk_size - 40, end)
            if boundary != -1:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = end - overlap  # move back by overlap
        if start <= 0:
            break

    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in one API call (respects max 2048 inputs/call)."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # response.data is sorted by index
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest() -> None:
    client = get_openai_client()
    db = lancedb.connect(LANCEDB_DIR)

    # Drop existing table for this run ID to ensure idempotence
    existing_tables = db.table_names()
    if TABLE_NAME in existing_tables:
        print(f"Table '{TABLE_NAME}' already exists — dropping and re-creating for idempotence.")
        db.drop_table(TABLE_NAME)

    # Build rows
    rows: list[dict] = []
    pdf_files = sorted(CORPUS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    all_chunks: list[tuple[str, int, int, str]] = []  # (doc_id, page, chunk_id, text)

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        reader = PdfReader(str(pdf_path))
        print(f"  [{doc_id}] {len(reader.pages)} pages")
        chunk_counter = 0
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            chunks = chunk_text(page_text)
            for chunk_text_value in chunks:
                all_chunks.append((doc_id, page_num, chunk_counter, chunk_text_value))
                chunk_counter += 1

    print(f"Total chunks to embed: {len(all_chunks)}")

    # Embed in batches of 100
    BATCH_SIZE = 100
    embeddings: list[list[float]] = []
    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c[3] for c in batch]
        print(f"  Embedding batch {batch_start // BATCH_SIZE + 1} "
              f"({len(texts)} chunks)…")
        embeddings.extend(embed_texts(client, texts))

    assert len(embeddings) == len(all_chunks), "Mismatch between chunks and embeddings"

    # Determine embedding dimension
    dim = len(embeddings[0])
    print(f"Embedding dimension: {dim}")

    # Assemble rows
    for (doc_id, page, chunk_id, text), embedding in zip(all_chunks, embeddings):
        rows.append({
            "doc_id": doc_id,
            "page": page,
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding,
        })

    # Define schema
    schema = pa.schema([
        pa.field("doc_id", pa.string()),
        pa.field("page", pa.int32()),
        pa.field("chunk_id", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), dim)),
    ])

    table = db.create_table(TABLE_NAME, data=rows, schema=schema)
    print(f"Created LanceDB table '{TABLE_NAME}' with {table.count_rows()} rows.")


if __name__ == "__main__":
    ingest()
