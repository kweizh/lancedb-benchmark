#!/usr/bin/env python3
"""Ingest PDFs from /app/corpus/ into a LanceDB table with OpenAI embeddings."""

import os
import glob

import lancedb
from openai import OpenAI
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORPUS_DIR = "/app/corpus/"
LANCEDB_URI = "/home/user/myproject/lancedb/"
TABLE_NAME = f"pdf_chunks_{os.environ['ZEALT_RUN_ID']}"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI's embedding API (batched)."""
    # OpenAI supports batching; we'll send up to 2048 texts at a time.
    all_embeddings: list[list[float]] = []
    batch_size = 2048
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        # Preserve order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([item.embedding for item in sorted_data])
    return all_embeddings


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract (page_number, text) pairs from a PDF. Page numbers are 1-based."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of approximately `chunk_size` characters."""
    if not text.strip():
        return []
    # If the text is shorter than chunk_size, return it as a single chunk.
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest() -> None:
    """Read all PDFs, chunk, embed, and store in LanceDB."""
    pdf_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {CORPUS_DIR}")

    # Collect all chunks with metadata.
    records: list[dict] = []
    for pdf_path in pdf_files:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        pages = extract_pages(pdf_path)
        for page_num, page_text in pages:
            chunks = chunk_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                records.append(
                    {
                        "doc_id": doc_id,
                        "page": page_num,
                        "chunk_id": f"{doc_id}_p{page_num}_c{chunk_idx}",
                        "text": chunk,
                    }
                )

    if not records:
        raise ValueError("No text extracted from any PDF.")

    # Embed all chunk texts.
    texts = [r["text"] for r in records]
    embeddings = embed_texts(texts)

    # Attach embeddings to records.
    for rec, emb in zip(records, embeddings):
        rec["embedding"] = emb

    # Connect to LanceDB and create / overwrite the table.
    db = lancedb.connect(LANCEDB_URI)

    # Drop existing table with the same name to make ingestion idempotent.
    existing_tables = db.table_names()
    if TABLE_NAME in existing_tables:
        db.drop_table(TABLE_NAME)

    db.create_table(TABLE_NAME, records)
    print(f"Ingested {len(records)} chunks into table '{TABLE_NAME}'.")


if __name__ == "__main__":
    ingest()