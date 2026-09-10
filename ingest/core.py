"""
core.py — shared plumbing for the ingestion pipeline.

Holds config, the Voyage embedding client, the Claude vision captioner,
and the Supabase upsert. Loaders import from here so the integration
details live in exactly one place.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import voyageai
from ollama import chat
from supabase import create_client, Client


# ---------------------------------------------------------------------------
# Config — read from environment. Set these in a .env or your shell.
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]   # service key: needed to write
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
EMBED_MODEL = "voyage-3"          # 1024-dim, must match vector(1024) in schema
EMBED_DIM = 1024
CAPTION_MODEL = "qwen2.5vl:3b"
BATCH_SIZE = 128                  # Voyage accepts batches; keeps requests cheap

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
voyage = voyageai.Client(api_key=VOYAGE_API_KEY)

# ---------------------------------------------------------------------------
# Chunk: the unit every loader produces and the pipeline consumes.
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    content: str
    source_type: str                       # 'code' | 'pdf' | 'image' | 'note' | 'link'
    source_path: Optional[str] = None
    image_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Embedding — batched. input_type='document' tells Voyage these are corpus
# items (queries get input_type='query' at search time; it matters for quality).
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = voyage.embed(batch, model=EMBED_MODEL, input_type="document")
                out.extend(resp.embeddings)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)   # backoff on rate limit / transient
    return out


# ---------------------------------------------------------------------------
# Vision captioning — local Ollama vision model.
# Images are captioned locally, then the caption is embedded into the
# same Voyage text vector space used by the rest of the RAG pipeline.
# ---------------------------------------------------------------------------

def caption_image(image_b64: str, media_type: str, context: str = "") -> str:
    prompt = (
        "Describe this image in detail for a search index. Capture any text, "
        "diagrams, UI elements, code, charts, objects, and structure visible. "
        "Be specific and factual so someone could find this image by searching "
        "its contents. Return only the description."
    )

    if context:
        prompt += f"\\nContext about where this image came from: {context}"

    response = chat(
        model=CAPTION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        options={"num_ctx": 8192},
    )
    return response["message"]["content"].strip()


# Upsert — embed any un-embedded chunks, then write. on_conflict skips dupes
# via the (content_hash, source_path) unique index, so re-running ingest is safe.
# ---------------------------------------------------------------------------
def upsert_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    embeddings = embed_texts([c.content for c in chunks])

    rows = []
    for c, emb in zip(chunks, embeddings):
        rows.append({
            "content": c.content,
            "embedding": emb,
            "source_type": c.source_type,
            "source_path": c.source_path,
            "image_path": c.image_path,
            "metadata": c.metadata,
            "content_hash": c.content_hash,
        })

    supabase.table("rag_chunks").upsert(
        rows, on_conflict="content_hash,source_path", ignore_duplicates=True
    ).execute()
    return len(rows)

