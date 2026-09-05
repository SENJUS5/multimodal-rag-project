"""
doc_loaders.py — loaders for PDFs, images, notes, and links.

Each returns a list[Chunk]. Kept together because they're shorter than the
code loader and share the same simple shape.
"""

import base64
import re
from pathlib import Path

import pypdf
import requests
from bs4 import BeautifulSoup

from core import Chunk, caption_image

MAX_CHARS = 2000          # prose chunk target
OVERLAP = 200             # carry context across chunk boundaries

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
IMAGE_MEDIA = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}


def _window(text: str, size: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Sliding-window split on prose, breaking at paragraph/sentence where possible."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):
            brk = text.rfind("\n\n", start, end)
            if brk == -1:
                brk = text.rfind(". ", start, end)
            if brk > start:
                end = brk + 1
        out.append(text[start:end].strip())
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


# --- PDF: one logical chunk per page-group, page number in metadata -----------
def load_pdf(path: str) -> list[Chunk]:
    p = Path(path)
    reader = pypdf.PdfReader(str(p))
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue                      # scanned page w/ no text layer; OCR later if needed
        for seg in _window(text):
            chunks.append(Chunk(
                content=seg,
                source_type="pdf",
                source_path=str(p),
                metadata={"page": page_num, "file": p.name},
            ))
    return chunks


# --- Image: caption via Claude vision, embed the caption, keep image pointer ---
def load_image(path: str) -> list[Chunk]:
    p = Path(path)
    if p.suffix.lower() not in IMAGE_EXTS:
        return []
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    media = IMAGE_MEDIA[p.suffix.lower()]
    caption = caption_image(b64, media, context=f"File: {p.name}")
    if not caption:
        return []
    return [Chunk(
        content=caption,
        source_type="image",
        source_path=str(p),
        image_path=str(p),                # so retrieval can hand back the real image
        metadata={"file": p.name},
    )]


# --- Note: plain text / markdown, windowed -----------------------------------
def load_note(path: str) -> list[Chunk]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    return [Chunk(content=seg, source_type="note", source_path=str(p),
                  metadata={"file": p.name})
            for seg in _window(text)]


# --- Link: fetch URL, strip to readable text, windowed -----------------------
def load_link(url: str) -> list[Chunk]:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "rag-ingest/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title = (soup.title.string if soup.title else url) or url
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()
    return [Chunk(content=seg, source_type="link", source_path=url,
                  metadata={"title": title.strip(), "url": url})
            for seg in _window(text)]
