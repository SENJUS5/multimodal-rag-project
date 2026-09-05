"""
ingest.py — orchestrator. Routes files/URLs to loaders, then upserts.

Usage:
    python ingest.py path /path/to/project       # walk a dir, ingest everything
    python ingest.py file /path/to/file.pdf       # single file
    python ingest.py link https://example.com     # single URL
    python ingest.py links urls.txt               # one URL per line

Run the schema (sql/01_schema.sql) in Supabase first.
"""

import sys
from pathlib import Path

from core import upsert_chunks
from loaders.code_loader import load_code_file, CODE_EXTS
from loaders.doc_loaders import (
    load_pdf, load_image, load_note, load_link, IMAGE_EXTS,
)

# directories we never want to crawl into
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
             "worktrees", ".pytest_cache", ".mypy_cache", "site-packages"}


def route_file(path: Path) -> list:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return load_pdf(str(path))
    if suf in IMAGE_EXTS:
        return load_image(str(path))
    if suf in {".md", ".txt"}:
        return load_note(str(path))
    if suf in CODE_EXTS:
        return load_code_file(str(path))
    return []                          # unknown type: skip


def ingest_path(root: str):
    root_p = Path(root)
    total = 0
    targets = [root_p] if root_p.is_file() else [
        p for p in root_p.rglob("*")
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
    ]
    for f in targets:
        try:
            chunks = route_file(f)
            if chunks:
                n = upsert_chunks(chunks)
                total += n
                print(f"  +{n:3d}  {f}")
        except Exception as e:
            print(f"  !!!   {f}  ({type(e).__name__}: {e})")
    print(f"\nDone. {total} chunks upserted from {root}.")


def ingest_links(urls: list[str]):
    total = 0
    for url in urls:
        try:
            chunks = load_link(url)
            n = upsert_chunks(chunks)
            total += n
            print(f"  +{n:3d}  {url}")
        except Exception as e:
            print(f"  !!!   {url}  ({type(e).__name__}: {e})")
    print(f"\nDone. {total} chunks upserted from {len(urls)} link(s).")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    mode, arg = sys.argv[1], sys.argv[2]

    if mode in ("path", "file"):
        ingest_path(arg)
    elif mode == "link":
        ingest_links([arg])
    elif mode == "links":
        urls = [l.strip() for l in Path(arg).read_text().splitlines() if l.strip()]
        ingest_links(urls)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
