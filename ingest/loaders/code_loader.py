"""
code_loader.py — structure-aware chunking for source files.

Naive line-window chunking shreds functions in half and kills retrieval.
Instead we split on top-level definitions (functions/classes) where we can,
keeping the file path and symbol name in metadata so hybrid keyword search
can match exact identifiers like `watch-pipeline` or `upsert_chunks`.

Strategy:
  * Python: use the `ast` module to find def/class boundaries precisely.
  * Everything else: regex on common definition keywords as a decent fallback,
    then a size cap so no single chunk gets too large to embed well.
"""

import ast
import re
from pathlib import Path

from core import Chunk

MAX_CHARS = 4000          # soft cap per chunk; keeps embeddings focused
CODE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".sh", ".sql",
    ".html", ".css", ".json", ".yaml", ".yml", ".md", ".txt",
}


def _split_python(src: str) -> list[tuple[str, str]]:
    """Return (symbol_name, code) pairs at top-level def/class granularity."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []                      # fall back to generic splitter
    lines = src.splitlines(keepends=True)
    pieces, last = [], 0
    nodes = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for i, node in enumerate(nodes):
        start = node.lineno - 1
        end = nodes[i + 1].lineno - 1 if i + 1 < len(nodes) else len(lines)
        if start > last:               # module-level code before this def
            head = "".join(lines[last:start]).strip()
            if head:
                pieces.append(("<module>", head))
        pieces.append((node.name, "".join(lines[start:end]).rstrip()))
        last = end
    return pieces


def _split_generic(src: str) -> list[tuple[str, str]]:
    """Coarse split for non-Python: break before def/function/class lines."""
    pat = re.compile(r"^\s*(?:def |class |function |export |const |func |public |private )",
                     re.MULTILINE)
    marks = [m.start() for m in pat.finditer(src)]
    if not marks:
        return [("<file>", src)]
    marks = [0] + marks
    pieces = []
    for i, start in enumerate(marks):
        end = marks[i + 1] if i + 1 < len(marks) else len(src)
        seg = src[start:end].strip()
        if seg:
            pieces.append(("<segment>", seg))
    return pieces


def _enforce_size(pieces: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Split any oversized piece into line-windows so embeddings stay tight."""
    out = []
    for name, code in pieces:
        if len(code) <= MAX_CHARS:
            out.append((name, code))
            continue
        lines = code.splitlines(keepends=True)
        buf = ""
        for ln in lines:
            if len(buf) + len(ln) > MAX_CHARS and buf:
                out.append((name, buf.rstrip()))
                buf = ""
            buf += ln
        if buf.strip():
            out.append((name, buf.rstrip()))
    return out


def load_code_file(path: str) -> list[Chunk]:
    p = Path(path)
    if p.suffix not in CODE_EXTS:
        return []
    src = p.read_text(encoding="utf-8", errors="replace")
    if not src.strip():
        return []

    if p.suffix == ".py":
        pieces = _split_python(src) or _split_generic(src)
    else:
        pieces = _split_generic(src)
    pieces = _enforce_size(pieces)

    chunks = []
    for symbol, code in pieces:
        chunks.append(Chunk(
            content=code,
            source_type="code",
            source_path=str(p),
            metadata={"symbol": symbol, "language": p.suffix.lstrip("."), "file": p.name},
        ))
    return chunks
