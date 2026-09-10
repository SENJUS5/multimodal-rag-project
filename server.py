"""
server.py — step 4: the MCP server. Exposes the RAG to Claude as tools.

Local stdio server: Claude Desktop launches this as a subprocess and discovers
its tools. The server is a passive provider — it never calls a model itself, it
just answers tool calls. Three tools, deliberately few (a focused tool surface
beats a sprawling one):

    rag_search        — hybrid search, the default. Use for almost everything.
    rag_search_typed  — same, scoped to one content type (code/pdf/image/note/link)
    rag_get_image     — given an image chunk's path, return the actual image bytes
                        so the model can view the diagram/screenshot, not just its caption

Run locally:
    pip install "mcp[cli]"
    python server.py            # stdio
Register with Claude Desktop: add to claude_desktop_config.json (see README).
"""

import sys
import base64
from pathlib import Path

# retrieval.py and core.py live in ingest/, alongside this file's directory.
# Add that folder to the import path so the server works no matter what
# working directory it's launched from (e.g. Claude Desktop launches it bare).
sys.path.insert(0, str(Path(__file__).resolve().parent / "ingest"))

from mcp.server.fastmcp import FastMCP, Image

from retrieval import hybrid_search, semantic_search, keyword_search

mcp = FastMCP("multimodal-rag")

IMAGE_MEDIA = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}


def _format(results: list[dict]) -> str:
    """Render results as compact text for the model. Token-aware: caps content,
    surfaces the metadata that aids follow-up (path, symbol, page, image_path)."""
    if not results:
        return "No matching chunks found."
    out = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata") or {}
        loc = meta.get("symbol") or meta.get("title") or meta.get("page") or ""
        header = f"[{i}] ({r['source_type']}) {r.get('source_path', '')}"
        if loc:
            header += f"  ·  {loc}"
        if r.get("image_path"):
            header += f"  ·  image_path={r['image_path']}"
        out.append(f"{header}\n{r['content'].strip()}")
    return "\n\n---\n\n".join(out)


@mcp.tool()
def rag_search(query: str, top_k: int = 8) -> str:
    """Search Brett's knowledge base (code, PDFs, image captions, notes, links)
    using hybrid semantic + keyword search. This is the default search tool —
    use it for conceptual questions AND exact identifiers (function names, file
    paths, error strings), since hybrid handles both. Returns ranked chunks with
    their source path and metadata. If a result has an image_path, you can call
    rag_get_image to view the actual image."""
    return _format(hybrid_search(query, top_k=top_k))


@mcp.tool()
def rag_search_typed(query: str, source_type: str, top_k: int = 8) -> str:
    """Hybrid search restricted to one content type. source_type must be one of:
    'code', 'pdf', 'image', 'note', 'link'. Use when you specifically want only
    code, or only notes, etc. — e.g. searching just the codebase for a function."""
    valid = {"code", "pdf", "image", "note", "link"}
    if source_type not in valid:
        return f"Invalid source_type {source_type!r}. Must be one of: {', '.join(sorted(valid))}."
    return _format(hybrid_search(query, top_k=top_k, source_type=source_type))


@mcp.tool()
def rag_get_image(image_path: str) -> Image:
    """Return a viewable version of an image, compressing large files."""

    from PIL import Image as PILImage
    import io

    p = Path(image_path)
    data = p.read_bytes()

    # Return original if already under 1 MB
    if len(data) <= 1_000_000:
        media = IMAGE_MEDIA.get(p.suffix.lower(), "image/png")
        return Image(data=data, format=media.split("/")[-1])

    # Resize and compress oversized images
    img = PILImage.open(io.BytesIO(data))
    img.thumbnail((1600, 1600))

    output = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    quality = 85
    while True:
        output.seek(0)
        output.truncate(0)
        img.save(output, format="JPEG", quality=quality, optimize=True)

        if output.tell() <= 900_000 or quality <= 40:
            break

        quality -= 5

    return Image(data=output.getvalue(), format="jpeg")

if __name__ == "__main__":
    mcp.run()
