"""
retrieval.py — step 3: search. Built in three layers so each is testable alone.

  semantic_search  — pure vector similarity (meaning)
  keyword_search   — pure full-text (exact tokens: identifiers, error strings)
  hybrid_search    — fuse the two with Reciprocal Rank Fusion (RRF)

Why RRF instead of adding raw scores: cosine distances and ts_rank values live
on totally different scales, so summing them lets one drown out the other. RRF
ignores the raw scores and fuses on *rank position* (1/(k+rank)), which is
scale-free and the standard hybrid-search merge. This is the part most tutorials
get wrong, and the part worth understanding for enterprise work.

Each search calls a Postgres RPC function (defined in sql/02_search_fns.sql)
so the heavy lifting stays in the database next to the indexes.
"""

from core import voyage, supabase, EMBED_MODEL


def _embed_query(q: str) -> list[float]:
    # input_type='query' (not 'document') — Voyage embeds queries and corpus
    # items into a shared space but asymmetrically; this measurably helps recall.
    return voyage.embed([q], model=EMBED_MODEL, input_type="query").embeddings[0]


def semantic_search(query: str, top_k: int = 10, source_type: str | None = None):
    """Vector similarity only. Good for paraphrased / conceptual queries."""
    rows = supabase.rpc("rag_semantic_search", {
        "query_embedding": _embed_query(query),
        "match_count": top_k,
        "filter_type": source_type,
    }).execute()
    return rows.data


def keyword_search(query: str, top_k: int = 10, source_type: str | None = None):
    """Full-text only. Good for exact identifiers, file names, error strings."""
    rows = supabase.rpc("rag_keyword_search", {
        "query_text": query,
        "match_count": top_k,
        "filter_type": source_type,
    }).execute()
    return rows.data


def hybrid_search(query: str, top_k: int = 10, source_type: str | None = None,
                  rrf_k: int = 60):
    """
    Fuse semantic + keyword via RRF, server-side. Returns the merged top_k.
    rrf_k=60 is the conventional constant; higher = flatter weighting of rank.
    """
    rows = supabase.rpc("rag_hybrid_search", {
        "query_text": query,
        "query_embedding": _embed_query(query),
        "match_count": top_k,
        "filter_type": source_type,
        "rrf_k": rrf_k,
    }).execute()
    return rows.data


if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]) or "how does the agent escalation chain work"
    print(f"Hybrid results for: {q!r}\n")
    for r in hybrid_search(q, top_k=5):
        meta = r.get("metadata", {})
        tag = meta.get("symbol") or meta.get("title") or meta.get("file") or ""
        print(f"[{r['source_type']:5}] {tag:30} {r['source_path']}")
        print(f"        {r['content'][:120].replace(chr(10), ' ')}...\n")
