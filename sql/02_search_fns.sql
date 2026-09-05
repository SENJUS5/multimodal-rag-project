-- 02_search_fns.sql — the three search RPCs called by retrieval.py.
-- Run this in Supabase after 01_schema.sql.

-- ---------------------------------------------------------------------------
-- Semantic: cosine distance on the embedding. `<=>` is pgvector cosine distance
-- (smaller = closer), so we order ascending and convert to a similarity score.
-- ---------------------------------------------------------------------------
create or replace function rag_semantic_search(
    query_embedding vector(1024),
    match_count int default 10,
    filter_type text default null
)
returns table (
    id bigint, content text, source_type text, source_path text,
    image_path text, metadata jsonb, score float
)
language sql stable as $$
    select id, content, source_type, source_path, image_path, metadata,
           1 - (embedding <=> query_embedding) as score
    from rag_chunks
    where embedding is not null
      and (filter_type is null or source_type = filter_type)
    order by embedding <=> query_embedding
    limit match_count;
$$;

-- ---------------------------------------------------------------------------
-- Keyword: full-text rank over the generated `fts` column.
-- websearch_to_tsquery handles natural phrasing ("foo bar", quoted phrases, -negation).
-- ---------------------------------------------------------------------------
create or replace function rag_keyword_search(
    query_text text,
    match_count int default 10,
    filter_type text default null
)
returns table (
    id bigint, content text, source_type text, source_path text,
    image_path text, metadata jsonb, score float
)
language sql stable as $$
    select id, content, source_type, source_path, image_path, metadata,
           ts_rank(fts, websearch_to_tsquery('english', query_text)) as score
    from rag_chunks
    where fts @@ websearch_to_tsquery('english', query_text)
      and (filter_type is null or source_type = filter_type)
    order by score desc
    limit match_count;
$$;

-- ---------------------------------------------------------------------------
-- Hybrid: Reciprocal Rank Fusion. Rank each result list independently, then
-- combine by sum of 1/(rrf_k + rank). Scale-free, so cosine and ts_rank can't
-- drown each other out. Pull a wider candidate pool (match_count * 4) from each
-- side before fusing so good items ranked low in one list still survive.
-- ---------------------------------------------------------------------------
create or replace function rag_hybrid_search(
    query_text text,
    query_embedding vector(1024),
    match_count int default 10,
    filter_type text default null,
    rrf_k int default 60
)
returns table (
    id bigint, content text, source_type text, source_path text,
    image_path text, metadata jsonb, score float
)
language sql stable as $$
    with sem as (
        select id, row_number() over (order by embedding <=> query_embedding) as rnk
        from rag_chunks
        where embedding is not null
          and (filter_type is null or source_type = filter_type)
        order by embedding <=> query_embedding
        limit match_count * 4
    ),
    kw as (
        select id, row_number() over (
                 order by ts_rank(fts, websearch_to_tsquery('english', query_text)) desc
               ) as rnk
        from rag_chunks
        where fts @@ websearch_to_tsquery('english', query_text)
          and (filter_type is null or source_type = filter_type)
        limit match_count * 4
    ),
    fused as (
        select coalesce(sem.id, kw.id) as id,
               coalesce(1.0 / (rrf_k + sem.rnk), 0)
             + coalesce(1.0 / (rrf_k + kw.rnk), 0) as score
        from sem
        full outer join kw on sem.id = kw.id
    )
    select c.id, c.content, c.source_type, c.source_path, c.image_path,
           c.metadata, f.score
    from fused f
    join rag_chunks c on c.id = f.id
    order by f.score desc
    limit match_count;
$$;
