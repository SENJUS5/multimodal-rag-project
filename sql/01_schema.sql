-- Multimodal RAG schema for Supabase (Postgres + pgvector)
-- Run this in the Supabase SQL editor, or via psql against your project.
--
-- Design notes:
--   * Single `chunks` table holds every content type (code, pdf, image, note, link).
--   * `embedding` is the semantic half of hybrid search (Voyage voyage-3 = 1024 dims).
--   * `fts` is the keyword half: a generated tsvector kept in sync automatically.
--   * `metadata` jsonb absorbs type-specific fields (file_path, symbol, url, page, etc.)
--     so we never have to migrate the schema when a new content type needs a new field.

-- 1. Enable pgvector (Supabase has it available; this just turns it on for your project).
create extension if not exists vector;

-- 2. Main table.
create table if not exists rag_chunks (
    id           bigint generated always as identity primary key,
    content      text        not null,                 -- the text that gets embedded + shown
    embedding    vector(1024),                          -- voyage-3 output; null until embedded
    source_type  text        not null,                  -- 'code' | 'pdf' | 'image' | 'note' | 'link'
    source_path  text,                                  -- file path or URL the chunk came from
    image_path   text,                                  -- set only for image chunks; pointer to the actual image
    metadata     jsonb       not null default '{}',     -- type-specific extras (symbol, page, lang, title...)
    content_hash text        not null,                  -- sha256 of content; dedupe + idempotent re-ingest
    created_at   timestamptz not null default now(),

    -- generated full-text column: stays in sync with `content`, no triggers needed.
    fts tsvector generated always as (to_tsvector('english', content)) stored
);

-- 3. Dedupe guard: same text from the same source won't get inserted twice.
create unique index if not exists rag_chunks_hash_uq
    on rag_chunks (content_hash, source_path);

-- 4. Vector index for semantic search.
--    HNSW = fast approximate nearest neighbor, good default for read-heavy personal RAG.
--    vector_cosine_ops because we'll normalize and compare by cosine distance.
create index if not exists rag_chunks_embedding_idx
    on rag_chunks using hnsw (embedding vector_cosine_ops);

-- 5. GIN index for the keyword half of hybrid search.
create index if not exists rag_chunks_fts_idx
    on rag_chunks using gin (fts);

-- 6. Cheap filter index for narrowing by content type ("only search my code").
create index if not exists rag_chunks_source_type_idx
    on rag_chunks (source_type);
