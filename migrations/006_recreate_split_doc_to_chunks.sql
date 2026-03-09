DROP TABLE documents;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    file_path TEXT UNIQUE,
    file_hash CHAR(64),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content TEXT NOT NULL
);

CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
ON document_chunks USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);