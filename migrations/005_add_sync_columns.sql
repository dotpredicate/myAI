TRUNCATE TABLE documents;

ALTER TABLE documents
    ADD COLUMN file_path TEXT UNIQUE,
    ADD COLUMN file_hash CHAR(64),
    ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE INDEX idx_documents_embedding
ON documents
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);