-- Link indexed documents to their repository so deletion cascades and searches
-- no longer rely on matching a file_path prefix.

ALTER TABLE documents ADD COLUMN repository_id INT;

UPDATE documents d
SET repository_id = r.id
FROM repositories r
WHERE left(d.file_path, char_length(r.internal_name) + 1) = r.internal_name || '/';

-- Documents that do not belong to any registered repository are stale.
DELETE FROM documents WHERE repository_id IS NULL;

ALTER TABLE documents ALTER COLUMN repository_id SET NOT NULL;

ALTER TABLE documents
    ADD CONSTRAINT documents_repository_id_fkey
    FOREIGN KEY (repository_id) REFERENCES repositories(id) ON DELETE CASCADE;

CREATE INDEX idx_documents_repository_id ON documents(repository_id);
