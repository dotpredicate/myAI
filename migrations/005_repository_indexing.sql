-- Indexing is opt-in per repository. Existing repositories keep their files
-- but stop being indexed until the user checks "Index for search".
ALTER TABLE repositories ADD COLUMN index_enabled INTEGER NOT NULL DEFAULT 0;
