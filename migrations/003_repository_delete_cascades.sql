-- Deleting a repository detaches it from every workspace instead of refusing.
-- SQLite cannot alter a foreign key in place, so rebuild workspace_sources.

CREATE TABLE workspace_sources_new (
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    UNIQUE(workspace_id, repository_id)
);

INSERT INTO workspace_sources_new (workspace_id, repository_id)
SELECT workspace_id, repository_id FROM workspace_sources;

DROP TABLE workspace_sources;
ALTER TABLE workspace_sources_new RENAME TO workspace_sources;

CREATE INDEX idx_workspace_sources_workspace ON workspace_sources(workspace_id);
