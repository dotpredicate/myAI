-- Workspace MVP: workspaces replace projects, repository access stops being a
-- security-policy dimension, agents get an optional model, and conversations
-- store the last model used by a prompt. Nothing is carried over from projects.

CREATE TABLE IF NOT EXISTS workspaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS workspace_sources (
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE RESTRICT,
    UNIQUE(workspace_id, repository_id)
);

-- Install the first workspace so the database is immediately usable.
-- It is a normal workspace: it can be renamed, given sources, or deleted when
-- another workspace exists.
INSERT INTO workspaces (name)
SELECT 'Scratch'
WHERE NOT EXISTS (SELECT 1 FROM workspaces);

-- Projects are dropped, not migrated. Conversations are bound to a workspace,
-- not to a project; the old column is dropped after workspace_id is filled in.
DROP INDEX IF EXISTS idx_conversations_project_id;
DROP TABLE IF EXISTS project_repository_access;
DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS agent_repository_access;

-- Rebuild agents: model configuration becomes optional.
ALTER TABLE agents RENAME TO agents_old;
CREATE TABLE agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    internal_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    instructions TEXT,
    provider_key TEXT,
    model_id TEXT,
    inference_config JSON NOT NULL DEFAULT '{}',
    spawnable_agents JSON NOT NULL DEFAULT '[]'
);
INSERT INTO agents
    (id, display_name, internal_name, description, instructions,
     provider_key, model_id, inference_config, spawnable_agents)
SELECT id, display_name, internal_name, description, instructions,
       provider_key, model_id, inference_config, spawnable_agents
FROM agents_old;
DROP TABLE agents_old;

-- A conversation remembers the last model used by a prompt.
ALTER TABLE conversations ADD COLUMN workspace_id INTEGER REFERENCES workspaces(id) ON DELETE SET NULL;
ALTER TABLE conversations ADD COLUMN provider_key TEXT;
ALTER TABLE conversations ADD COLUMN model_id TEXT;
ALTER TABLE conversations ADD COLUMN agent_id TEXT;
ALTER TABLE conversations ADD COLUMN inference_config JSON NOT NULL DEFAULT '{}';

-- Conversations belong to a workspace; existing ones join the first workspace.
UPDATE conversations
SET workspace_id = (SELECT id FROM workspaces ORDER BY id LIMIT 1)
WHERE workspace_id IS NULL;

ALTER TABLE conversations DROP COLUMN project_id;

-- Repositories no longer carry a security policy. Shell access is read-only;
-- writes go through approved replace_file/apply_patch calls.
ALTER TABLE repositories DROP COLUMN security_policy;

CREATE UNIQUE INDEX IF NOT EXISTS idx_repositories_path_unique ON repositories(path);
CREATE INDEX IF NOT EXISTS idx_conversations_workspace_id ON conversations(workspace_id);
CREATE INDEX IF NOT EXISTS idx_workspace_sources_workspace ON workspace_sources(workspace_id);
