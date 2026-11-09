CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    display_name TEXT NOT NULL,
    internal_name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    instructions TEXT,
    default_agent_id INT REFERENCES agents(id) ON DELETE SET NULL,
    provider_key TEXT,
    model_id TEXT,
    inference_config JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS project_repository_access (
    id SERIAL PRIMARY KEY,
    project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    repository_id INT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    security_policy_override security_policy,
    UNIQUE(project_id, repository_id)
);

ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS project_id INT REFERENCES projects(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS conversations_project_id_idx
    ON conversations(project_id);
