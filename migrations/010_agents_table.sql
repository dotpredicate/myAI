CREATE TYPE security_policy AS ENUM ('read-only', 'privileged-write', 'write');

ALTER TABLE repositories ALTER COLUMN security_policy TYPE security_policy USING security_policy::security_policy;

CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    display_name TEXT NOT NULL,
    internal_name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    provider_key TEXT NOT NULL,
    model_id TEXT NOT NULL,
    inference_config JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_repository_access (
    id SERIAL PRIMARY KEY,
    agent_id INT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    repository_id INT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    security_policy_override security_policy,
    UNIQUE(agent_id, repository_id)
);