CREATE TABLE IF NOT EXISTS repositories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    internal_name TEXT NOT NULL UNIQUE,
    repo_type TEXT NOT NULL CHECK (repo_type IN ('plain', 'git')),
    path TEXT NOT NULL,
    security_policy TEXT NOT NULL CHECK (security_policy IN ('read-only', 'privileged-write', 'write'))
);

CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    internal_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    instructions TEXT,
    provider_key TEXT NOT NULL,
    model_id TEXT NOT NULL,
    inference_config JSON NOT NULL DEFAULT '{}',
    spawnable_agents JSON NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS agent_repository_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    security_policy_override TEXT CHECK (security_policy_override IN ('read-only', 'privileged-write', 'write')),
    UNIQUE(agent_id, repository_id)
);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    internal_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    instructions TEXT,
    default_agent_id INTEGER REFERENCES agents(id) ON DELETE SET NULL,
    provider_key TEXT,
    model_id TEXT,
    inference_config JSON NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS project_repository_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    security_policy_override TEXT CHECK (security_policy_override IN ('read-only', 'privileged-write', 'write')),
    UNIQUE(project_id, repository_id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    blocking_message_id INTEGER,
    project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sequence_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    elements JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(conversation_id, sequence_id)
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL UNIQUE,
    file_hash TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding BLOB NOT NULL CHECK (length(embedding) = 3072),
    UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS inference_providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    base_url TEXT NOT NULL,
    api_key TEXT,
    models JSON NOT NULL DEFAULT '[]',
    timeout_seconds INTEGER NOT NULL DEFAULT 120,
    headers JSON NOT NULL DEFAULT '{}',
    extra_body JSON NOT NULL DEFAULT '{}',
    include_reasoning_content INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sd_cpp_image_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    recipe TEXT NOT NULL,
    files JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_repository_id ON documents(repository_id);
CREATE INDEX IF NOT EXISTS idx_conversations_project_id ON conversations(project_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_sequence ON messages(conversation_id, sequence_id);
