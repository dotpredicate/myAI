-- OpenAI-compatible providers configured by the user. The built-in
-- llama.cpp provider stays registered in code; this table is for external
-- endpoints.

CREATE TABLE inference_providers (
    id SERIAL PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'openai_compatible',
    base_url TEXT NOT NULL,
    api_key TEXT,
    models JSONB NOT NULL DEFAULT '[]',
    timeout_seconds INT NOT NULL DEFAULT 120,
    headers JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB NOT NULL DEFAULT '{}',
    include_reasoning_content BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT inference_providers_kind_check CHECK (kind = 'openai_compatible')
);
