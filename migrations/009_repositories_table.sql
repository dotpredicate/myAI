DROP TABLE IF EXISTS repositories;
CREATE TABLE IF NOT EXISTS repositories (
    id SERIAL PRIMARY KEY,
    display_name TEXT NOT NULL,
    internal_name TEXT UNIQUE NOT NULL,
    repo_type TEXT NOT NULL,
    path TEXT NOT NULL,
    security_policy TEXT NOT NULL
);