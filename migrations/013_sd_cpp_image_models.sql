DROP TABLE IF EXISTS sd_cpp_image_models;
CREATE TABLE IF NOT EXISTS sd_cpp_image_models (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    recipe TEXT NOT NULL,
    files JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
