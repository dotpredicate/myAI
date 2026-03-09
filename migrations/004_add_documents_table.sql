-- Migration to add pgvector extension and documents table

-- Enable the pgvector extension if not already enabled - needs to be done by the superuser
-- CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table for RAG
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL
);
