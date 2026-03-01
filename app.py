from flask import Flask, jsonify, request, Response
import psycopg2
import os
import glob
import requests
from datetime import datetime

# Ollama API base URL
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434/api')

# Utility functions for Ollama

def get_ollama_models():
    """Return list of models available from Ollama."""
    try:
        res = requests.get(f"{OLLAMA_ENDPOINT}/tags")
        res.raise_for_status()
        data = res.json()
        return [{'id': m['name'], 'name': m['name']} for m in data.get('models', [])]
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []


def ask_model_stream(model_id, prompt_text):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": True
    }
    try:
        res = requests.post(f"{OLLAMA_ENDPOINT}/chat", json=payload, stream=True)
        res.raise_for_status()
        for line in res.iter_lines():
            if not line:
                continue
            yield line
    except Exception as e:
        yield f"Error: {e}"


import json
import hashlib
import binascii

app = Flask(__name__)

# Database connection
conn = psycopg2.connect(
    dbname='myai',
    user='myai',
    password='myai',
    host='localhost',
    port='5432'
)

# Initialize database schema and migrations table
def init_database():
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS migrations ( 
                order_executed INTEGER GENERATED ALWAYS AS IDENTITY,
                name VARCHAR(255) NOT NULL,
                hash CHARACTER(64) NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL
            )
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS migrations_name ON migrations (name)
        """)
    # Apply migrations
    migration_dir = 'migrations'
    migration_files = sorted(glob.glob(os.path.join(migration_dir, '*.sql')))
    for migration_file in migration_files:
        migration_name = os.path.basename(migration_file)
        # Check if migration already exists
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM migrations WHERE name = %s", (migration_name,))
            if cur.fetchone():
                print(f'Skipping already executed migration: {migration_name}')
                continue
        # Execute migration SQL
        with open(migration_file, 'r') as f:
            sql = f.read()
        with conn.cursor() as cur:
            cur.execute(sql)
        # Record migration execution
        start_time = datetime.now()
        hash_bytes = hashlib.sha256(migration_name.encode('utf-8')).digest()
        hash_str = binascii.hexlify(hash_bytes).decode('utf-8')
        end_time = datetime.now()
        with conn.cursor() as cur:
            cur.execute("INSERT INTO migrations (name, hash, start_time, end_time) VALUES (%s, %s, %s, %s)",
                        (migration_name, hash_str, start_time, end_time))
        conn.commit()
        print(f'Applied migration: {migration_file}')

@app.route('/api/models')
def get_models():
    return jsonify({"models": get_ollama_models()})

@app.route('/api/prompt', methods=['POST'])
def prompt_model():
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    model_id = data.get('model_id', '0')
    reply = ask_model_stream(model_id, prompt)
    return Response((chunk for chunk in reply), mimetype='text/event-stream')

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    init_database()
    app.run(debug=True)
