from flask import Flask, jsonify, request, Response
import psycopg2
import os
import glob
import requests
from datetime import datetime
import json
import hashlib
import binascii
from typing import List, Dict, Optional, Any

app = Flask(__name__)
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434/api')

conn = psycopg2.connect(
    dbname='myai',
    user='myai',
    password='myai',
    host='localhost',
    port='5432'
)

def create_conversation(conn: psycopg2.extensions.connection, title: str) -> int:
    """Insert a new conversation row with title and return its id."""
    with conn.cursor() as cur:
        cur.execute("INSERT INTO conversations (title) VALUES (%s) RETURNING id", (title,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to create conversation")
        conv_id = row[0]
        conn.commit()
        return conv_id


def insert_message(conn: psycopg2.extensions.connection, conv_id: int, role: str, elements: List[Dict[str, str]]) -> None:
    """Persist a full array of elements for a message."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages (conversation_id, role, elements, created_at)
            VALUES (%s, %s, %s, NOW())
            """,
            (conv_id, role, json.dumps(elements))
        )
        conn.commit()

def ask_model_stream(model_id: str, prompt_text: str):
    """Return a generator yielding raw bytes from Ollama streaming API."""
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
        yield f"Error: {e}".encode()

def init_database():
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS migrations (
                order_executed INTEGER GENERATED ALWAYS AS IDENTITY,
                name VARCHAR(255) NOT NULL,
                hash CHARACTER(64) NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS migrations_name ON migrations (name);
            """
        )
        conn.commit()

    migration_dir = 'migrations'
    migration_files = sorted(glob.glob(os.path.join(migration_dir, '*.sql')))
    for migration_file in migration_files:
        migration_name = os.path.basename(migration_file)
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM migrations WHERE name = %s", (migration_name,))
            if cur.fetchone():
                print(f'Skipping already executed migration: {migration_name}')
                continue
        with open(migration_file, 'r') as f:
            sql = f.read()
        with conn.cursor() as cur:
            cur.execute(sql)
        start_time = datetime.now()
        hash_bytes = hashlib.sha256(migration_name.encode('utf-8')).digest()
        hash_str = binascii.hexlify(hash_bytes).decode('utf-8')
        end_time = datetime.now()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO migrations (name, hash, start_time, end_time)
                VALUES (%s, %s, %s, %s)
                """,
                (migration_name, hash_str, start_time, end_time)
            )
        conn.commit()
        print(f'Applied migration: {migration_file}')

@app.route('/api/models')
def get_models():
    try:
        res = requests.get(f"{OLLAMA_ENDPOINT}/tags")
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"Error fetching models: {e}")
        data = {"models": []}
    return jsonify({"models": [{'id': m['name'], 'name': m['name']} for m in data.get('models', [])]})

@app.route('/api/prompt', methods=['POST'])
def prompt_model():
    payload: dict[str, Any] = request.get_json()
    prompt: str = payload.get('prompt', '')
    model_id: str = payload.get('model_id', '')

    conv_id = create_conversation(conn, prompt)
    insert_message(conn, conv_id, 'user', [{'type': 'message', 'content': prompt}])

    chat_gen = ask_model_stream(model_id, prompt)

    def generate_stream():
        elems: List[Dict[str, Any]] = []
        buffered_element_type: Optional[str] = None
        buffered_element_content: Optional[str] = None
        for chunk in chat_gen:
            data: Dict[str, Any] = json.loads(chunk.decode())
            message = data.get('message', {})
            if 'thinking' in message:
                elem_type = 'thinking'
                content = message.get('thinking', '')
            else:
                elem_type = 'message'
                content = message.get('content', '')
            if buffered_element_type is None:
                buffered_element_type = elem_type
                buffered_element_content = ''
            if elem_type != buffered_element_type:
                elems.append({
                    'type': buffered_element_type,
                    'content': buffered_element_content
                })
                buffered_element_content = ''
                buffered_element_type = elem_type
            buffered_element_content += content
            yield chunk
        if buffered_element_content is not None:
            elems.append({
                'type': buffered_element_type,
                'content': buffered_element_content
            })
        insert_message(conn, conv_id, 'assistant', elems)

    return Response(generate_stream(), mimetype='text/plain')

@app.route('/api/conversations')
def get_conversations():
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
        rows = cur.fetchall()
    conversations = [{'id': r[0], 'title': r[1], 'created_at': r[2]} for r in rows]
    return jsonify(conversations)

@app.route('/api/conversation/<int:conv_id>')
def get_conversation(conv_id):
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, created_at FROM conversations WHERE id = %s", (conv_id,))
        row = cur.fetchone()
        if not row:
            return jsonify({'error': 'Not found'}), 404
        cur.execute("SELECT id, role, elements, created_at FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        messages = [{'id': r[0], 'role': r[1], 'elements': r[2], 'created_at': r[3]} for r in cur.fetchall()]
    return jsonify({'id': row[0], 'title': row[1], 'created_at': row[2], 'messages': messages})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    init_database()
    app.run(debug=True)
