from flask import Flask, jsonify, request, Response
import psycopg2
from psycopg2.extensions import connection
import os
import glob
import json
import hashlib
import binascii
from typing import List, Dict, Optional, Any, Generator
import openai
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from datetime import datetime


app = Flask(__name__)
LLAMA_CPP_ENDPOINT = os.getenv('LLAMA_CPP_ENDPOINT', 'http://localhost:8080')
client = openai.Client(api_key='dummy', base_url=LLAMA_CPP_ENDPOINT)

conn = psycopg2.connect(
    dbname='myai',
    user='myai',
    password='myai',
    host='localhost',
    port='5432'
)

def create_conversation(conn: connection, title: str) -> int:
    """Insert a new conversation row with title and return its id."""
    with conn.cursor() as cur:
        cur.execute("INSERT INTO conversations (title) VALUES (%s) RETURNING id", (title,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to create conversation")
        conv_id = row[0]
        conn.commit()
        return conv_id
    

def insert_message(conn: connection, conv_id: int, role: str, elements: List[Dict[str, str]]) -> None:
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

def get_messages_for_continuation(conn: connection, conv_id: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    with conn.cursor() as cur:
        cur.execute("SELECT role, elements FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        for row in cur.fetchall():
            role, elements = row
            
            thinking = None
            message = None

            for element in elements:
                elem_type = element.get('type') if isinstance(element, dict) else element.type
                elem_content = element.get('content') if isinstance(element, dict) else element.content
                if elem_type == 'thinking':
                    thinking = elem_content
                elif elem_type == 'message':
                    message = elem_content
                else:
                    raise ValueError(f'Unhandled element type {elem_type}')
            
            msg = {'role': role}
            if thinking is not None:
                msg['thinking'] = thinking
            if message is not None:
                msg['content'] = message 
            messages.append(msg)
    
    return messages

def ask_model_stream(model_id: str, prompt_text: str, conversation_context: Optional[list[dict[str, Any]]] = None) -> Generator[openai.types.Completion, None, None]:

    if conversation_context:
        messages = list(conversation_context)
    else:
        messages = []
    messages.append({'role': 'user', 'content': prompt_text})
    
    res = client.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=True,
    )
    for chunk in res:
        yield chunk

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
        res = client.models.list()
        models = list(res)
    except Exception as e:
        print(f"Error fetching models: {e}")
        models = []
    return jsonify({"models": [{'id': m.id, 'name': m.id} for m in models]})

@app.route('/api/prompt', methods=['POST'])
def prompt_model():
    payload: dict[str, Any] = request.get_json()
    prompt: str = payload.get('prompt', '')
    model_id: str = payload.get('model_id', '')
    conversation_id: Optional[int] = payload.get('conversation_id')

    if conversation_id:
        messages_context = get_messages_for_continuation(conn, conversation_id)
        chat_gen = ask_model_stream(model_id, prompt, messages_context)
        conv_id = conversation_id
    else:
        conv_id = create_conversation(conn, prompt)
        chat_gen = ask_model_stream(model_id, prompt)

    insert_message(conn, conv_id, 'user', [{'type': 'message', 'content': prompt}])

    def generate_stream():
        elems: List[Dict[str, Any]] = []
        buffered_element_type: Optional[str] = None
        buffered_element_content: Optional[str] = None
        for chunk in chat_gen:
            delta: ChoiceDelta = chunk.choices[0].delta
            delta_dict = delta.model_dump()
            if 'reasoning_content' in delta_dict:
                elem_type = 'thinking'
                content = delta.reasoning_content
            elif delta.content is not None:
                elem_type = 'message'
                content = delta.content
            else:
                print('Skipping empty delta', delta)
                continue
            print(delta)

            if buffered_element_type is None:
                buffered_element_type = elem_type
                buffered_element_content = ''
            if elem_type != buffered_element_type:
                elems.append({
                    'type': buffered_element_type,
                    'content': buffered_element_content
                })
                print(f'Finished {buffered_element_type} {buffered_element_content}')
                buffered_element_content = ''
                buffered_element_type = elem_type
            buffered_element_content += content
            message = {}
            if buffered_element_type == 'message':
                message['content'] = content
            elif buffered_element_type == 'thinking':
                message['thinking'] = content
            yield json.dumps({'message': message})
        if buffered_element_content is not None:
            elems.append({
                'type': buffered_element_type,
                'content': buffered_element_content
            })
        insert_message(conn, conv_id, 'assistant', elems)



    response = Response(generate_stream(), mimetype='text/plain')
    response.headers['X-Conversation-ID'] = str(conv_id)
    return response

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
