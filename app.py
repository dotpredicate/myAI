# Convert from Flask to FastAPI
from fastapi import BackgroundTasks, FastAPI, Request, UploadFile, File, Body
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import uvicorn
import requests
from fastapi.staticfiles import StaticFiles
import psycopg2
from psycopg2.extensions import connection
from pgvector.psycopg2 import register_vector
import os
import pathlib
import glob
import json
import hashlib
import binascii
import subprocess
from typing import Dict, Optional, Any, Generator, NamedTuple, Union, List
import openai
from openai.types.chat.chat_completion_message_param import *
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from datetime import datetime


app = FastAPI(title="MyAI FastAPI")

KNOWLEDGE_FOLDER = os.getenv('KNOWLEDGE_FOLDER', './knowledge')
os.makedirs(KNOWLEDGE_FOLDER, exist_ok=True)

# Endpoint for completions
LLAMA_CPP_ENDPOINT = os.getenv('LLAMA_CPP_ENDPOINT', 'http://localhost:1234')
completions_endpoint = openai.Client(api_key='dummy', base_url=LLAMA_CPP_ENDPOINT)

conn = psycopg2.connect(
    dbname='myai',
    user='myai',
    password='myai',
    host='localhost',
    port='5432'
)

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Define the function schema for OpenAI function calling
FUNCTIONS = [
    {
        "name": "run_shell_command",
        "description": "Execute a shell command and return the output.",
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Execute a shell command and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                },
                "required": ["command"],
            },
        },
    }
]

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
    

def insert_message(conn: connection, conv_id: int, role: str, element: Dict[str, str]) -> int:
    """Persist a full array of elements for a message."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages (conversation_id, role, elements, created_at)
            VALUES (%s, %s, %s, NOW())
            RETURNING id
            """,
            (conv_id, role, json.dumps(element))
        )
        (id,) = cur.fetchone()
        conn.commit()
    print(f'Inserted new message {id}')
    return id

def get_messages_for_continuation(conn: connection, conv_id: int) -> list[dict[str, any]]:
    messages = []

    with conn.cursor() as cur:
        cur.execute("SELECT id, role, elements FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        for row in cur.fetchall():
            message_id, role, element = row
            
            new_messages = to_oai_completions_elements(message_id, role, element)
            messages.extend(new_messages)
    return messages

def ask_model_stream(model_id: str, prompt_text: str, conversation_context: Optional[list[dict[str, Any]]] = None) -> Generator[ChatCompletionChunk, None, None]:

    if conversation_context:
        messages = list(conversation_context)
    else:
        messages = []
    messages.append({'role': 'user', 'content': prompt_text})
    res = completions_endpoint.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=True,
        tools=FUNCTIONS,
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
    print('Migrations completed')

@app.get('/api/models')
async def get_models():
    try:
        res = completions_endpoint.models.list()
        models = list(res)
    except Exception as e:
        print(f"Error fetching models: {e}")
        models = []
    return JSONResponse(content={"models": [{'id': m.id, 'name': m.id} for m in models]})

class ShellResult(NamedTuple):
    command: str
    returncode: int
    stdout: str
    stderr: str

class Message(NamedTuple):
    content: str
class Thinking(NamedTuple):
    content: str
class ToolCall(NamedTuple):
    name: str
    parameters: str
StreamingElement = Union[Thinking, Message, ToolCall]

class ToolCallResult(NamedTuple):
    name: str
    parameters: str
    result: str
FinishedElement = Union[Thinking, Message, ToolCallResult]

def to_json_dict(element: FinishedElement) -> dict[str, Any]:
    match element:
        case Message(content=content):
            return {'type': 'message', 'content': content}
        case Thinking(content=content):
            return {'type': 'thinking', 'content': content}
        case ToolCallResult(name=name, parameters=parameters, result=result):
            return {'type': 'tool_call', 'name': name, 'parameters': parameters, 'result': result}

def to_oai_completions_elements(message_id: int, role: str, myai_dict: dict[str, any]) -> list[dict[str, Any]]:
    messages = []
    elem_type = myai_dict['type']
    if elem_type == 'thinking':
        # TODO: Pass thinking back to the model
        # msg = {'role': role}               
        # msg['thinking'] = element['content']
        # messages.append(msg)
        pass
    elif elem_type == 'message':
        msg = {'role': role}               
        msg['content'] = myai_dict['content'] 
        messages.append(msg)
    elif elem_type == 'tool_call':
        tool_call = {'role': 'assistant'}
        tool_call['content'] = ''
        tool_call['tool_calls'] = [{
            'id': str(message_id),
            'type': 'function',
            'function': {
                'name': myai_dict['name'],
                'arguments': myai_dict['parameters'],
            },
        }]
        tool_call_result = {
            'role': 'tool',
            'tool_call_id': str(message_id),
            'content': myai_dict['result']
        }
        messages.append(tool_call)
        messages.append(tool_call_result)
        print(messages)
    else:
        raise ValueError(f'Unhandled element type {elem_type}')
    return messages

class DeltaProcessor:
    def __init__(self):
        self.buffered_element: Optional[StreamingElement] = None

    def process(self, chunk: ChatCompletionChunk) -> tuple[Optional[StreamingElement], Optional[StreamingElement]]:
        delta = chunk.choices[0].delta
        delta_dict = delta.model_dump()
        # Determine element type and content
        new_elem: Optional[StreamingElement]
        if 'tool_calls' in delta_dict and delta.tool_calls is not None:
            func = delta.tool_calls[0].function
            new_elem = ToolCall(func.name, func.arguments)
        elif 'reasoning_content' in delta_dict:
            new_elem = Thinking(delta_dict['reasoning_content'])
        elif delta.content is not None:
            new_elem = Message(delta.content)
        else:
            print(f'Unknown delta {delta}')
            new_elem = None

        finalized_element: Optional[StreamingElement] = None
        match (self.buffered_element, new_elem):
            case (None, elem):
                self.buffered_element = elem
            case (Message(content=c1), Message(content=c2)):
                self.buffered_element = Message(c1 + c2)
            case (Thinking(content=c1), Thinking(content=c2)):
                self.buffered_element = Thinking(c1 + c2)
            case (ToolCall(name=name, parameters=p1), ToolCall(parameters=p2)):
                self.buffered_element = ToolCall(name, p1 + p2)
            case _:
                finalized_element = self.buffered_element
                self.buffered_element = new_elem
        return (new_elem, finalized_element)

def run_shell_command(tool_call: ToolCall) -> ShellResult:
    try:
        params = json.loads(tool_call.parameters)
        command: str = params['command']
        if not isinstance(command, str):
            raise ValueError('Cannot parse the command field. Double-check if input is valid JSON.')
    except Exception as e:
        return ShellResult(
            command='',
            returncode=-1,
            stdout='',
            stderr=str(e),
        )
    completed = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        executable='/bin/bash',
    )
    return ShellResult(command=command, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def run_tool_call(call: ToolCall) -> ToolCallResult:
    if call.name != 'run_shell_command':
        raise ValueError('Only shell commands supported for now')
    shell = run_shell_command(call)
    output_str = json.dumps({'command': shell.command, 'returncode': shell.returncode, 'stdout': shell.stdout, 'stderr': shell.stderr})
    return ToolCallResult(call.name, call.parameters, output_str)

@app.post('/api/prompt')
async def prompt_model(request: Request):
    payload: dict[str, Any] = await request.json()
    prompt: str = payload.get('prompt', '')
    model_id: Optional[str] = payload.get('model_id')
    if model_id is None:
        raise ValueError('Unknown model')
    conversation_id: Optional[int] = payload.get('conversation_id')

    if conversation_id:
        messages_context = get_messages_for_continuation(conn, conversation_id)
    else:
        conversation_id = create_conversation(conn, prompt)
        messages_context = []

    user_message = Message(content=prompt)
    user_message_dict = to_json_dict(user_message)
    user_message_id = insert_message(conn, conversation_id, 'user', user_message_dict)
    user_message_oai_dict = to_oai_completions_elements(user_message_id, 'user', user_message_dict)

    messages_context.extend(user_message_oai_dict)


    def generate_stream():
        run_next_loop = True
        while run_next_loop:
            processor = DeltaProcessor()
            print(f'{model_id=}, {messages_context=}')
            chat_gen_inner = completions_endpoint.chat.completions.create(
                model=model_id,
                messages=messages_context,
                stream=True,
                tools=FUNCTIONS,
            )
            run_next_loop = False
            for chunk in chat_gen_inner:
                (delta, aggregated_element) = processor.process(chunk)
                if delta is not None:
                    print(f'Processed new delta: {delta}')
                # Stream assistant content.
                match delta:
                    case Message(content=content):
                        yield json.dumps({'type': 'response', 'content': content})
                    case Thinking(content=content):
                        yield json.dumps({'type': 'reasoning', 'content': content})
                    case _:
                        print(f'Not broadcasting {delta}')

                if aggregated_element is not None:
                    print(f'Aggregated new element: {aggregated_element}')
                match aggregated_element:
                    case Message() as message:
                        result_json = to_json_dict(message)
                        msg_id = insert_message(conn, conversation_id, 'assistant', result_json)
                        oai_elems = to_oai_completions_elements(msg_id, 'assistant', result_json)
                        messages_context.extend(oai_elems)
                    case Thinking() as thinking:
                        result_json = to_json_dict(thinking)
                        msg_id = insert_message(conn, conversation_id, 'assistant', result_json)
                        oai_elems = to_oai_completions_elements(msg_id, 'assistant', result_json)
                        messages_context.extend(oai_elems)
                    case ToolCall() as call:
                        result = run_tool_call(call)
                        print(f'Tool call finished: {result}')
                        result_json = to_json_dict(result)
                        yield json.dumps(result_json)
                        msg_id = insert_message(conn, conversation_id, 'assistant', result_json)
                        oai_elems = to_oai_completions_elements(msg_id, 'assistant', result_json)
                        messages_context.extend(oai_elems)
                        # Loop back for tool call
                        run_next_loop = True

    stream = StreamingResponse(generate_stream(), media_type='text/plain')
    stream.headers['X-Conversation-ID'] = str(conversation_id)
    return stream

@app.get('/api/conversations')
async def get_conversations():
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
        rows = cur.fetchall()
    conversations = [{'id': r[0], 'title': r[1], 'created_at': r[2].isoformat()} for r in rows]
    return JSONResponse(content=conversations)

@app.get('/api/conversation/{conv_id}')
async def get_conversation(conv_id: int):
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, created_at FROM conversations WHERE id = %s", (conv_id,))
        row = cur.fetchone()
    if not row:
        return JSONResponse(status_code=404, content={'error': 'Not found'})
    # Fetch messages for the conversation
    with conn.cursor() as cur:
        cur.execute("SELECT id, role, elements, created_at FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        messages = [{'id': r[0], 'role': r[1], 'elements': r[2], 'created_at': r[3].isoformat()} for r in cur.fetchall()]
    return JSONResponse(content={'id': row[0], 'title': row[1], 'created_at': row[2].isoformat(), 'messages': messages})


EMBEDDING_MODEL = 'unsloth/embeddinggemma-300m-GGUF'
# Endpoint for completions
LLAMA_CPP_EMBEDDINGS_URL = os.getenv('LLAMA_CPP_ENDPOINT', 'http://localhost:2345')
embeddings_endpoint = openai.Client(api_key='dummy', base_url=LLAMA_CPP_EMBEDDINGS_URL + '/v1')

def get_token_chunks(text, max_tokens=400, overlap=50) -> tuple[list[list[int]], list[str]]:
    response = requests.post(
        f"{LLAMA_CPP_EMBEDDINGS_URL}/tokenize",
        json={"content": text, "with_pieces": True}
    ).json()

    all_tokens = response["tokens"]

    token_ids: list[list[int]] = []
    text_chunks: list[str] = []

    for i in range(0, len(all_tokens), max_tokens - overlap):
        chunk = all_tokens[i : i + max_tokens]
        chunk_tokens = []
        chunk_text = ''
        for piece in chunk:
            chunk_tokens.append(piece["id"])
            piece_content = piece["piece"]
            if isinstance(piece_content, str):
                chunk_text += piece_content
            elif isinstance(piece_content, list):
                chunk_text += bytes(piece_content).decode('utf-8', errors='replace')

        token_ids.append(chunk_tokens)
        text_chunks.append(chunk_text)
    assert len(token_ids) == len(text_chunks)
    return token_ids, text_chunks

# Search endpoint – returns top_k relevant documents
@app.post('/api/search')
async def search(payload: dict = Body(...)):
    query_text: str = payload.get('query', '')
    top_k: int = int(payload.get('top_k', 3))
    if not query_text:
        return JSONResponse(status_code=400, content={'error': 'query required'})
    # Embed query
    embedding_resp = embeddings_endpoint.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query_text
    )
    query_vec = embedding_resp.data[0].embedding
    with conn.cursor() as cur:
        cur.execute(
            """SELECT document_id, file_path, chunk_index, chunk_text, embedding <=> %s::vector AS score 
            FROM document_chunks
            JOIN documents ON documents.id=document_chunks.document_id
            ORDER BY score ASC LIMIT %s""",
            (query_vec, top_k)
        )
        rows = cur.fetchall()
        results = []
        for row in rows:
            doc_id, file_path, chunk_index, content, score = row
            snippet = content[:200] + ('...' if len(content) > 200 else '')
            results.append({'id': doc_id, 'title': f"{file_path} ({chunk_index})", 'snippet': snippet, 'score': score})
    return JSONResponse(content={'results': results})

def process_folder():
    for root, _, files in os.walk(KNOWLEDGE_FOLDER):
        for fname in files:
            if pathlib.Path(fname).suffix.lower() not in {
                ".md", ".txt", ".java", ".py", ".c", ".cpp", ".js", ".ts"
            }:
                print(f"Skipping unhandled file type {fname}")
                continue

            full_path = pathlib.Path(root) / fname
            try:
                full_text = full_path.read_text(encoding="utf-8")
            except Exception as exc:
                print(f"[WARN] Unreadable file {full_path}: {exc}")
                continue

            file_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
            token_ids, chunk_texts = get_token_chunks(full_text)
            embedding_resp = embeddings_endpoint.embeddings.create(
                model=EMBEDDING_MODEL,
                input=token_ids
            )
            print(f'{fname} - got {len(embedding_resp.data)} embeddings')
            assert len(embedding_resp.data) == len(chunk_texts)

            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE file_path=%s", (fname,))
                cur.execute(
                    "INSERT INTO documents (file_path, file_hash, content) VALUES (%s, %s, %s) RETURNING id",
                    (fname, file_hash, full_text)
                )
                (doc_id,) = cur.fetchone()
                for index, (embedding, chunk_text) in enumerate(zip(embedding_resp.data, chunk_texts)):
                    cur.execute(
                        "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s)",
                        (doc_id, index, chunk_text, embedding.embedding)
                    )
                conn.commit()
            print(f"[OK] {full_path}")

@app.post("/api/sync")
async def sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_folder)
    return JSONResponse({"status": "sync started"})

@app.on_event("startup")
async def startup_event():
    init_database()
    process_folder()

# Mount static files
app.mount("/", StaticFiles(directory="static"), name="static")
