from fastapi import BackgroundTasks, FastAPI, Request, UploadFile, File, Body
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import requests
from fastapi.staticfiles import StaticFiles
import psycopg2
from psycopg2.extensions import connection
import os
from pathlib import Path
import glob
import json
import hashlib
import binascii
import subprocess
from typing import Dict, Optional, Any, Generator, NamedTuple, Union, List
from typing import Tuple
import openai
from openai.types.chat.chat_completion_message_param import *
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from datetime import datetime


def get_document_by_path(path: str) -> Optional[Tuple[int, str]]:
    with mk_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, file_hash FROM documents WHERE file_path = %s",
            (path,),
        )
        row = cur.fetchone()
        return row if row else None


def upsert_document(conn: connection, path: str, file_hash: str, content: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM documents WHERE file_path = %s",
            (path,),
        )
        row = cur.fetchone()
        if row:
            doc_id = row[0]
            cur.execute(
                "UPDATE documents SET file_hash = %s, content = %s WHERE id = %s",
                (file_hash, content, doc_id),
            )
        else:
            cur.execute(
                "INSERT INTO documents (file_path, file_hash, content) VALUES (%s, %s, %s) RETURNING id",
                (path, file_hash, content),
            )
            (doc_id,) = cur.fetchone()
    return doc_id


def replace_document_chunks(conn: connection, doc_id: int, chunk_texts: List[str], chunk_embeddings: List[Any]):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
        for index, (embedding, chunk_text) in enumerate(zip(chunk_embeddings, chunk_texts)):
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s)",
                (doc_id, index, chunk_text, embedding),
            )


app = FastAPI(title="MyAI FastAPI")

REPOSITORIES_DIR = os.getenv('REPOSITORIES_DIR', os.path.expanduser('~/.myai/repositories'))
REPOSITORIES_VROOT = "/repositories"
WORKSPACE_DIR = os.getenv('WORKSPACE_DIR', os.path.expanduser('~/.myai/workspace'))
WORKSPACE_VROOT = "/workspace"
os.makedirs(REPOSITORIES_DIR, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)
def is_safe_vpath(vpath: Path, expected_vroot: Path) -> tuple[bool, str]:
    try:
        parts = vpath.parts
        if ".." in parts:
            return False, "Path traversal detected: '..' not allowed"
        if not vpath.is_relative_to(expected_vroot):
            return False, f"Path must be under {expected_vroot}"
        return True, ""
    except Exception as e:
        return False, f"Path validation error: {str(e)}"

def vpath_to_realpath(vpath: Path, vroot: str, base_dir: str) -> Path:
    parts = list(vpath.parts)
    if parts[0] == "/":
        parts = parts[1:]          # usuń leading '/'
    if parts and parts[0] == vroot.lstrip("/"):
        parts = parts[1:]

    # Teraz łączymy z realnym katalogiem repozytorium
    return Path(base_dir) / Path(*parts)


LLAMA_CPP_ENDPOINT = os.getenv('LLAMA_CPP_ENDPOINT', 'http://localhost:1234')
completions_endpoint = openai.Client(api_key='dummy', base_url=LLAMA_CPP_ENDPOINT)

def mk_conn():
    ret = psycopg2.connect(
        dbname='myai',
        user='myai',
        password='myai',
        host='localhost',
        port='5432'
    )
    ret.autocommit = False
    return ret

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
    },
    {
        "name": "run_semantic_search",
        "description": (
            "Search the repositories for the most relevant documents "
            "based on a natural-language prompt."
        ),
        "type": "function",
        "function": {
            "name": "run_semantic_search",
            "description": (
                "Search the repositories for the most relevant documents "
                "based on a natural-language prompt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Natural-language query to search for",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Number of documents to return",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
{
    "name": "propose_replace",
    "description": "Propose replacing a target file in the repositories folder with a source file from the workspace.",
    "type": "function",
    "function": {
        "name": "propose_replace",
        "description": "Replace a file in repositories with a file from workspace. Provide absolute paths: target='/repositories/foo/bar.txt', source='/workspace/baz/qux.txt'. Do not use '..'.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Absolute path to target file in repositories folder (e.g., '/repositories/foo/bar.txt')"},
                "source": {"type": "string", "description": "Absolute path to source file in workspace folder (e.g., '/workspace/baz/qux.txt')"},
            },
            "required": ["target", "source"],
        },
    },
},
{
    "name": "propose_diff",
    "description": "Proposes applying a diff file to a target file in the repositories.",
    "type": "function",
    "function": {
        "name": "propose_diff",
        "description": "Apply a diff from workspace to a target file in repositories. Provide absolute paths: target='/repositories/foo/bar.txt', diff_path='/workspace/patch.diff'. Do not use '..'.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Absolute path to target file in repositories folder (e.g., '/repositories/foo/bar.txt')"},
                "diff_path": {"type": "string", "description": "Absolute path to diff file in workspace folder (e.g., '/workspace/patch.diff')"},
            },
            "required": ["target", "diff_path"],
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

def init_database(conn: connection):
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

    # Run the command inside the sandbox.
    sandbox_result = run_sandboxed_command(command, WORKSPACE_DIR, REPOSITORIES_DIR)
    if 'error' in sandbox_result:
        return ShellResult(command=command, returncode=-1, stdout='', stderr=sandbox_result['error'])
    return ShellResult(
        command=command,
        returncode=sandbox_result.get('exit_code', -1),
        stdout=sandbox_result.get('stdout', ''),
        stderr=sandbox_result.get('stderr', ''),
    )


# ---------------------------------------------------------------------------
# Bubblewrap sandbox execution
# ---------------------------------------------------------------------------
# The following function provides a lightweight sandboxing layer using
# `bubblewrap` (bwrap). It isolates the executed command from the host
# filesystem, exposing only a read‑only view of the repositories repository
# (`reference_path`) and a writable workspace (`workspace_path`). This is
# particularly useful for running untrusted code or scripts that should
# not modify the underlying repositories base.
# ---------------------------------------------------------------------------
def run_sandboxed_command(command: str, workspace_path: str, reference_path: str) -> Dict[str, Any]:
    # Resolve absolute paths for safety
    workspace_path = os.path.abspath(workspace_path)
    reference_path = os.path.abspath(reference_path)

    # Build the bwrap command arguments
    bwrap_args = [
        "bwrap",
        # Mount essential system directories as read‑only
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin",
        "--proc", "/proc",
        "--dev", "/dev",
        # Unshare namespaces for isolation
        "--unshare-all",
        # Writable workspace
        "--bind", workspace_path, WORKSPACE_VROOT,
        # Set working directory inside the sandbox
        "--chdir", "/",
    ]

    for entry in os.scandir(reference_path):
        # bwrap --ro-bind follows symlinks
        bwrap_args.extend(["--ro-bind", entry.path, f"{REPOSITORIES_VROOT}/{entry.name}"])

    # Execute the provided command via bash
    bwrap_args.extend([
        "bash", "-c", command
    ])

    try:
        result = subprocess.run(
            bwrap_args,
            capture_output=True,
            text=True,
            timeout=30,  # safety timeout
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out"}
    except Exception as e:
        return {"error": str(e)}

def run_propose_replace(tool_call: ToolCall) -> ToolCallResult:
    """Proposes replacing a target file with a source file."""
    try:
        params = json.loads(tool_call.parameters)
        target_vpath_str = params.get("target")
        source_vpath_str = params.get("source")
        
        if not target_vpath_str or not source_vpath_str:
            raise ValueError("Missing target or source path")
        
        target_vpath = Path(target_vpath_str)
        source_vpath = Path(source_vpath_str)
        
        target_safe, target_err = is_safe_vpath(target_vpath, Path(REPOSITORIES_VROOT))
        if not target_safe:
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": target_err}))
        
        source_safe, source_err = is_safe_vpath(source_vpath, Path(WORKSPACE_VROOT))
        if not source_safe:
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": source_err}))
        
        target_realpath = vpath_to_realpath(target_vpath, REPOSITORIES_VROOT, REPOSITORIES_DIR)
        source_realpath = vpath_to_realpath(source_vpath, WORKSPACE_VROOT, WORKSPACE_DIR)
        
        if not source_realpath.exists():
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": f"Source file not found: {source_vpath}"}))
        
        if target_realpath.exists():
            target_content = target_realpath.read_text(encoding='utf-8')
        else:
            target_content = ""
        
        source_content = source_realpath.read_text(encoding='utf-8')
        
        proposal = {
            "type": "replace_proposal",
            "target": str(target_vpath),
            "source": str(source_vpath),
            "preview": f"Replace {target_vpath} with content from {source_vpath}",
            "target_sample": target_content[:200] + ("..." if len(target_content) > 200 else ""),
            "source_sample": source_content[:200] + ("..." if len(source_content) > 200 else ""),
            "status": "pending"
        }
        return ToolCallResult(tool_call.name, tool_call.parameters, json.dumps(proposal))
    except json.JSONDecodeError as e:
        return ToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": f"Invalid JSON: {str(e)}"}))
    except Exception as e:
        return ToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": f"Unexpected error: {str(e)}"}))

def run_propose_diff(tool_call: ToolCall) -> ToolCallResult:
    """Proposes applying a diff to a target file."""
    # Analogiczna logika jak w propose_replace
    try:
        params = json.loads(tool_call.parameters)
        target_vpath_str = params.get("target")
        diff_vpath_str = params.get("diff_path")
        
        if not target_vpath_str or not diff_vpath_str:
            raise ValueError("Missing target or diff_path")
        
        target_vpath = Path(target_vpath_str)
        diff_vpath = Path(diff_vpath_str)
        
        target_safe, target_err = is_safe_vpath(target_vpath, Path(REPOSITORIES_VROOT))
        if not target_safe:
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": target_err}))
        
        diff_safe, diff_err = is_safe_vpath(diff_vpath, Path(WORKSPACE_VROOT))
        if not diff_safe:
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": diff_err}))
        
        target_realpath = vpath_to_realpath(target_vpath, REPOSITORIES_VROOT, REPOSITORIES_DIR)
        diff_realpath = vpath_to_realpath(diff_vpath, WORKSPACE_VROOT, WORKSPACE_DIR)
        
        if not target_realpath.exists():
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": f"Target file not found: {target_vpath}"}))
        
        if not diff_realpath.exists():
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": f"Diff file not found: {diff_vpath}"}))
        
        diff_content = diff_realpath.read_text(encoding='utf-8')
        
        proposal = {
            "type": "diff_proposal",
            "target": str(target_vpath),
            "diff": str(diff_vpath),
            "preview": f"Apply diff {diff_vpath} to {target_vpath}",
            "diff_preview": diff_content[:300] + ("..." if len(diff_content) > 300 else ""),
            "status": "pending"
        }
        return ToolCallResult(tool_call.name, tool_call.parameters, json.dumps(proposal))
    except json.JSONDecodeError as e:
        return ToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": f"Invalid JSON: {str(e)}"}))
    except Exception as e:
        return ToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": f"Unexpected error: {str(e)}"}))

def run_semantic_search(tool_call: ToolCall) -> ToolCallResult:
    try:
        params = json.loads(tool_call.parameters)
        prompt: str = params["prompt"]
        top_k: int = int(params.get("top_k", 5))
    except Exception as exc:
        return ToolCallResult(
            name=tool_call.name,
            parameters=tool_call.parameters,
            result=json.dumps({"error": f"bad parameters: {exc}"})
        )

    try:
        results = semantic_search(prompt, top_k)
    except Exception as exc:
        return ToolCallResult(
            name=tool_call.name,
            parameters=tool_call.parameters,
            result=json.dumps({"error": f"search failed: {exc}"})
        )
    return ToolCallResult(
        name=tool_call.name,
        parameters=tool_call.parameters,
        result=json.dumps({"results": json.dumps(results)})
    )

def run_tool_call(call: ToolCall) -> ToolCallResult:
    if call.name == 'run_shell_command':
        shell = run_shell_command(call)
        output_str = json.dumps({'command': shell.command, 'returncode': shell.returncode, 'stdout': shell.stdout, 'stderr': shell.stderr})
        return ToolCallResult(call.name, call.parameters, output_str)
    elif call.name == 'run_semantic_search':
        return run_semantic_search(call)
    elif call.name == 'propose_replace':
        return run_propose_replace(call)
    elif call.name == 'propose_diff':
        return run_propose_diff(call)
    else:
        raise ValueError(f'Unsupported tool call {call.name}')

@app.post('/api/prompt')
async def prompt_model(request: Request):
    payload: dict[str, Any] = await request.json()
    prompt: str = payload.get('prompt', '')
    model_id: Optional[str] = payload.get('model_id')
    if model_id is None:
        raise ValueError('Unknown model')
    conversation_id: Optional[int] = payload.get('conversation_id')

    with mk_conn() as conn:
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
                reasoning_effort='high',
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
                        yield json.dumps({'type': 'response', 'content': content}) + '\n'
                    case Thinking(content=content):
                        yield json.dumps({'type': 'reasoning', 'content': content}) + '\n'
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
                        result_json = to_json_dict(result)
                        yield json.dumps(result_json) + '\n'
                        msg_id = insert_message(conn, conversation_id, 'assistant', result_json)
                        oai_elems = to_oai_completions_elements(msg_id, 'assistant', result_json)
                        messages_context.extend(oai_elems)
                        # Loop back for tool call
                        run_next_loop = True
        print('Stream finished')

    stream = StreamingResponse(generate_stream(), media_type='application/x-ndjson')
    stream.headers['X-Conversation-ID'] = str(conversation_id)
    return stream

@app.get('/api/conversations')
async def get_conversations():
    with mk_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
        rows = cur.fetchall()
    conversations = [{'id': r[0], 'title': r[1], 'created_at': r[2].isoformat()} for r in rows]
    return JSONResponse(content=conversations)

@app.get('/api/conversation/{conv_id}')
async def get_conversation(conv_id: int):
    with mk_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, title, created_at FROM conversations WHERE id = %s", (conv_id,))
        row = cur.fetchone()
        if not row:
            return JSONResponse(status_code=404, content={'error': 'Not found'})
        cur.execute("SELECT id, role, elements, created_at FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        messages = [{'id': r[0], 'role': r[1], 'elements': r[2], 'created_at': r[3].isoformat()} for r in cur.fetchall()]
        # Fetch messages for the conversation
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

class SearchHit(NamedTuple):
    document_id: int
    file_path: str
    chunk_index: int
    score: float
    text: str

def semantic_search(query: str, top_k: int) -> list[SearchHit]:
    embedding_resp = embeddings_endpoint.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_vec = embedding_resp.data[0].embedding
    with mk_conn() as conn, conn.cursor() as cur:
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
            results.append(SearchHit(doc_id, file_path, chunk_index, score, content))
        return results

@app.post('/api/search')
async def search(payload: dict = Body(...)):
    query_text: str = payload.get('query', '')
    top_k: int = int(payload.get('top_k', 3))
    if not query_text:
        return JSONResponse(status_code=400, content={'error': 'query required'})
    results = []
    for hit in semantic_search(query_text, top_k):
        snippet = hit.text[:200] + ('...' if len(hit.text) > 200 else '')
        results.append({'id': hit.document_id, 'title': f"{hit.file_path} ({hit.chunk_index})", 'snippet': snippet, 'score': hit.score})
    return JSONResponse(content={'results': results})

def process_folder():
    """Incrementally sync repositories folder.

    For each file, compute its hash and compare it against the stored value.
    Only files that are new or changed are processed.
    """
    for repo_dir in os.listdir(REPOSITORIES_DIR):
        repo_full_path = str(Path(REPOSITORIES_DIR) / repo_dir)
        is_git_result = subprocess.run(
            [
                "git", "-C", repo_full_path, "rev-parse", "--is-inside-work-tree"
            ],
            capture_output=True,
            text=True
        )
        is_git = is_git_result.returncode == 0

        if is_git:
            ls_git_result = subprocess.run(
                [
                    "git", "-C", repo_full_path, "ls-files" 
                ],
                capture_output=True,
                text=True
            )
            if ls_git_result.returncode != 0:
                print(f"[ERROR]: Couldn't list directories of Git repo {repo_dir} at {repo_full_path}: {ls_git_result.stderr}")
                continue
            files = ls_git_result.stdout.splitlines()
        else:
            files: list[str] = []
            for root, _, fnames in os.walk(repo_full_path):
                for fname in fnames:
                    f = Path(root) / fname
                    f = f.relative_to(repo_full_path)
                    files.append(f)

        for fname in files:
            if Path(fname).suffix.lower() not in {
                ".properties",
                ".md", ".txt",
                ".json", ".xml", ".csv", ".yml", ".yaml", 
                ".mill", ".java", 
                ".c", ".cpp", 
                ".html", ".hbs", ".js", ".ts",
                ".py", ".sh",
                ".sql",
            }:
                print(f"Skipping unhandled file type {fname}")
                continue

            full_path = Path(repo_full_path) / fname
            relative_path = str(full_path.relative_to(REPOSITORIES_DIR))
            try:
                file_bytes = full_path.read_bytes()
                file_text = file_bytes.decode()
            except Exception as exc:
                print(f"[WARN] Unreadable file {relative_path}: {exc}")
                continue
            
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            existing = get_document_by_path(relative_path)
            if existing and existing[1] == file_hash:
                print(f"[SKIP] {relative_path} unchanged")
                continue

            token_ids, chunk_texts = get_token_chunks(file_text)
            embedding_resp = embeddings_endpoint.embeddings.create(
                model=EMBEDDING_MODEL,
                input=token_ids
            )
            assert len(embedding_resp.data) == len(chunk_texts)

            with mk_conn() as conn:
                doc_id = upsert_document(conn, relative_path, file_hash, file_text)
                replace_document_chunks(conn, doc_id, chunk_texts, [e.embedding for e in embedding_resp.data])
                print(f"[OK] {relative_path} - {len(embedding_resp.data)} embeddings")

@app.post("/api/sync")
async def sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_folder)
    return JSONResponse({"status": "sync started"})

@app.on_event("startup")
async def startup_event():
    with mk_conn() as conn:
        init_database(conn)
    process_folder()

# Mount static files
app.mount("/", StaticFiles(directory="static"), name="static")
