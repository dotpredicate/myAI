from flask import Flask, jsonify, request, Response
import psycopg2
from psycopg2.extensions import connection
import os
import glob
import json
import hashlib
import binascii
import subprocess
from typing import List, Dict, Optional, Any, Generator, NamedTuple, Union
import openai
from openai.types.chat.chat_completion_message_param import *
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from datetime import datetime


app = Flask(__name__)
LLAMA_CPP_ENDPOINT = os.getenv('LLAMA_CPP_ENDPOINT', 'http://localhost:1234')
client = openai.Client(api_key='dummy', base_url=LLAMA_CPP_ENDPOINT)

conn = psycopg2.connect(
    dbname='myai',
    user='myai',
    password='myai',
    host='localhost',
    port='5432'
)

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

def get_messages_for_continuation(conn: connection, conv_id: int) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []

    with conn.cursor() as cur:
        cur.execute("SELECT id, role, elements FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        for row in cur.fetchall():
            message_id, role, elements = row
            
            for element in elements:
                elem_type = element['type']
                if elem_type == 'thinking':
                    msg = {'role': role}               
                    msg['thinking'] = element['content']
                    messages.append(msg)
                elif elem_type == 'message':
                    msg = {'role': role}               
                    msg['content'] = element['content'] 
                    messages.append(msg)
                elif elem_type == 'tool_call':
                    tool_call = {'role': 'assistant'}
                    tool_call['tool_calls'] = [{
                        'id': str(message_id),
                        'type': 'function',
                        'function': {
                            'name': element['name'],
                            'arguments': element['parameters'],
                        },
                    }]
                    tool_call_result = {
                        'role': 'tool',
                        'tool_call_id': str(message_id),
                        'content': element['result']
                    }
                    messages.append(tool_call)
                    messages.append(tool_call_result)
                    print(messages)
                else:
                    raise ValueError(f'Unhandled element type {elem_type}')
    
    return messages

def ask_model_stream(model_id: str, prompt_text: str, conversation_context: Optional[list[dict[str, Any]]] = None) -> Generator[ChatCompletionChunk, None, None]:

    if conversation_context:
        messages = list(conversation_context)
    else:
        messages = []
    messages.append({'role': 'user', 'content': prompt_text})
    res = client.chat.completions.create(
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

@app.route('/api/models')
def get_models():
    try:
        res = client.models.list()
        models = list(res)
    except Exception as e:
        print(f"Error fetching models: {e}")
        models = []
    return jsonify({"models": [{'id': m.id, 'name': m.id} for m in models]})



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
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            executable='/bin/bash',
        )
        return ShellResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except Exception as e:
        return ShellResult(command=command, returncode=-1, stdout="", stderr=str(e))


def run_tool_call(call: ToolCall) -> ToolCallResult:
    if call.name != 'run_shell_command':
        raise ValueError('Only shell commands supported for now')
    shell = run_shell_command(call)
    output_str = json.dumps({'command': shell.command, 'returncode': shell.returncode, 'stdout': shell.stdout, 'stderr': shell.stderr})
    return ToolCallResult(call.name, call.parameters, output_str)

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
        processor = DeltaProcessor()
        persisted_elements: List[dict[str, Any]] = []
        for chunk in chat_gen:
            (delta, aggregated_element) = processor.process(chunk)
            if delta is not None:
                print(f'Processed new delta: {delta}')
            match delta:
                case Message(content=content):
                    yield json.dumps({'message': {'content': content}})
                case Thinking(content=content):
                    yield json.dumps({'message': {'thinking': content}})
                case _:
                    print(f'Not broadcasting {delta}')
            if aggregated_element is not None:
                print(f'Aggregated new element: {aggregated_element}')
            match aggregated_element:
                case Message() as message:
                    result_json = to_json_dict(message)
                    persisted_elements.append(result_json)
                case Thinking() as thinking:
                    result_json = to_json_dict(thinking)
                    persisted_elements.append(result_json)
                case ToolCall() as call:
                    result = run_tool_call(call)
                    print(f'Tool call finished: {result}')
                    result_json = to_json_dict(result)
                    yield json.dumps(result_json)
                    persisted_elements.append(result_json)

        # Flush any remaining buffered content
        # for e in final_elements:
        #     aggr_elems.append(e)
        insert_message(conn, conv_id, 'assistant', persisted_elements)

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
