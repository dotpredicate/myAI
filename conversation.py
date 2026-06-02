from dataclasses import dataclass
from typing import Dict, Literal, Optional, Any, Generator, List, TypedDict, Union
import json
from psycopg2.extensions import connection
from inference.engine import (
    StreamingMessage,
    StreamingThinking,
    FinishedMessage,
    FinishedThinking,
    FinishedToolCall,
    FinishedToolCallResult,
    FinishedElement,
)

from inference.llama_cpp_server import (
    DeltaProcessor,
    ChatContext,
    run_chat_completion_stream
)

class ConversationBlockedError(Exception):
    def __init__(self, blocking_message_id: int):
        self.blocking_message_id = blocking_message_id

# @dataclass
# class Message(frozen=True):
#     pass

class StoredMessage(TypedDict):
    type: Literal['message']
    author: Literal['user'] | Literal['assistant']
    content: str
    scopes: list[str]

class StoredThinking(TypedDict):
    type: Literal['thinking']
    content: str

# FIXME: Split into tool call and block
class StoredToolCallResultOrBlock(TypedDict):
    type: Literal['tool_call']
    name: str
    parameters: str
    result: str
    is_blocking: bool
    status: Literal['pending'] | Literal['completed']
    

class StoredToolCallResult(TypedDict):
    type: Literal['tool_call_result']
    original_message_id: int
    result: str

class StoredToolCallDecision(TypedDict):
    type: Literal['tool_call_decision']
    original_message_id: int
    decision: Literal['approve'] | Literal['reject']
    comment: Optional[str]

StoredElement = Union[StoredMessage, StoredThinking, StoredToolCallResultOrBlock, StoredToolCallResult, StoredToolCallDecision]

@dataclass(frozen=True)
class PersistedConversationElement:
    id: int
    msg: StoredElement


def to_stored_elem(element: FinishedElement) -> StoredElement:
    match element:
        case FinishedMessage(content=content):
            return {'type': 'message', 'author': 'assistant', 'content': content, 'scopes': []}
        case FinishedThinking(content=content):
            return {'type': 'thinking', 'content': content}
        case FinishedToolCallResult(name=name, parameters=parameters, result=result, is_blocking=is_blocking):
            return {
                'type': 'tool_call',
                'name': name,
                'parameters': parameters,
                'result': result,
                'is_blocking': is_blocking,
                'status': 'pending' if is_blocking else 'completed'
            }
        case _:
            raise ValueError(f'Unhandled tool call type {type(element)}')

def create_conversation(conn: connection) -> int:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO conversations DEFAULT VALUES RETURNING id")
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to create conversation")
        conv_id = row[0]
        conn.commit()
        return conv_id

def insert_message(conn: connection, conv_id: int, role: str, element: StoredElement) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages (conversation_id, role, elements, created_at)
            VALUES (%s, %s, %s, NOW())
            RETURNING id
            """,
            (conv_id, role, json.dumps(element))
        )
        row = cur.fetchone()
        assert row is not None
        return row[0]

def get_blocking_message_id(conn: connection, conv_id: int) -> Optional[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT blocking_message_id FROM conversations WHERE id = %s",
            (conv_id,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

def get_scopes_from_last_user_message(conn: connection, conv_id: int) -> List[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT elements FROM messages WHERE conversation_id = %s AND role = 'user' ORDER BY created_at DESC LIMIT 1", (conv_id,))
        row = cur.fetchone()
        if row and (elem := row[0]) and elem['type'] == 'message' and elem['author'] == 'user':
            return elem['scopes']
        else:
            return []

def prepare_conversation_with_prompt(conn: connection, prompt: str, conv_id: Optional[int] = None, scopes: Optional[List[str]] = None) -> int:
    if conv_id and (blocking_id := get_blocking_message_id(conn, conv_id)):
        raise ConversationBlockedError(blocking_id)
    
    if not conv_id:
        conv_id = create_conversation(conn)
    
    user_message_dict: StoredMessage = {
        'type': 'message',
        'author': 'user',
        'content': prompt,
        'scopes': scopes if scopes else []
    }
        
    insert_message(conn, conv_id, 'user', user_message_dict)
    conn.commit()
    return conv_id

def get_messages_for_continuation(conn: connection, conv_id: int) -> tuple[ChatContext, List[str]]:
    ctx = ChatContext()
    scopes: list[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT id, role, elements FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        for row in cur.fetchall():
            message_id, role, element = row
            element_dict = json.loads(element) if isinstance(element, str) else element
            ctx.append_from_db(message_id, element_dict)
            if role == 'user':
                scopes = element_dict.get('scopes') or []
    return ctx, scopes

def continue_conversation(conn: connection, conv_id: int, model_id: str, functions: List[Any]) -> Generator[bytes, None, None]:
    from tools import run_tool_call
    context, scopes = get_messages_for_continuation(conn, conv_id)

    run_next_loop = True
    while run_next_loop:
        processor = DeltaProcessor()
        chat_gen_inner = run_chat_completion_stream(model_id, context, functions)
        run_next_loop = False
        for chunk in chat_gen_inner:
            (delta, aggregated_element) = processor.process(chunk)
            if aggregated_element is not None:
                match aggregated_element:
                    case FinishedMessage() | FinishedThinking():
                        result_json = to_stored_elem(aggregated_element)
                        msg_id = insert_message(conn, conv_id, 'assistant', result_json)
                        context.append_finalized(msg_id, aggregated_element)
                        yield json.dumps({'type': 'finalized', 'id': msg_id}).encode() + b'\n'
                    case FinishedToolCall():
                        # Pass the extracted scopes to the tool call
                        result = run_tool_call(aggregated_element, scopes=scopes)
                        result_json = to_stored_elem(result)
                        msg_id = insert_message(conn, conv_id, 'assistant', result_json)
                        conn.commit()
                        yield json.dumps(result_json).encode() + b'\n'
                        context.append_finalized(msg_id, result)
                        if result.is_blocking:
                            cur = conn.cursor()
                            cur.execute(
                                "UPDATE conversations SET blocking_message_id = %s WHERE id = %s",
                                (msg_id, conv_id),
                            )
                            run_next_loop = False
                        else:
                            run_next_loop = True
                        yield json.dumps({'type': 'finalized', 'id': msg_id}).encode() + b'\n'
            if delta is not None:
                if isinstance(delta, StreamingMessage):
                    yield json.dumps({'type': 'message', 'content': delta.content}).encode() + b'\n'
                elif isinstance(delta, StreamingThinking):
                    yield json.dumps({'type': 'thinking', 'content': delta.content}).encode() + b'\n'
    conn.commit()
    print("Stream finished")

def get_conversations(conn: connection) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [{'id': r[0], 'title': r[1], 'created_at': r[2].isoformat()} for r in rows]

def get_conversation_details(conn: connection, conv_id: int) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, created_at, blocking_message_id FROM conversations WHERE id = %s", (conv_id,))
        row = cur.fetchone()
        if not row:
            return None
        cur.execute("SELECT id, role, elements, created_at FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        messages = [{'id': r[0], 'role': r[1], 'element': r[2], 'created_at': r[3].isoformat()} for r in cur.fetchall()]
        return {
            'id': row[0],
            'title': row[1],
            'created_at': row[2].isoformat(),
            'blocking_message_id': row[3],
            'messages': messages
        }

def decide_tool_call(conn: connection, conv_id: int, msg_id: int, decision: Literal['approve'] | Literal['reject'], comment: str = "") -> bool:
    cur = conn.cursor()
    cur.execute('SELECT elements FROM messages WHERE id = %s AND conversation_id = %s', (msg_id, conv_id))
    row = cur.fetchone()
    if not row:
        raise ValueError('message not found')
    
    elem_json = row[0]
    elem = json.loads(elem_json) if isinstance(elem_json, str) else elem_json

    decision_elem: StoredToolCallDecision = {
        'type': 'tool_call_decision',
        'decision': decision,
        'original_message_id': msg_id,
        'comment': comment
    }
    
    insert_message(conn, conv_id, 'assistant', decision_elem)
    cur.execute('UPDATE conversations SET blocking_message_id = NULL WHERE id = %s', (conv_id,))

    executed = False
    if decision == 'approve':
        if elem.get('type') != 'tool_call':
            raise ValueError('original message is not a tool call')
        from tools import run_tool_call
        tool_call = FinishedToolCall(name=elem.get('name'), parameters=elem.get('parameters'))
        
        scopes = get_scopes_from_last_user_message(conn, conv_id)
        result = run_tool_call(tool_call, privileged=True, scopes=scopes)
        result_elem: StoredToolCallResult = {
            'type': 'tool_call_result',
            'original_message_id': msg_id,
            'result': result.result,
        }
        insert_message(conn, conv_id, 'system', result_elem)
        executed = True
    
    conn.commit()
    return executed

def delete_conversation(conn: connection, conv_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM conversations WHERE id = %s", (conv_id,))
        if not cur.fetchone():
            return False
        cur.execute("DELETE FROM messages WHERE conversation_id = %s", (conv_id,))
        cur.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))
        conn.commit()
        return True
