from typing import Dict, Optional, Any, Generator, Union, List
import json
from psycopg2.extensions import connection
from inference import (
    Message,
    Thinking,
    ToolCall,
    ToolCallResult,
    StreamingElement,
    FinishedElement,
    DeltaProcessor,
    ChatContext,
    run_chat_completion_stream
)

class ConversationBlockedError(Exception):
    def __init__(self, blocking_message_id: int):
        self.blocking_message_id = blocking_message_id

def to_json_dict(element: FinishedElement) -> dict[str, Any]:
    match element:
        case Message(content=content):
            return {'type': 'message', 'content': content}
        case Thinking(content=content):
            return {'type': 'thinking', 'content': content}
        case ToolCallResult(name=name, parameters=parameters, result=result, is_blocking=is_blocking):
            return {'type': 'tool_call', 'name': name, 'parameters': parameters, 'result': result, 'is_blocking': is_blocking}


def create_conversation(conn: connection) -> int:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO conversations DEFAULT VALUES RETURNING id")
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to create conversation")
        conv_id = row[0]
        conn.commit()
        return conv_id

def insert_message(conn: connection, conv_id: int, role: str, element: Dict[str, Any]) -> int:
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
    return id

def get_blocking_message_id(conn: connection, conv_id: int) -> Optional[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT blocking_message_id FROM conversations WHERE id = %s",
            (conv_id,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

def prepare_conversation_with_prompt(conn: connection, prompt: str, conv_id: Optional[int] = None) -> int:
    """Prepares a conversation by creating it if needed and inserting the user prompt.
    
    Raises:
        ConversationBlockedError: If the conversation is waiting for a tool call decision.
    """
    if conv_id and (blocking_id := get_blocking_message_id(conn, conv_id)):
        raise ConversationBlockedError(blocking_id)
    
    if not conv_id:
        conv_id = create_conversation(conn)
    
    user_message = Message(content=prompt)
    user_message_dict = to_json_dict(user_message)
    insert_message(conn, conv_id, 'user', user_message_dict)
    conn.commit()
    return conv_id

def get_messages_for_continuation(conn: connection, conv_id: int) -> ChatContext:
    ctx = ChatContext()
    with conn.cursor() as cur:
        cur.execute("SELECT id, role, elements FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        for row in cur.fetchall():
            message_id, role, element = row
            element_dict = json.loads(element) if isinstance(element, str) else element
            element_dict['role'] = role
            ctx.append_from_db(message_id, element_dict)
    return ctx

def continue_conversation(conn: connection, conv_id: int, model_id: str, functions: List[Any]) -> Generator[bytes, None, None]:
    from tools import run_tool_call
    context = get_messages_for_continuation(conn, conv_id)
    run_next_loop = True
    while run_next_loop:
        processor = DeltaProcessor()
        chat_gen_inner = run_chat_completion_stream(model_id, context, functions)
        run_next_loop = False
        for chunk in chat_gen_inner:
            (delta, aggregated_element) = processor.process(chunk)
            if delta is not None:
                if isinstance(delta, Message):
                    yield json.dumps({'type': 'response', 'content': delta.content}).encode() + b'\n'
                elif isinstance(delta, Thinking):
                    yield json.dumps({'type': 'reasoning', 'content': delta.content}).encode() + b'\n'
            if aggregated_element is not None:
                match aggregated_element:
                    case Message() | Thinking() as msg:
                        result_json = to_json_dict(msg)
                        msg_id = insert_message(conn, conv_id, 'assistant', result_json)
                        context.append_finalized(msg_id, msg)
                    case ToolCall() as call:
                        result = run_tool_call(call)
                        result_json = to_json_dict(result)
                        if result.is_blocking:
                            result_json['status'] = 'pending'
                        else:
                            result_json['status'] = 'completed'
                        msg_id = insert_message(conn, conv_id, 'assistant', result_json)
                        result_json['id'] = msg_id
                        yield json.dumps(result_json).encode() + b'\n'
                        context.append_finalized(msg_id, result)
                        if result.is_blocking:
                            cur = conn.cursor()
                            cur.execute(
                                "UPDATE conversations SET blocking_message_id = %s WHERE id = %s",
                                (msg_id, conv_id),
                            )
                            conn.commit()
                            run_next_loop = False
                        else:
                            run_next_loop = True
    # Stream finished

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

def decide_tool_call(conn: connection, conv_id: int, msg_id: int, decision: str) -> bool:
    """Handle the user's approval or rejection of a tool call proposal."""
    cur = conn.cursor()
    # Fetch the message elements
    cur.execute('SELECT elements FROM messages WHERE id = %s AND conversation_id = %s', (msg_id, conv_id))
    row = cur.fetchone()
    if not row:
        raise ValueError('message not found')
    
    elem_json = row[0]
    elem = json.loads(elem_json) if isinstance(elem_json, str) else elem_json

    # Build the decision element
    decision_elem: Dict[str, Any] = {
        'type': 'tool_decision',
        'decision': decision,
        'original_message_id': msg_id,
    }

    # Insert decision message
    insert_message(conn, conv_id, 'assistant', decision_elem)

    # Clear blocking status
    cur.execute('UPDATE conversations SET blocking_message_id = NULL WHERE id = %s', (conv_id,))

    executed = False
    if decision == 'approve':
        # Reconstruct the tool call
        if elem.get('type') != 'tool_call':
            raise ValueError('original message is not a tool call')
        from tools import run_tool_call
        tool_call = ToolCall(name=elem.get('name'), parameters=elem.get('parameters'))
        # Execute with privilege to perform the operation
        result = run_tool_call(tool_call, privileged=True)
        # Insert tool result message
        result_elem: Dict[str, Any] = {
            'type': 'tool_result',
            'original_message_id': msg_id,
            'result': result.result,
        }
        insert_message(conn, conv_id, 'assistant', result_elem)
        executed = True
    
    conn.commit()
    return executed
