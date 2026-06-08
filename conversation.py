from typing import AsyncGenerator, Literal, Optional, Any, List
import json
from pydantic import BaseModel
from psycopg2.extensions import connection

from domain import (
    Message,
    ScopeSpec,
    Thinking,
    ToolCallFinishedOrBlocked,
    ToolCallResult,
    ToolCallDecision,
    ConversationElement,
    stored_element_adapter,
)
from inference import (
    StreamingMessage,
    StreamingThinking,
    FinishedMessage,
    FinishedThinking,
    FinishedToolCall,
    FinishedToolCallResult,
    FinishedElement,
    Tool,
)

from inference import registry
from log_config import get_logger

logger = get_logger(__name__)


class GenerationRequest(BaseModel):
    agent_id: Optional[str] = None
    provider_key: Optional[str] = None
    model_id: Optional[str] = None
    scopes: list[ScopeSpec] = []


class PromptRequest(GenerationRequest):
    prompt: str
    conversation_id: Optional[int] = None


class ContinueRequest(GenerationRequest):
    pass


class ConversationBlockedError(Exception):
    def __init__(self, blocking_message_id: int):
        self.blocking_message_id = blocking_message_id

def to_conv_elem(element: FinishedElement) -> ConversationElement:
    match element:
        case FinishedMessage(content=content):
            return Message(author='assistant', content=content, scopes=[])
        case FinishedThinking(content=content):
            return Thinking(content=content)
        case FinishedToolCallResult(name=name, parameters=parameters, result=result, is_blocking=is_blocking):
            return ToolCallFinishedOrBlocked(
                name=name,
                parameters=parameters,
                result=result,
                is_blocking=is_blocking,
                status='pending' if is_blocking else 'completed'
            )
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


def insert_message(conn: connection, conv_id: int, role: str, element: ConversationElement) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages (conversation_id, role, elements, created_at)
            VALUES (%s, %s, %s, NOW())
            RETURNING id
            """,
            (conv_id, role, json.dumps(element.model_dump()))
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


def get_scopes_from_last_user_message(conn: connection, conv_id: int) -> List[ScopeSpec]:
    with conn.cursor() as cur:
        cur.execute("SELECT elements FROM messages WHERE conversation_id = %s AND role = 'user' ORDER BY created_at DESC LIMIT 1", (conv_id,))
        row = cur.fetchone()
        if row:
            elem_dict = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            elem = stored_element_adapter.validate_python(elem_dict)
            if isinstance(elem, Message) and elem.author == 'user':
                return elem.scopes or []
        return []


def prepare_conversation_with_prompt(conn: connection, prompt: str, conv_id: Optional[int] = None, scopes: Optional[List[ScopeSpec]] = None) -> int:
    if conv_id and (blocking_id := get_blocking_message_id(conn, conv_id)):
        raise ConversationBlockedError(blocking_id)
    
    if not conv_id:
        conv_id = create_conversation(conn)
    
    user_message = Message(
        author='user',
        content=prompt,
        scopes=scopes if scopes else []
    )
        
    insert_message(conn, conv_id, 'user', user_message)
    conn.commit()
    return conv_id


def get_messages_for_continuation(conn: connection, conv_id: int) -> tuple[list[tuple[int, ConversationElement]], list[ScopeSpec]]:
    ctx: list[tuple[int, ConversationElement]] = []
    scopes: list[ScopeSpec] = []
    with conn.cursor() as cur:
        cur.execute("SELECT id, role, elements FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        for row in cur.fetchall():
            message_id, role, element = row
            element_dict = json.loads(element) if isinstance(element, str) else element
            parsed = stored_element_adapter.validate_python(element_dict)
            ctx.append((message_id, parsed))
            if role == 'user' and isinstance(parsed, Message):
                scopes = parsed.scopes or []
    return ctx, scopes


async def continue_conversation(conn: connection, conv_id: int, model_id: str, functions: list[Tool], provider_key: str, inference_config: dict[str, Any] = {}) -> AsyncGenerator[bytes, None]:
    from tools import run_tool_call
    context, scopes = get_messages_for_continuation(conn, conv_id)  # scopes: list[ScopeSpec]

    provider = registry.get(provider_key)

    run_next_loop = True
    while run_next_loop:
        chat_gen_inner = provider.run_chat_completion_stream(model_id, context, functions)
        run_next_loop = False
        async for delta, aggregated_element in chat_gen_inner:
            if aggregated_element is not None:
                match aggregated_element:
                    case FinishedMessage() | FinishedThinking():
                        result = to_conv_elem(aggregated_element)
                        msg_id = insert_message(conn, conv_id, 'assistant', result)
                        context.append((msg_id, result))
                        yield json.dumps({'type': 'finalized', 'id': msg_id}).encode() + b'\n'
                    case FinishedToolCall():
                        assert isinstance(aggregated_element, FinishedToolCall)
                        # Pass the extracted scopes to the tool call
                        tool_result = await run_tool_call(aggregated_element, scopes=scopes)
                        result_element = to_conv_elem(tool_result)
                        msg_id = insert_message(conn, conv_id, 'assistant', result_element)
                        conn.commit()
                        yield result_element.model_dump_json().encode() + b'\n'
                        context.append((msg_id, result_element))
                        if tool_result.is_blocking:
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
    logger.info("Stream finished")


def get_conversations(conn: connection) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [{'id': r[0], 'title': r[1], 'created_at': r[2].isoformat()} for r in rows]


def get_conversation_details(conn: connection, conv_id: int) -> Optional[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, created_at, blocking_message_id FROM conversations WHERE id = %s", (conv_id,))
        row = cur.fetchone()
        if not row:
            return None
        cur.execute("SELECT id, role, elements, created_at FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        messages = []
        for r in cur.fetchall():
            elem_dict = json.loads(r[2]) if isinstance(r[2], str) else r[2]
            parsed = stored_element_adapter.validate_python(elem_dict)
            messages.append({'id': r[0], 'role': r[1], 'element': parsed.model_dump(), 'created_at': r[3].isoformat()})
        return {
            'id': row[0],
            'title': row[1],
            'created_at': row[2].isoformat(),
            'blocking_message_id': row[3],
            'messages': messages
        }


async def decide_tool_call(conn: connection, conv_id: int, msg_id: int, decision: Literal['approve', 'reject'], comment: str = "") -> bool:
    cur = conn.cursor()
    cur.execute('SELECT elements FROM messages WHERE id = %s AND conversation_id = %s', (msg_id, conv_id))
    row = cur.fetchone()
    if not row:
        raise ValueError('message not found')
    
    elem_dict = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    elem = stored_element_adapter.validate_python(elem_dict)

    decision_elem = ToolCallDecision(
        decision=decision,
        original_message_id=msg_id,
        comment=comment or ""
    )
    
    insert_message(conn, conv_id, 'assistant', decision_elem)
    cur.execute('UPDATE conversations SET blocking_message_id = NULL WHERE id = %s', (conv_id,))

    executed = False
    if decision == 'approve':
        if not isinstance(elem, ToolCallFinishedOrBlocked):
            raise ValueError('original message is not a tool call')
        from tools import run_tool_call
        tool_call = FinishedToolCall(name=elem.name, parameters=elem.parameters)
        
        scopes = get_scopes_from_last_user_message(conn, conv_id)
        result = await run_tool_call(tool_call, privileged=True, scopes=scopes)
        result_elem = ToolCallResult(
            original_message_id=msg_id,
            result=result.result,
        )
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