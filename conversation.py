from typing import AsyncGenerator, Literal, Optional, Any, List
import json
from pydantic import BaseModel
from psycopg2.extensions import connection
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from domain import (
    Message,
    Thinking,
    ToolCallFinishedOrBlocked,
    ToolCallResult,
    ToolCallDecision,
    ConversationElement,
    stored_element_adapter,
    ScopeSpec,
    SecurityPolicy,
)
from inference import (
    StreamingMessage,
    StreamingThinking,
    FinishedMessage,
    FinishedThinking,
    FinishedToolCall,
    FinishedElement,
    ChatContext,
    registry
)
from repositories import get_repo_by_id, get_repo_by_name
from tools import Tool, run_tool_call, TOOL_REGISTRY

from log_config import get_logger
from database import mk_conn
from agents import get_agent_by_name, AgentConfig

logger = get_logger(__name__)

router = APIRouter()


class UserScopeChoice(BaseModel):
    internal_name: str
    security_policy_override: Optional[SecurityPolicy]

class GenerationRequest(BaseModel):
    agent_id: Optional[str] = None
    provider_key: Optional[str] = None
    model_id: Optional[str] = None
    scopes: list[UserScopeChoice] = []

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
            (conv_id,)
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

def resolve_scope(choice: UserScopeChoice, agent: Optional[AgentConfig] = None) -> ScopeSpec:
    """
    Priority: User override > Agent override > Repository policy.
    """
    # 1. User override
    if choice.security_policy_override is not None:
        return ScopeSpec(
            internal_name=choice.internal_name,
            security_policy=choice.security_policy_override,
        )
    # 2. Agent override
    if agent is not None:
        for ap in agent.repository_access:
            if ap.repository_internal_name == choice.internal_name and ap.security_policy_override is not None:
                return ScopeSpec(
                    internal_name=choice.internal_name,
                    security_policy=ap.security_policy_override,
                )
    # 3. Repository policy
    repo = get_repo_by_name(choice.internal_name)
    if repo is None:
        raise ValueError(f"Repository '{choice.internal_name}' not found")
    return ScopeSpec(
        internal_name=choice.internal_name,
        security_policy=repo.security_policy,
    )

def _resolve_agent(agent_id: Optional[str], fallback_provider: Optional[str], fallback_model: Optional[str], req_scopes: list[UserScopeChoice]) -> tuple:
    """Resolve (provider_key, model_id, inference_config, scopes, agent_prompt) from agent or fallback.
    Returns None for provider_key if validation fails (caller handles error)."""
    if agent_id:
        agent = get_agent_by_name(agent_id)
        if agent is None:
            raise ValueError(f"Agent '{agent_id}' not found")
        scopes = [resolve_scope(s, agent=agent) for s in req_scopes]
        extra_scopes = {s.internal_name for s in scopes}
        for agent_policy in agent.repository_access:
            repo_key = agent_policy.repository_internal_name
            if repo_key not in extra_scopes:
                # Add default policy of Agent (agent override > repo policy)
                if agent_policy.security_policy_override is None:
                    repo_config = get_repo_by_id(agent_policy.repository_id)
                    if repo_config is None:
                        raise Exception(f"Repository {agent_policy.repository_id} not found")
                    resolved_policy = repo_config.security_policy
                else:
                    resolved_policy = agent_policy.security_policy_override
                scopes.append(ScopeSpec(
                    internal_name=repo_key,
                    security_policy=resolved_policy
                ))
            
        return agent.provider_key, agent.model_id, agent.inference_config, scopes, agent.instructions
    if not fallback_provider:
        raise ValueError('provider_key required')
    if not fallback_model:
        raise ValueError('model_id required')
    # No agent \u2014 user override > repo policy
    resolved_scopes = [resolve_scope(s) for s in req_scopes]
    return fallback_provider, fallback_model, {}, resolved_scopes, None

def prepare_conversation_with_prompt(conn: connection, prompt: str, conv_id: Optional[int], scopes: List[ScopeSpec]) -> int:
    if conv_id and (blocking_id := get_blocking_message_id(conn, conv_id)):
        raise ConversationBlockedError(blocking_id)
    
    if not conv_id:
        conv_id = create_conversation(conn)
    
    user_message = Message(
        author='user',
        content=prompt,
        scopes=scopes
    )
        
    insert_message(conn, conv_id, 'user', user_message)
    conn.commit()
    return conv_id

def get_messages_for_continuation(conn: connection, conv_id: int) -> list[tuple[int, ConversationElement]]:
    ctx: list[tuple[int, ConversationElement]] = []
    with conn.cursor() as cur:
        cur.execute("SELECT id, role, elements FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conv_id,))
        for row in cur.fetchall():
            message_id, role, element = row
            element_dict = json.loads(element) if isinstance(element, str) else element
            parsed = stored_element_adapter.validate_python(element_dict)
            ctx.append((message_id, parsed))
    return ctx

async def continue_conversation(conn: connection, conv_id: int, functions: list[Tool], agent_id: Optional[str] = None, provider_key: Optional[str] = None, model_id: Optional[str] = None, extra_scopes: list[UserScopeChoice] = []) -> AsyncGenerator[bytes, None]:
    provider_key, model_id, inference_config, scopes, agent_prompt = _resolve_agent(agent_id, provider_key, model_id, extra_scopes)
    messages = get_messages_for_continuation(conn, conv_id)

    provider = registry.get(provider_key)

    run_next_loop = True
    while run_next_loop:
        chat_context = ChatContext(messages=messages, scopes=scopes, tools=functions, instructions=agent_prompt)
        chat_gen_inner = provider.run_chat_completion_stream(model_id, chat_context, functions)
        run_next_loop = False
        async for delta, aggregated_element in chat_gen_inner:
            if aggregated_element is not None:
                match aggregated_element:
                    case FinishedMessage() | FinishedThinking():
                        result = to_conv_elem(aggregated_element)
                        msg_id = insert_message(conn, conv_id, 'assistant', result)
                        messages.append((msg_id, result))
                        yield json.dumps({'type': 'finalized', 'id': msg_id}).encode() + b'\n'
                    case FinishedToolCall(name=name, parameters=parameters):
                        assert isinstance(aggregated_element, FinishedToolCall)
                        # Pass the extracted scopes to the tool call
                        tool_result = await run_tool_call(name, parameters, False, scopes)
                        result_element = ToolCallFinishedOrBlocked(
                            name=name,
                            parameters=parameters,
                            result=tool_result.result,
                            is_blocking=tool_result.is_blocking,
                            status='pending' if tool_result.is_blocking else 'completed'
                        )
                        msg_id = insert_message(conn, conv_id, 'assistant', result_element)
                        conn.commit()
                        yield result_element.model_dump_json().encode() + b'\n'
                        messages.append((msg_id, result_element))
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

        scopes = get_scopes_from_last_user_message(conn, conv_id)
        result = await run_tool_call(elem.name, elem.parameters, privileged=True, scopes=scopes)
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



@router.post('/api/conversations/prompt')
async def prompt_model(payload: PromptRequest):
    prompt = payload.prompt
    conversation_id = payload.conversation_id
    try:
        provider_key, model_id, inference_config, scopes, agent_prompt = _resolve_agent(
            payload.agent_id, payload.provider_key, payload.model_id, payload.scopes,
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={'error': str(e)})

    try:
        conn = mk_conn()
        conversation_id = prepare_conversation_with_prompt(conn, prompt, conversation_id, scopes=scopes)
    except ConversationBlockedError as e:
        return JSONResponse(
            status_code=403,
            content={"error": "Action required", "blocking_message_id": e.blocking_message_id}
        )

    return StreamingResponse(
        continue_conversation(conn, conversation_id, TOOL_REGISTRY, agent_id=payload.agent_id, provider_key=payload.provider_key, model_id=payload.model_id, extra_scopes=payload.scopes),
        media_type='application/x-ndjson', headers={'X-Conversation-ID': str(conversation_id)}
    )



@router.get('/api/conversations')
async def list_conversations():
    conn = mk_conn()
    conversations = get_conversations(conn)
    return JSONResponse(content=conversations)



@router.get('/api/conversations/{conv_id}')
async def get_conversation(conv_id: int):
    conn = mk_conn()
    details = get_conversation_details(conn, conv_id)
    if not details:
        return JSONResponse(status_code=404, content={'error': 'Not found'})
    return JSONResponse(content=details)



@router.delete('/api/conversations/{conv_id}')
async def delete_conversation_endpoint(conv_id: int):
    try:
        conn = mk_conn()
        success = delete_conversation(conn, conv_id)
        if not success:
            return JSONResponse(status_code=404, content={'error': 'Conversation not found'})
        return JSONResponse(content={'status': 'deleted', 'id': conv_id})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})



@router.post('/api/conversations/{conv_id}/tool_calls/{msg_id}/decide')
async def decide_tool_call_endpoint(conv_id: int, msg_id: int, request: Request):
    payload = await request.json()
    decision = payload.get('decision')
    comment = payload.get('comment')
    if decision not in {'approve', 'reject'}:
        return JSONResponse(status_code=400, content={'error': 'invalid decision'})

    try:
        conn = mk_conn()
        executed = await decide_tool_call(conn, conv_id, msg_id, decision, comment=comment)
        return JSONResponse(content={'status': 'success', 'executed': executed})
    except ValueError as e:
        return JSONResponse(status_code=400, content={'error': str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})



@router.post('/api/conversations/{conversation_id}/continue')
async def continue_conversation_endpoint(conversation_id: int, payload: ContinueRequest):
    conn = mk_conn()
    details = get_conversation_details(conn, conversation_id)
    if not details:
        return JSONResponse(status_code=404, content={'error': 'conversation not found'})
    return StreamingResponse(
        continue_conversation(conn, conversation_id, TOOL_REGISTRY, agent_id=payload.agent_id, provider_key=payload.provider_key, model_id=payload.model_id, extra_scopes=payload.scopes),
        media_type='application/x-ndjson', headers={'X-Conversation-ID': str(conversation_id)}
    )
