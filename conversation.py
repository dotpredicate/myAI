from typing import AsyncGenerator, Literal, Optional, Any, Union
import asyncio
import json
from pydantic import BaseModel
from psycopg import AsyncConnection
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
    StreamingToolCall,
    StreamingElement,
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
    security_policy_override: Optional[SecurityPolicy] = None

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


class MessageChunk(BaseModel):
    type: Literal['message'] = 'message'
    content: str
    role: Literal['user', 'assistant'] = 'assistant'

class ThinkingChunk(BaseModel):
    type: Literal['thinking'] = 'thinking'
    content: str

class ToolCallEvent(BaseModel):
    type: Literal['tool_call'] = 'tool_call'
    name: str
    parameters: str
    result: str
    is_blocking: bool
    status: Literal['pending'] | Literal['completed']

class ElementFinalized(BaseModel):
    type: Literal['finalized'] = 'finalized'
    id: int
    sequence_id: int

class GenerationErrorEvent(BaseModel):
    type: Literal['generation_error'] = 'generation_error'
    message: str

StreamEvent = Union[MessageChunk, ThinkingChunk, ElementFinalized, ToolCallEvent, GenerationErrorEvent]

class StoredMessageRecord(BaseModel):
    id: int
    conversation_id: int
    sequence_id: int
    role: str
    element: ConversationElement
    created_at: str

class ConversationBlockedError(Exception):
    def __init__(self, blocking_message_id: int):
        self.blocking_message_id = blocking_message_id

class ActiveGenerationExistsError(Exception):
    pass

class ActiveGeneration:
    def __init__(self, conversation_id: int, last_finalized_sequence_id: Optional[int]) -> None:
        self.conversation_id = conversation_id
        self.task: Optional[asyncio.Task[None]] = None
        self.status: Literal['running', 'finished', 'failed', 'blocked', 'cancelled'] = 'running'
        self.buffered_element: Optional[StreamingElement] = None
        self.pending_events: list[StreamEvent] = []
        self.last_finalized_sequence_id = last_finalized_sequence_id
        self.subscribers: set[asyncio.Queue[Optional[StreamEvent]]] = set()
        self.lock = asyncio.Lock()
        self.error: Optional[str] = None

    async def subscribe(self) -> tuple[Optional[int], list[StreamEvent], asyncio.Queue[Optional[StreamEvent]]]:
        queue: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue()
        async with self.lock:
            boundary_sequence_id = self.last_finalized_sequence_id
            snapshot_events = list(self.pending_events)
            if buffer_event := streaming_element_to_stream_event(self.buffered_element):
                snapshot_events.append(buffer_event)
            if self.status == 'running':
                self.subscribers.add(queue)
            else:
                queue.put_nowait(None)
        return boundary_sequence_id, snapshot_events, queue

    async def unsubscribe(self, queue: asyncio.Queue[Optional[StreamEvent]]) -> None:
        async with self.lock:
            self.subscribers.discard(queue)

    async def apply_event(self, event: StreamEvent) -> None:
        async with self.lock:
            if isinstance(event, MessageChunk):
                self.buffered_element = merge_streaming_element(
                    self.buffered_element,
                    StreamingMessage(event.content),
                )
            elif isinstance(event, ThinkingChunk):
                self.buffered_element = merge_streaming_element(
                    self.buffered_element,
                    StreamingThinking(event.content),
                )
            elif isinstance(event, ToolCallEvent):
                self.buffered_element = None
                self.pending_events.append(event)
                if event.is_blocking:
                    self.status = 'blocked'
            elif isinstance(event, ElementFinalized):
                self.buffered_element = None
                self.pending_events.clear()
                self.last_finalized_sequence_id = event.sequence_id
            elif isinstance(event, GenerationErrorEvent):
                self.status = 'failed'
                self.error = event.message
            subscribers = list(self.subscribers)
        for queue in subscribers:
            queue.put_nowait(event)

    async def finish(self) -> None:
        async with self.lock:
            if self.status == 'running':
                self.status = 'finished'
            for queue in list(self.subscribers):
                queue.put_nowait(None)
            self.subscribers.clear()

    async def fail(self, message: str) -> None:
        await self.apply_event(GenerationErrorEvent(message=message))
        await self.finish()

    async def cancel(self) -> None:
        async with self.lock:
            self.status = 'cancelled'
            for queue in list(self.subscribers):
                queue.put_nowait(None)
            self.subscribers.clear()

active_generations: dict[int, ActiveGeneration] = {}
active_generations_lock = asyncio.Lock()

def to_conv_elem(element: FinishedElement) -> ConversationElement:
    match element:
        case FinishedMessage(content=content):
            return Message(author='assistant', content=content)
        case FinishedThinking(content=content):
            return Thinking(content=content)
        case _:
            raise ValueError(f'Unhandled tool call type {type(element)}')

def merge_streaming_element(current: Optional[StreamingElement], delta: StreamingElement) -> StreamingElement:
    match (current, delta):
        case (StreamingMessage(c1), StreamingMessage(c2)):
            return StreamingMessage(c1 + c2)
        case (StreamingThinking(c1), StreamingThinking(c2)):
            return StreamingThinking(c1 + c2)
        case (StreamingToolCall(n1, p1), StreamingToolCall(n2, p2)):
            return StreamingToolCall(n1 or n2, (p1 or "") + (p2 or ""))
        case _:
            return delta

def streaming_element_to_stream_event(element: Optional[StreamingElement]) -> Optional[StreamEvent]:
    if isinstance(element, StreamingMessage):
        return MessageChunk(content=element.content, role='assistant')
    if isinstance(element, StreamingThinking):
        return ThinkingChunk(content=element.content)
    return None

def stored_record_to_stream_events(record: StoredMessageRecord) -> list[StreamEvent]:
    finalized = ElementFinalized(id=record.id, sequence_id=record.sequence_id)
    match record.element:
        case Message(author=author, content=content):
            return [MessageChunk(content=content, role=author), finalized]
        case Thinking(content=content):
            return [ThinkingChunk(content=content), finalized]
        case ToolCallFinishedOrBlocked(name=name, parameters=parameters, result=result, is_blocking=is_blocking, status=status):
            return [
                ToolCallEvent(
                    name=name,
                    parameters=parameters,
                    result=result,
                    is_blocking=is_blocking,
                    status=status,
                ),
                finalized,
            ]
        case ToolCallResult(original_message_id=original_message_id, result=result):
            return [
                ToolCallEvent(
                    name=f"tool_result:{original_message_id}",
                    parameters="",
                    result=result,
                    is_blocking=False,
                    status='completed',
                ),
                finalized,
            ]
        case _:
            return [finalized]

def serialize_ndjson_event(event: StreamEvent) -> str:
    return event.model_dump_json() + '\n'

async def create_conversation(conn: AsyncConnection) -> int:
    async with conn.cursor() as cur:
        await cur.execute("INSERT INTO conversations DEFAULT VALUES RETURNING id")
        row = await cur.fetchone()
        if not row:
            raise RuntimeError("Failed to create conversation")
        conv_id = row[0]
        return conv_id


async def get_latest_sequence_id(conn: AsyncConnection, conv_id: int) -> Optional[int]:
    async with conn.cursor() as cur:
        await cur.execute("SELECT MAX(sequence_id) FROM messages WHERE conversation_id = %s", (conv_id,))
        row = await cur.fetchone()
        return row[0] if row and row[0] is not None else None

async def _allocate_next_sequence_id(conn: AsyncConnection, conv_id: int) -> int:
    async with conn.cursor() as cur:
        await cur.execute("SELECT id FROM conversations WHERE id = %s FOR UPDATE", (conv_id,))
        if not await cur.fetchone():
            raise ValueError('conversation not found')
    latest = await get_latest_sequence_id(conn, conv_id)
    return 1 if latest is None else latest + 1

async def insert_message(conn: AsyncConnection, conv_id: int, role: str, element: ConversationElement) -> StoredMessageRecord:
    sequence_id = await _allocate_next_sequence_id(conn, conv_id)
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO messages (conversation_id, sequence_id, role, elements, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            RETURNING id, created_at
            """,
            (conv_id, sequence_id, role, json.dumps(element.model_dump()))
        )
        row = await cur.fetchone()
        assert row is not None
        return StoredMessageRecord(
            id=row[0],
            conversation_id=conv_id,
            sequence_id=sequence_id,
            role=role,
            element=element,
            created_at=row[1].isoformat(),
        )

async def get_messages_after(
    conn: AsyncConnection,
    conv_id: int,
    after_sequence_id: Optional[int] = None,
    until_sequence_id: Optional[int] = None,
) -> list[StoredMessageRecord]:
    conditions = ["conversation_id = %s"]
    params: list[Any] = [conv_id]
    if after_sequence_id is not None:
        conditions.append("sequence_id > %s")
        params.append(after_sequence_id)
    if until_sequence_id is not None:
        conditions.append("sequence_id <= %s")
        params.append(until_sequence_id)

    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT id, conversation_id, sequence_id, role, elements, created_at
            FROM messages
            WHERE {' AND '.join(conditions)}
            ORDER BY sequence_id ASC
            """,
            params,
        )
        records: list[StoredMessageRecord] = []
        async for row in cur:
            elem_dict = json.loads(row[4]) if isinstance(row[4], str) else row[4]
            parsed = stored_element_adapter.validate_python(elem_dict)
            records.append(StoredMessageRecord(
                id=row[0],
                conversation_id=row[1],
                sequence_id=row[2],
                role=row[3],
                element=parsed,
                created_at=row[5].isoformat(),
            ))
        return records


async def get_blocking_message_id(conn: AsyncConnection, conv_id: int) -> Optional[int]:
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT blocking_message_id FROM conversations WHERE id = %s",
            (conv_id,)
        )
        row = await cur.fetchone()
        return row[0] if row and row[0] is not None else None

async def resolve_scope(choice: UserScopeChoice, agent: Optional[AgentConfig] = None) -> ScopeSpec:
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
    repo = await get_repo_by_name(choice.internal_name)
    if repo is None:
        raise ValueError(f"Repository '{choice.internal_name}' not found")
    return ScopeSpec(
        internal_name=choice.internal_name,
        security_policy=repo.security_policy,
    )

async def _resolve_agent(agent_id: Optional[str], fallback_provider: Optional[str], fallback_model: Optional[str], req_scopes: list[UserScopeChoice]) -> tuple:
    """Resolve (provider_key, model_id, inference_config, scopes, agent_prompt) from agent or fallback.
    Returns None for provider_key if validation fails (caller handles error)."""
    if agent_id:
        agent = await get_agent_by_name(agent_id)
        if agent is None:
            raise ValueError(f"Agent '{agent_id}' not found")
        scopes = [await resolve_scope(s, agent=agent) for s in req_scopes]
        extra_scopes = {s.internal_name for s in scopes}
        for agent_policy in agent.repository_access:
            repo_key = agent_policy.repository_internal_name
            if repo_key not in extra_scopes:
                # Add default policy of Agent (agent override > repo policy)
                if agent_policy.security_policy_override is None:
                    repo_config = await get_repo_by_id(agent_policy.repository_id)
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
    # No agent: user override > repo policy
    resolved_scopes = [await resolve_scope(s) for s in req_scopes]
    return fallback_provider, fallback_model, {}, resolved_scopes, None

async def prepare_and_start_generation(
    prompt: str,
    conversation_id: Optional[int],
    functions: list[Tool],
    agent_id: Optional[str] = None,
    provider_key: Optional[str] = None,
    model_id: Optional[str] = None,
    extra_scopes: list[UserScopeChoice] = [],
) -> tuple[int, StoredMessageRecord]:
    if conversation_id is None:
        async with mk_conn() as conn:
            conversation_id = await create_conversation(conn)
            user_record = await insert_message(conn, conversation_id, 'user', Message(author='user', content=prompt))
            await conn.commit()
        async with active_generations_lock:
            state = ActiveGeneration(conversation_id, user_record.sequence_id)
            active_generations[conversation_id] = state
    else:
        async with mk_conn() as conn:
            blocking_id = await get_blocking_message_id(conn, conversation_id)
            if blocking_id is not None:
                raise ConversationBlockedError(blocking_id)
            latest_sequence_id = await get_latest_sequence_id(conn, conversation_id)
        async with active_generations_lock:
            existing = active_generations.get(conversation_id)
            if existing is not None and existing.status == 'running':
                raise ActiveGenerationExistsError()
            state = ActiveGeneration(conversation_id, latest_sequence_id)
            active_generations[conversation_id] = state
        try:
            async with mk_conn() as conn:
                user_record = await insert_message(conn, conversation_id, 'user', Message(author='user', content=prompt))
                await conn.commit()
        except Exception:
            async with active_generations_lock:
                if active_generations.get(conversation_id) is state:
                    active_generations.pop(conversation_id, None)
            raise

    state.task = asyncio.create_task(_run_generation_task(
        state,
        functions,
        agent_id=agent_id,
        provider_key=provider_key,
        model_id=model_id,
        extra_scopes=extra_scopes,
    ))
    return conversation_id, user_record

async def get_messages_for_continuation(conn: AsyncConnection, conv_id: int) -> list[tuple[int, ConversationElement]]:
    ctx: list[tuple[int, ConversationElement]] = []
    async with conn.cursor() as cur:
        await cur.execute("SELECT id, role, elements FROM messages WHERE conversation_id = %s ORDER BY sequence_id ASC", (conv_id,))
        async for row in cur:
            message_id, role, element = row
            element_dict = json.loads(element) if isinstance(element, str) else element
            parsed = stored_element_adapter.validate_python(element_dict)
            ctx.append((message_id, parsed))
    return ctx

async def continue_conversation(conn: AsyncConnection, conv_id: int, functions: list[Tool], agent_id: Optional[str] = None, provider_key: Optional[str] = None, model_id: Optional[str] = None, extra_scopes: list[UserScopeChoice] = []) -> AsyncGenerator[StreamEvent, None]:
    provider_key, model_id, inference_config, scopes, agent_prompt = await _resolve_agent(agent_id, provider_key, model_id, extra_scopes)
    messages = await get_messages_for_continuation(conn, conv_id)

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
                        record = await insert_message(conn, conv_id, 'assistant', result)
                        await conn.commit()
                        messages.append((record.id, result))
                        yield ElementFinalized(id=record.id, sequence_id=record.sequence_id)
                    case FinishedToolCall(name=name, parameters=parameters):
                        assert isinstance(aggregated_element, FinishedToolCall)
                        # Pass the extracted scopes to the tool call
                        tool_result = await run_tool_call(name, parameters, False, scopes)
                        result_element = ToolCallFinishedOrBlocked(
                            name=name,
                            parameters=parameters,
                            result=tool_result.result,
                            is_blocking=tool_result.is_blocking,
                            status='pending' if tool_result.is_blocking else 'completed',
                            scopes=scopes
                        )
                        record = await insert_message(conn, conv_id, 'assistant', result_element)
                        yield ToolCallEvent(
                            name=name,
                            parameters=parameters,
                            result=tool_result.result,
                            is_blocking=tool_result.is_blocking,
                            status='pending' if tool_result.is_blocking else 'completed',
                        )
                        messages.append((record.id, result_element))
                        if tool_result.is_blocking:
                            async with conn.cursor() as cur:
                                await cur.execute(
                                    "UPDATE conversations SET blocking_message_id = %s WHERE id = %s",
                                    (record.id, conv_id),
                                )
                            run_next_loop = False
                        else:
                            run_next_loop = True
                        await conn.commit()
                        yield ElementFinalized(id=record.id, sequence_id=record.sequence_id)
            if delta is not None:
                if isinstance(delta, StreamingMessage):
                    yield MessageChunk(content=delta.content)
                elif isinstance(delta, StreamingThinking):
                    yield ThinkingChunk(content=delta.content)
    logger.info("Stream finished")

async def get_active_generation(conv_id: int) -> Optional[ActiveGeneration]:
    async with active_generations_lock:
        state = active_generations.get(conv_id)
        if state is not None and state.status == 'running':
            return state
        return None

async def _run_generation_task(
    state: ActiveGeneration,
    functions: list[Tool],
    agent_id: Optional[str] = None,
    provider_key: Optional[str] = None,
    model_id: Optional[str] = None,
    extra_scopes: list[UserScopeChoice] = [],
) -> None:
    try:
        async with mk_conn() as conn:
            async for event in continue_conversation(
                conn,
                state.conversation_id,
                functions,
                agent_id=agent_id,
                provider_key=provider_key,
                model_id=model_id,
                extra_scopes=extra_scopes,
            ):
                await state.apply_event(event)
        await state.finish()
    except asyncio.CancelledError:
        await state.cancel()
        raise
    except Exception as exc:
        logger.exception("Generation failed for conversation %s", state.conversation_id)
        await state.fail(str(exc))
    finally:
        async with active_generations_lock:
            if active_generations.get(state.conversation_id) is state:
                active_generations.pop(state.conversation_id, None)

async def start_generation(
    conv_id: int,
    functions: list[Tool],
    agent_id: Optional[str] = None,
    provider_key: Optional[str] = None,
    model_id: Optional[str] = None,
    extra_scopes: list[UserScopeChoice] = [],
) -> ActiveGeneration:
    async with mk_conn() as conn:
        latest_sequence_id = await get_latest_sequence_id(conn, conv_id)
    async with active_generations_lock:
        existing = active_generations.get(conv_id)
        if existing is not None and existing.status == 'running':
            raise ActiveGenerationExistsError()
        state = ActiveGeneration(conv_id, latest_sequence_id)
        active_generations[conv_id] = state
        state.task = asyncio.create_task(_run_generation_task(
            state,
            functions,
            agent_id=agent_id,
            provider_key=provider_key,
            model_id=model_id,
            extra_scopes=extra_scopes,
        ))
        return state

async def stream_conversation_events(
    conv_id: int,
    after_sequence_id: Optional[int] = None,
) -> AsyncGenerator[StreamEvent, None]:
    state = await get_active_generation(conv_id)

    if state is None:
        async with mk_conn() as conn:
            records = await get_messages_after(conn, conv_id, after_sequence_id)
        for record in records:
            for event in stored_record_to_stream_events(record):
                yield event
        return

    boundary_sequence_id, snapshot_events, queue = await state.subscribe()
    try:
        async with mk_conn() as conn:
            records = await get_messages_after(conn, conv_id, after_sequence_id, boundary_sequence_id)
        for record in records:
            for event in stored_record_to_stream_events(record):
                yield event
        for event in snapshot_events:
            yield event
        while True:
            queued_event = await queue.get()
            if queued_event is None:
                break
            yield queued_event
    finally:
        await state.unsubscribe(queue)

async def get_conversations(conn: AsyncConnection) -> list[dict[str, Any]]:
    async with conn.cursor() as cur:
        await cur.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
        rows = await cur.fetchall()
    return [{'id': r[0], 'title': r[1], 'created_at': r[2].isoformat()} for r in rows]

async def get_conversation_details(conn: AsyncConnection, conv_id: int) -> Optional[dict[str, Any]]:
    async with conn.cursor() as cur:
        await cur.execute("SELECT id, title, created_at, blocking_message_id FROM conversations WHERE id = %s", (conv_id,))
        row = await cur.fetchone()
        if not row:
            return None
        await cur.execute("SELECT id, sequence_id, role, elements, created_at FROM messages WHERE conversation_id = %s ORDER BY sequence_id ASC", (conv_id,))
        messages = []
        async for r in cur:
            elem_dict = json.loads(r[3]) if isinstance(r[3], str) else r[3]
            parsed = stored_element_adapter.validate_python(elem_dict)
            messages.append({'id': r[0], 'sequence_id': r[1], 'role': r[2], 'element': parsed.model_dump(), 'created_at': r[4].isoformat()})
        return {
            'id': row[0],
            'title': row[1],
            'created_at': row[2].isoformat(),
            'blocking_message_id': row[3],
            'messages': messages
        }

async def decide_tool_call(conn: AsyncConnection, conv_id: int, msg_id: int, decision: Literal['approve', 'reject'], comment: str = "") -> bool:
    async with conn.cursor() as cur:
        await conn.set_autocommit(False)
        await cur.execute('SELECT elements FROM messages WHERE id = %s AND conversation_id = %s', (msg_id, conv_id))
        row = await cur.fetchone()
        if not row:
            raise ValueError('message not found')
        
        elem_dict = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        elem = stored_element_adapter.validate_python(elem_dict)

        decision_elem = ToolCallDecision(
            decision=decision,
            original_message_id=msg_id,
            comment=comment or ""
        )
        
        await insert_message(conn, conv_id, 'assistant', decision_elem)
        await cur.execute('UPDATE conversations SET blocking_message_id = NULL WHERE id = %s', (conv_id,))

        executed = False
        if decision == 'approve':
            if not isinstance(elem, ToolCallFinishedOrBlocked):
                raise ValueError('original message is not a tool call')

            # FIXME: A new approach to this
            result = await run_tool_call(elem.name, elem.parameters, privileged=True, scopes=elem.scopes)
            result_elem = ToolCallResult(
                original_message_id=msg_id,
                result=result.result,
            )
            await insert_message(conn, conv_id, 'system', result_elem)
            executed = True
        
        await conn.commit()
        return executed

async def delete_conversation(conn: AsyncConnection, conv_id: int) -> bool:
    state = await get_active_generation(conv_id)
    if state is not None and state.task is not None:
        state.task.cancel()
        try:
            await state.task
        except asyncio.CancelledError:
            pass
        await state.cancel()
    async with conn.cursor() as cur:
        await conn.set_autocommit(False)
        await cur.execute("SELECT id FROM conversations WHERE id = %s", (conv_id,))
        if not await cur.fetchone():
            return False
        await cur.execute("DELETE FROM messages WHERE conversation_id = %s", (conv_id,))
        await cur.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))
        await conn.commit()
        return True

@router.post('/api/conversations/prompt')
async def prompt_model(payload: PromptRequest):
    try:
        conversation_id, user_record = await prepare_and_start_generation(
            payload.prompt,
            payload.conversation_id,
            TOOL_REGISTRY,
            agent_id=payload.agent_id,
            provider_key=payload.provider_key,
            model_id=payload.model_id,
            extra_scopes=payload.scopes,
        )
    except ConversationBlockedError as e:
        return JSONResponse(
            status_code=403,
            content={"error": "Action required", "blocking_message_id": e.blocking_message_id}
        )
    except ActiveGenerationExistsError:
        return JSONResponse(status_code=409, content={"error": "conversation is already generating"})

    async def stream():
        async for event in stream_conversation_events(conversation_id, user_record.sequence_id):
            yield serialize_ndjson_event(event)

    return StreamingResponse(
        stream(),
        media_type='application/x-ndjson', headers={'X-Conversation-ID': str(conversation_id)}
    )

@router.get('/api/conversations')
async def list_conversations():
    async with mk_conn() as conn:
        conversations = await get_conversations(conn)
    return JSONResponse(content=conversations)

@router.get('/api/conversations/{conv_id}')
async def get_conversation(conv_id: int):
    async with mk_conn() as conn:
        details = await get_conversation_details(conn, conv_id)
    if not details:
        return JSONResponse(status_code=404, content={'error': 'Not found'})
    return JSONResponse(content=details)

@router.get('/api/conversations/{conv_id}/stream')
async def stream_conversation_endpoint(conv_id: int, after_sequence_id: Optional[int] = None):
    async with mk_conn() as conn:
        details = await get_conversation_details(conn, conv_id)
        if not details:
            return JSONResponse(status_code=404, content={'error': 'conversation not found'})

    async def stream():
        async for event in stream_conversation_events(conv_id, after_sequence_id):
            yield serialize_ndjson_event(event)

    return StreamingResponse(
        stream(),
        media_type='application/x-ndjson', headers={'X-Conversation-ID': str(conv_id)}
    )

@router.delete('/api/conversations/{conv_id}')
async def delete_conversation_endpoint(conv_id: int):
    try:
        async with mk_conn() as conn:
            success = await delete_conversation(conn, conv_id)
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
        async with mk_conn() as conn:
            executed = await decide_tool_call(conn, conv_id, msg_id, decision, comment=comment)
        return JSONResponse(content={'status': 'success', 'executed': executed})
    except ValueError as e:
        return JSONResponse(status_code=400, content={'error': str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

@router.post('/api/conversations/{conversation_id}/continue')
async def continue_conversation_endpoint(conversation_id: int, payload: ContinueRequest):
    async with mk_conn() as conn:
        details = await get_conversation_details(conn, conversation_id)
        if not details:
            return JSONResponse(status_code=404, content={'error': 'conversation not found'})
        if details.get('blocking_message_id') is not None:
            return JSONResponse(
                status_code=403,
                content={"error": "Action required", "blocking_message_id": details.get('blocking_message_id')}
            )
        after_sequence_id = await get_latest_sequence_id(conn, conversation_id)

    try:
        await start_generation(
            conversation_id,
            TOOL_REGISTRY,
            agent_id=payload.agent_id,
            provider_key=payload.provider_key,
            model_id=payload.model_id,
            extra_scopes=payload.scopes,
        )
    except ActiveGenerationExistsError:
        return JSONResponse(status_code=409, content={"error": "conversation is already generating"})

    async def stream():
        async for event in stream_conversation_events(conversation_id, after_sequence_id):
            yield serialize_ndjson_event(event)

    return StreamingResponse(
        stream(),
        media_type='application/x-ndjson', headers={'X-Conversation-ID': str(conversation_id)}
    )
