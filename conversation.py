from typing import AsyncGenerator, Literal, Optional, Any, Union, cast
import asyncio
import json
from pydantic import BaseModel
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from domain import (
    AgentConfig,
    Message,
    Thinking,
    ToolCall,
    ApprovalRequest,
    ToolCallResult,
    ToolCallDecision,
    ConversationElement,
    ScopeSpec,
    stored_element_adapter,
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
    EmptyCompletionError,
    ChatContext,
    registry
)
from tools import FunctionDefinition, Tool, ToolContext, run_tool_call, TOOL_REGISTRY
import system

from log_config import get_logger
from database import AsyncConnection, mk_conn
from agents import get_agent_by_name
from projects import get_project_by_name, resolve_project_id
from scopes import UserScopeChoice, resolve_scopes, scopes_for_agent

logger = get_logger(__name__)

router = APIRouter()


class GenerationRequest(BaseModel):
    agent_id: Optional[str] = None
    project_id: Optional[str] = None
    provider_key: Optional[str] = None
    model_id: Optional[str] = None
    inference_config: dict[str, Any] = {}
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

class ApprovalRequestEvent(BaseModel):
    type: Literal['approval_request'] = 'approval_request'
    original_message_id: int

class ToolCallDecisionEvent(BaseModel):
    type: Literal['tool_call_decision'] = 'tool_call_decision'
    original_message_id: int
    decision: Literal['approve', 'reject']
    comment: Optional[str] = None

class ToolCallResultEvent(BaseModel):
    type: Literal['tool_call_result'] = 'tool_call_result'
    original_message_id: int
    result: str

class ElementFinalized(BaseModel):
    type: Literal['finalized'] = 'finalized'
    id: int
    sequence_id: int

class GenerationErrorEvent(BaseModel):
    type: Literal['generation_error'] = 'generation_error'
    message: str

StreamEvent = Union[MessageChunk, ThinkingChunk, ElementFinalized, ToolCallEvent, ApprovalRequestEvent, ToolCallDecisionEvent, ToolCallResultEvent, GenerationErrorEvent]

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
            elif isinstance(event, ApprovalRequestEvent):
                self.buffered_element = None
                self.pending_events.append(event)
                self.status = 'blocked'
            elif isinstance(event, ToolCallDecisionEvent):
                self.buffered_element = None
                self.pending_events.append(event)
            elif isinstance(event, ToolCallResultEvent):
                self.buffered_element = None
                self.pending_events.append(event)
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
        case ToolCall(name=name, parameters=parameters, result=result):
            return [
                ToolCallEvent(
                    name=name,
                    parameters=parameters,
                    result=result,
                ),
                finalized,
            ]
        case ApprovalRequest(original_message_id=original_message_id):
            return [
                ApprovalRequestEvent(original_message_id=original_message_id),
                finalized,
            ]
        case ToolCallDecision(original_message_id=original_message_id, decision=decision, comment=comment):
            return [
                ToolCallDecisionEvent(
                    original_message_id=original_message_id,
                    decision=decision,
                    comment=comment,
                ),
                finalized,
            ]
        case ToolCallResult(original_message_id=original_message_id, result=result):
            return [
                ToolCallResultEvent(
                    original_message_id=original_message_id,
                    result=result,
                ),
                finalized,
            ]
        case _:
            return [finalized]

def serialize_ndjson_event(event: StreamEvent) -> str:
    return event.model_dump_json() + '\n'

async def create_conversation(conn: AsyncConnection, project_id: Optional[int] = None) -> int:
    async with conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO conversations (project_id) VALUES (%s) RETURNING id",
            (project_id,),
        )
        row = await cur.fetchone()
        if not row:
            raise RuntimeError("Failed to create conversation")
        conv_id = row[0]
        return conv_id


async def get_conversation_project(conn: AsyncConnection, conv_id: int) -> tuple[Optional[int], Optional[str]]:
    """Return the conversation's project id and internal name, if any."""
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT c.project_id, p.internal_name
            FROM conversations c
            LEFT JOIN projects p ON p.id = c.project_id
            WHERE c.id = %s
            """,
            (conv_id,),
        )
        row = await cur.fetchone()
        return (row[0], row[1]) if row else (None, None)


async def get_latest_sequence_id(conn: AsyncConnection, conv_id: int) -> Optional[int]:
    async with conn.cursor() as cur:
        await cur.execute("SELECT MAX(sequence_id) FROM messages WHERE conversation_id = %s", (conv_id,))
        row = await cur.fetchone()
        return row[0] if row and row[0] is not None else None

async def _allocate_next_sequence_id(conn: AsyncConnection, conv_id: int) -> int:
    async with conn.cursor() as cur:
        await cur.execute("SELECT id FROM conversations WHERE id = %s", (conv_id,))
        if not await cur.fetchone():
            raise ValueError('conversation not found')
    latest = await get_latest_sequence_id(conn, conv_id)
    return 1 if latest is None else latest + 1

async def insert_message(conn: AsyncConnection, conv_id: int, role: str, element: ConversationElement) -> StoredMessageRecord:
    await conn.begin_immediate()
    sequence_id = await _allocate_next_sequence_id(conn, conv_id)
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO messages (conversation_id, sequence_id, role, elements, created_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
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


async def _resolve_agent(
    agent_id: Optional[str],
    fallback_provider: Optional[str],
    fallback_model: Optional[str],
    req_scopes: list[UserScopeChoice],
    inference_config: dict[str, Any] = {},
    resolved_scopes: Optional[list[ScopeSpec]] = None,
) -> tuple:
    """Resolve provider, model, scopes, instructions and the agent config."""
    if agent_id:
        agent = await get_agent_by_name(agent_id)
        if agent is None:
            raise ValueError(f"Agent '{agent_id}' not found")
        scopes = resolved_scopes if resolved_scopes is not None else await scopes_for_agent(agent, req_scopes)
        return agent.provider_key, agent.model_id, agent.inference_config, scopes, agent.instructions, agent
    if not fallback_provider:
        raise ValueError('provider_key required')
    if not fallback_model:
        raise ValueError('model_id required')
    scopes = resolved_scopes if resolved_scopes is not None else await resolve_scopes(None, req_scopes)
    return fallback_provider, fallback_model, inference_config, scopes, None, None


def tools_for_agent(agent: Optional[AgentConfig]) -> list[Tool]:
    """Expose spawn_agent only to agents with an allowlist, with that list in the description."""
    filtered = [tool for tool in TOOL_REGISTRY if tool["name"] != "spawn_agent"]
    if agent is None or not agent.spawnable_agents:
        return filtered
    spawn = next(tool for tool in TOOL_REGISTRY if tool["name"] == "spawn_agent")
    schema = cast(FunctionDefinition, dict(spawn["schema"]))
    schema["description"] = (
        f"{schema['description']} Allowed agents: {', '.join(agent.spawnable_agents)}."
    )
    filtered.append(Tool(name=spawn["name"], schema=schema, executor=spawn["executor"]))
    return filtered

async def prepare_and_start_generation(
    prompt: str,
    conversation_id: Optional[int],
    agent_id: Optional[str] = None,
    project_id: Optional[str] = None,
    provider_key: Optional[str] = None,
    model_id: Optional[str] = None,
    extra_scopes: list[UserScopeChoice] = [],
    inference_config: dict[str, Any] = {},
) -> tuple[int, StoredMessageRecord]:
    if conversation_id is None:
        numeric_project_id: Optional[int] = None
        if project_id is not None:
            numeric_project_id = await resolve_project_id(project_id)
            if numeric_project_id is None:
                raise ValueError(f"Project '{project_id}' not found")
        async with mk_conn() as conn:
            conversation_id = await create_conversation(conn, numeric_project_id)
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
            _, stored_project_name = await get_conversation_project(conn, conversation_id)
        if project_id is not None and project_id != stored_project_name:
            raise ValueError('a conversation cannot change its project')
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
        agent_id=agent_id,
        provider_key=provider_key,
        model_id=model_id,
        extra_scopes=extra_scopes,
        inference_config=inference_config,
    ))
    return conversation_id, user_record

async def get_messages_for_continuation(conv_id: int) -> list[tuple[int, ConversationElement]]:
    ctx: list[tuple[int, ConversationElement]] = []
    async with mk_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, role, elements FROM messages WHERE conversation_id = %s ORDER BY sequence_id ASC", (conv_id,))
            async for row in cur:
                message_id, role, element = row
                element_dict = json.loads(element) if isinstance(element, str) else element
                parsed = stored_element_adapter.validate_python(element_dict)
                ctx.append((message_id, parsed))
    return ctx


async def _insert_message(conv_id: int, role: str, element: ConversationElement) -> StoredMessageRecord:
    async with mk_conn() as conn:
        return await insert_message(conn, conv_id, role, element)


async def _insert_blocking_tool_call(
    conv_id: int,
    element: ToolCall,
) -> tuple[StoredMessageRecord, StoredMessageRecord]:
    """Persist the tool call, its approval request and the block atomically."""
    async with mk_conn() as conn:
        tool_record = await insert_message(
            conn,
            conv_id,
            'assistant',
            element,
        )
        approval_record = await insert_message(
            conn,
            conv_id,
            'assistant',
            ApprovalRequest(original_message_id=tool_record.id),
        )
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE conversations SET blocking_message_id = %s WHERE id = %s",
                (tool_record.id, conv_id),
            )
        return tool_record, approval_record


async def prompt_conversation(
        conv_id: int,
        agent_id: Optional[str] = None, provider_key: Optional[str] = None, model_id: Optional[str] = None,
        extra_scopes: list[UserScopeChoice] = [],
        inference_config: dict[str, Any] = {},
        resolved_scopes: Optional[list[ScopeSpec]] = None,
        spawn_depth: int = 0,
    ) -> AsyncGenerator[StreamEvent, None]:
    provider_key, model_id, inference_config, scopes, agent_prompt, agent = await _resolve_agent(
        agent_id, provider_key, model_id, extra_scopes, inference_config, resolved_scopes
    )
    async with mk_conn() as conn:
        _, project_name = await get_conversation_project(conn, conv_id)
    project = await get_project_by_name(project_name) if project_name else None
    functions = tools_for_agent(agent)
    messages = await get_messages_for_continuation(conv_id)
    tool_context = ToolContext(
        conversation_id=conv_id,
        scopes=scopes,
        agent=agent,
        spawn_depth=spawn_depth,
    )

    try:
        provider = registry.get(provider_key)
    except KeyError as exc:
        raise ValueError(f"Provider '{provider_key}' is not available") from exc

    run_next_loop = True
    attempt = 0
    while run_next_loop:
        chat_context = ChatContext(
            messages=messages,
            scopes=scopes,
            tools=functions,
            instructions=agent_prompt,
            project_instructions=project.instructions if project else None,
        )
        chat_gen_inner = provider.run_chat_completion_stream(model_id, inference_config, chat_context)
        run_next_loop = False
        emitted = False
        try:
            async for delta, aggregated_element in chat_gen_inner:
                if aggregated_element is not None:
                    match aggregated_element:
                        case FinishedMessage() | FinishedThinking():
                            result = to_conv_elem(aggregated_element)
                            record = await _insert_message(conv_id, 'assistant', result)
                            messages.append((record.id, result))
                            emitted = True
                            yield ElementFinalized(id=record.id, sequence_id=record.sequence_id)
                        case FinishedToolCall(name=name, parameters=parameters):
                            assert isinstance(aggregated_element, FinishedToolCall)
                            tool_result = await run_tool_call(name, parameters, tool_context)
                            result_element = ToolCall(
                                name=name,
                                parameters=parameters,
                                result=tool_result.result,
                                scopes=scopes
                            )
                            if tool_result.is_blocking:
                                record, approval_record = await _insert_blocking_tool_call(conv_id, result_element)
                                emitted = True
                                yield ToolCallEvent(
                                    name=name,
                                    parameters=parameters,
                                    result=tool_result.result,
                                )
                                messages.append((record.id, result_element))
                                yield ElementFinalized(id=record.id, sequence_id=record.sequence_id)
                                approval_element = ApprovalRequest(original_message_id=record.id)
                                messages.append((approval_record.id, approval_element))
                                yield ApprovalRequestEvent(original_message_id=record.id)
                                yield ElementFinalized(id=approval_record.id, sequence_id=approval_record.sequence_id)
                                run_next_loop = False
                                aclose = getattr(chat_gen_inner, "aclose", None)
                                if aclose is not None:
                                    await aclose()
                                break
                            else:
                                record = await _insert_message(conv_id, 'assistant', result_element)
                                emitted = True
                                yield ToolCallEvent(
                                    name=name,
                                    parameters=parameters,
                                    result=tool_result.result,
                                )
                                messages.append((record.id, result_element))
                                yield ElementFinalized(id=record.id, sequence_id=record.sequence_id)
                                run_next_loop = True
                if delta is not None:
                    if isinstance(delta, StreamingMessage):
                        emitted = True
                        yield MessageChunk(content=delta.content)
                    elif isinstance(delta, StreamingThinking):
                        emitted = True
                        yield ThinkingChunk(content=delta.content)
        except EmptyCompletionError:
            # Some providers end a stream with finish_reason=stop and no content
            # or tool call. Retry once, but never repeat a stream that emitted
            # anything: tool execution already happened.
            if attempt == 0 and not emitted:
                attempt += 1
                logger.warning("Empty completion for conversation %s; retrying once", conv_id)
                run_next_loop = True
                continue
            raise
    logger.info("Stream finished")

async def get_active_generation(conv_id: int) -> Optional[ActiveGeneration]:
    async with active_generations_lock:
        state = active_generations.get(conv_id)
        if state is not None and state.status == 'running':
            return state
        return None

async def _run_generation_task(
    state: ActiveGeneration,
    agent_id: Optional[str] = None,
    provider_key: Optional[str] = None,
    model_id: Optional[str] = None,
    extra_scopes: list[UserScopeChoice] = [],
    inference_config: dict[str, Any] = {},
) -> None:
    try:
        async for event in prompt_conversation(
            state.conversation_id,
            agent_id=agent_id,
            provider_key=provider_key,
            model_id=model_id,
            extra_scopes=extra_scopes,
            inference_config=inference_config,
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
    agent_id: Optional[str] = None,
    provider_key: Optional[str] = None,
    model_id: Optional[str] = None,
    extra_scopes: list[UserScopeChoice] = [],
    inference_config: dict[str, Any] = {},
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
            agent_id=agent_id,
            provider_key=provider_key,
            model_id=model_id,
            extra_scopes=extra_scopes,
            inference_config=inference_config,
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
        await cur.execute(
            """
            SELECT c.id, c.title, c.created_at, c.blocking_message_id, p.internal_name
            FROM conversations c
            LEFT JOIN projects p ON p.id = c.project_id
            WHERE c.id = %s
            """,
            (conv_id,),
        )
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
            'project_id': row[4],
            'messages': messages
        }

async def decide_tool_call(conv_id: int, msg_id: int, decision: Literal['approve', 'reject'], comment: str = "") -> Optional[str]:
    """Returns the execution result on approve, None on reject."""
    async with mk_conn() as conn:
        await conn.begin_immediate()
        async with conn.cursor() as cur:
            await cur.execute(
                'SELECT messages.elements, conversations.blocking_message_id '
                'FROM messages JOIN conversations ON conversations.id = messages.conversation_id '
                'WHERE messages.id = %s AND messages.conversation_id = %s',
                (msg_id, conv_id),
            )
            row = await cur.fetchone()
            if not row:
                raise ValueError('message not found')
        
            elem_dict, blocking_message_id = row
            if blocking_message_id != msg_id:
                # Approving twice (or approving a stale message after reload)
                # must not execute the same privileged tool again.
                raise ValueError('tool call is not the current blocking message')

            elem = stored_element_adapter.validate_python(
                json.loads(elem_dict) if isinstance(elem_dict, str) else elem_dict
            )
            if decision == 'approve' and not isinstance(elem, ToolCall):
                raise ValueError('original message is not a tool call')

            # Atomic gate: exactly one caller clears the block. Everyone else
            # sees no row and gets a ValueError instead of executing the tool.
            await cur.execute(
                'UPDATE conversations SET blocking_message_id = NULL '
                'WHERE id = %s AND blocking_message_id = %s RETURNING id',
                (conv_id, msg_id),
            )
            if await cur.fetchone() is None:
                raise ValueError('tool call is not the current blocking message')

            decision_elem = ToolCallDecision(
                decision=decision,
                original_message_id=msg_id,
                comment=comment or "",
            )
            await insert_message(conn, conv_id, 'assistant', decision_elem)

    if decision == 'reject':
        return None

    assert isinstance(elem, ToolCall)
    interrupted: BaseException | None = None
    try:
        result = await run_tool_call(
            elem.name,
            elem.parameters,
            ToolContext(
                conversation_id=conv_id,
                privileged=True,
                scopes=elem.scopes,
            ),
        )
        result_text = result.result
    except BaseException as exc:
        result_text = json.dumps({"error": f"Tool execution failed: {exc}"})
        if isinstance(exc, Exception):
            logger.exception("Tool execution failed for conversation %s", conv_id)
        else:
            logger.warning("Tool execution interrupted for conversation %s: %s", conv_id, exc)
            interrupted = exc

    async with mk_conn() as conn:
        await insert_message(
            conn,
            conv_id,
            'system',
            ToolCallResult(original_message_id=msg_id, result=result_text),
        )
        
    if interrupted is not None:
        raise interrupted
    return result_text


async def cancel_generation(conv_id: int) -> bool:
    """Stop a running generation while keeping its transcript and workspace."""
    state = await get_active_generation(conv_id)
    if state is None:
        return False

    cancelled = False
    if state.task is not None and not state.task.done():
        state.task.cancel()
        cancelled = True
        try:
            await state.task
        except asyncio.CancelledError:
            pass
    if state.status == 'running':
        await state.cancel()
        cancelled = True
    return cancelled


async def delete_conversation(conv_id: int) -> bool:
    await cancel_generation(conv_id)
    async with mk_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM conversations WHERE id = %s RETURNING id", (conv_id,))
            if not await cur.fetchone():
                return False
    system.remove_workspace(conv_id)
    return True

@router.post('/api/conversations/prompt')
async def prompt_model(payload: PromptRequest):
    try:
        conversation_id, user_record = await prepare_and_start_generation(
            payload.prompt,
            payload.conversation_id,
            agent_id=payload.agent_id,
            project_id=payload.project_id,
            provider_key=payload.provider_key,
            model_id=payload.model_id,
            extra_scopes=payload.scopes,
            inference_config=payload.inference_config,
        )
    except ConversationBlockedError as e:
        return JSONResponse(
            status_code=403,
            content={"error": "Action required", "blocking_message_id": e.blocking_message_id}
        )
    except ActiveGenerationExistsError:
        return JSONResponse(status_code=409, content={"error": "conversation is already generating"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

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

@router.post('/api/conversations/{conv_id}/cancel')
async def cancel_conversation_endpoint(conv_id: int):
    async with mk_conn() as conn:
        details = await get_conversation_details(conn, conv_id)
        if not details:
            return JSONResponse(status_code=404, content={'error': 'conversation not found'})
    if not await cancel_generation(conv_id):
        return JSONResponse(status_code=409, content={'error': 'conversation is not generating'})
    return JSONResponse(content={'status': 'cancelled', 'id': conv_id})


@router.delete('/api/conversations/{conv_id}')
async def delete_conversation_endpoint(conv_id: int):
    try:
        success = await delete_conversation(conv_id)
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
        result = await decide_tool_call(conv_id, msg_id, decision, comment=comment)
        return JSONResponse(content={'status': 'success', 'executed': result is not None, 'result': result})
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

    project_id = details.get('project_id')
    if payload.project_id is not None and payload.project_id != project_id:
        return JSONResponse(status_code=400, content={"error": "a conversation cannot change its project"})

    try:
        await start_generation(
            conversation_id,
            agent_id=payload.agent_id,
            provider_key=payload.provider_key,
            model_id=payload.model_id,
            extra_scopes=payload.scopes,
            inference_config=payload.inference_config,
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
