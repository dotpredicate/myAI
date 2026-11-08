import datetime
from typing import Any, AsyncIterator, Optional, cast

import openai
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam
from openai.types.shared_params.function_definition import FunctionDefinition

from inference.engine import (
    StreamingMessage,
    StreamingThinking,
    StreamingToolCall,
    StreamingElement,
    FinishedMessage,
    FinishedThinking,
    FinishedToolCall,
    FinishedElement,
    ChatContext,
    InferenceProvider,
    InferenceParam,
    InferenceParamType,
    Model,
)

from domain import (
    Message,
    Thinking,
    ToolCall,
    ApprovalRequest,
    ToolCallResult,
    ToolCallDecision,
    InferenceProviderConfig,
)
from log_config import get_logger
from tools import Tool

logger = get_logger(__name__)

ENVIRONMENT_PROMPT = """## Environment
- You work in a Linux sandbox built with bubblewrap (bwrap). Shell commands
  run as `bash -c` and return stdout, stderr and the exit code.
- /workspace is your writable scratch directory for this conversation.
- /tmp is ephemeral; it is cleared between commands.
- Repositories are mounted under /repositories according to their policies:
  read-only cannot be changed, privileged-write only through an approved change,
  write can be modified directly.
- Only these system paths are available: /usr, /lib, /lib64, /bin, /proc, /dev.
- The host filesystem, including /home, is not visible.
- There is no network access, no GPU, no audio and no access to other host devices.
- Only system-installed binaries are available; you cannot install packages."""

def _build_system_prompt(context: ChatContext, today: Optional[datetime.date] = None) -> str:
    current_date = today or datetime.date.today()
    sections = [f"Current date: {current_date.isoformat()}", ENVIRONMENT_PROMPT]

    if context.scopes:
        repositories = "\n".join(
            f"- /repositories/{scope.internal_name} (policy: {scope.security_policy})"
            for scope in context.scopes
        )
        sections.append(f"Available repositories:\n{repositories}")

    if context.instructions and context.instructions.strip():
        sections.append(f"Agent instructions:\n{context.instructions.strip()}")

    return "\n\n".join(sections)


def _to_oai_messages(
    context: ChatContext,
    include_reasoning_content: bool = True,
) -> list[ChatCompletionMessageParam]:
    result: list[ChatCompletionMessageParam] = []
    system_content = _build_system_prompt(context)
    logger.debug(
        "Built system prompt with %d repository scopes and agent instructions=%s",
        len(context.scopes),
        bool(context.instructions and context.instructions.strip()),
    )
    result.append({'role': 'system', 'content': system_content})

    pending_thinking: Optional[str] = None

    blocked_ids = {
        element.original_message_id
        for _, element in context.messages
        if isinstance(element, ApprovalRequest)
    }

    for message_id, element in context.messages:
        match element:
            case Thinking():
                pending_thinking = element.content

            case Message():
                msg: dict[str, object] = {'role': element.author, 'content': element.content}
                if element.author == 'assistant' and pending_thinking is not None:
                    if include_reasoning_content:
                        msg['reasoning_content'] = pending_thinking
                    pending_thinking = None
                    logger.debug("Added thinking: %s", msg)
                result.append(cast(ChatCompletionMessageParam, msg))

            case ToolCall():
                tc: dict[str, object] = {
                    'role': 'assistant',
                    'tool_calls': [{
                        'id': str(message_id),
                        'type': 'function',
                        'function': {
                            'name': element.name,
                            'arguments': element.parameters,
                        },
                    }]
                }
                if pending_thinking is not None:
                    if include_reasoning_content:
                        tc['reasoning_content'] = pending_thinking
                        tc['content'] = ''
                    pending_thinking = None
                result.append(cast(ChatCompletionMessageParam, tc))
                if message_id not in blocked_ids:
                    result.append({
                        'role': 'tool',
                        'tool_call_id': str(message_id),
                        'content': str(element.result)
                    })

            case ApprovalRequest():
                pass

            case ToolCallResult():
                pending_thinking = None  # drop trailing thinking, defensive
                result.append({
                    'role': 'tool',
                    'tool_call_id': str(element.original_message_id),
                    'content': str(element.result)
                })

            case ToolCallDecision():
                if element.decision == 'reject':
                    pending_thinking = None  # drop trailing thinking, defensive
                    comment_text = element.comment
                    if comment_text:
                        content = f"User rejected this tool call with comment: {comment_text}".strip()
                    else:
                        content = "User rejected this tool call"
                    result.append({
                        'role': 'tool',
                        'tool_call_id': str(element.original_message_id),
                        'content': content
                    })
                else:
                    # approve decisions: explicitly ignored, drop pending thinking if any
                    pending_thinking = None

    # trailing thinking with no following element is silently dropped
    return result

def _to_oai_tools(tools: list[Tool]) -> list[ChatCompletionToolUnionParam]:
    return [{"type": "function", "function": cast(FunctionDefinition, t["schema"])} for t in tools]

class DeltaProcessor:
    def __init__(self) -> None:
        self.buffered_element: Optional[StreamingElement] = None
        self.tool_calls: dict[int, tuple[Optional[str], str]] = {}
        self.finish_reason: Optional[str] = None
        self.has_content = False
        self.has_tool_call = False

    def _finalize_buffered(self) -> Optional[FinishedElement]:
        element = self.buffered_element
        self.buffered_element = None
        if element is None:
            return None
        match element:
            case StreamingMessage(content=c):
                return FinishedMessage(content=c)
            case StreamingThinking(content=c):
                return FinishedThinking(content=c)
            case StreamingToolCall(name=n, parameters=p):
                if n is None:
                    return None
                return FinishedToolCall(name=n, parameters=p or "")
        return None

    def _finalize_tool_calls(self) -> list[FinishedElement]:
        elements: list[FinishedElement] = []
        for index in sorted(self.tool_calls):
            name, arguments = self.tool_calls[index]
            if name is None:
                logger.warning("Dropping tool call without a name")
                continue
            elements.append(FinishedToolCall(name=name, parameters=arguments))
        self.tool_calls.clear()
        return elements

    def process(self, chunk: ChatCompletionChunk) -> tuple[Optional[StreamingElement], list[FinishedElement]]:
        if not chunk.choices:
            return None, []
        choice = chunk.choices[0]
        if choice.finish_reason is not None:
            self.finish_reason = choice.finish_reason

        delta = choice.delta
        finalized: list[FinishedElement] = []

        if delta.tool_calls:
            self.has_tool_call = True
            buffered = self._finalize_buffered()
            if buffered is not None:
                finalized.append(buffered)
            for call in delta.tool_calls:
                index = getattr(call, 'index', 0)
                if index is None:
                    index = 0
                name, arguments = self.tool_calls.get(index, (None, ""))
                function = getattr(call, 'function', None)
                if function is not None:
                    if function.name:
                        name = function.name
                    arguments += function.arguments or ""
                self.tool_calls[index] = (name, arguments)

        new_elem: Optional[StreamingElement] = None
        content = delta.content
        reasoning = getattr(delta, 'reasoning_content', None)
        if content is not None and (content != "" or reasoning is None):
            new_elem = StreamingMessage(content)
            if content.strip():
                self.has_content = True
        elif reasoning is not None:
            new_elem = StreamingThinking(reasoning)

        if new_elem is not None and self.tool_calls:
            finalized.extend(self._finalize_tool_calls())

        if new_elem is not None:
            match (self.buffered_element, new_elem):
                case (None, elem):
                    self.buffered_element = elem
                case (StreamingMessage(c1), StreamingMessage(c2)):
                    self.buffered_element = StreamingMessage(c1 + c2)
                case (StreamingThinking(c1), StreamingThinking(c2)):
                    self.buffered_element = StreamingThinking(c1 + c2)
                case _:
                    buffered = self._finalize_buffered()
                    if buffered is not None:
                        finalized.append(buffered)
                    self.buffered_element = new_elem
        return new_elem, finalized

    def flush(self) -> list[FinishedElement]:
        elements: list[FinishedElement] = []
        buffered = self._finalize_buffered()
        if buffered is not None:
            elements.append(buffered)
        elements.extend(self._finalize_tool_calls())
        return elements


async def stream_events(
    raw_stream: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[tuple[Optional[StreamingElement], Optional[FinishedElement]]]:
    processor = DeltaProcessor()
    async for chunk in raw_stream:
        streaming, finished = processor.process(chunk)
        for element in finished:
            yield None, element
        if streaming is not None:
            yield streaming, None

    finalized = processor.flush()
    truncated = processor.finish_reason in ('length', 'content_filter')
    if truncated:
        finalized = [element for element in finalized if not isinstance(element, FinishedToolCall)]
    for element in finalized:
        yield None, element
    if truncated:
        raise ValueError(f"Provider stopped with finish_reason={processor.finish_reason}")
    if processor.finish_reason == 'stop' and not processor.has_content and not processor.has_tool_call:
        if not any(isinstance(element, FinishedThinking) for element in finalized):
            raise ValueError("Provider returned an empty completion")


class OpenAICompatibleProvider(InferenceProvider):
    def __init__(
        self,
        config: InferenceProviderConfig,
        client: Optional[openai.AsyncOpenAI] = None,
    ) -> None:
        self.config = config
        self._client = client

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key or "dummy",
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                max_retries=0,
                default_headers=self.config.headers or None,
            )
        return self._client

    async def run_chat_completion_stream(
        self,
        model_id: str,
        inference_config: dict[str, Any],
        context: ChatContext,
    ) -> AsyncIterator[tuple[Optional[StreamingElement], Optional[FinishedElement]]]:
        params: dict[str, Any] = {
            "model": model_id,
            "messages": _to_oai_messages(
                context,
                include_reasoning_content=self.config.include_reasoning_content,
            ),
            "stream": True,
        }
        if context.tools:
            params["tools"] = _to_oai_tools(context.tools)
            params["parallel_tool_calls"] = False
        temperature = inference_config.get("temperature")
        if temperature is not None:
            params["temperature"] = temperature
        if self.config.extra_body:
            params["extra_body"] = self.config.extra_body

        raw_stream = await self.client.chat.completions.create(**params)
        async for event in stream_events(raw_stream):
            yield event

    async def list_models(self) -> list[Model]:
        try:
            page = await self.client.models.list()
            models = [
                Model(
                    id=model.id,
                    created=getattr(model, "created", 0) or 0,
                    owned_by=getattr(model, "owned_by", self.config.key) or self.config.key,
                )
                for model in page.data
            ]
            if models:
                return models
        except Exception as exc:
            logger.warning("Failed to list models for provider %s: %s", self.config.key, exc)
        return [Model(id=model, created=0, owned_by=self.config.key) for model in self.config.models]

    def get_inference_params(self) -> list[InferenceParam]:
        return [
            InferenceParam(
                name="temperature",
                type=InferenceParamType.FLOAT,
                default=None,
                min=0.0,
                max=2.0,
                step=0.05,
                description="Sampling temperature",
            ),
        ]
