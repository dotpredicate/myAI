import datetime
from typing import Optional, cast

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam
from openai.types.shared_params.function_definition import FunctionDefinition

from inference import (
    StreamingMessage,
    StreamingThinking,
    StreamingToolCall,
    StreamingElement,
    FinishedMessage,
    FinishedThinking,
    FinishedToolCall,
    FinishedElement,
    ChatContext
)

from domain import (
    Message,
    Thinking,
    ToolCallFinishedOrBlocked,
    ToolCallResult,
    ToolCallDecision,
)
from log_config import get_logger
from tools import Tool

logger = get_logger(__name__)

def _to_oai_messages(context: ChatContext) -> list[ChatCompletionMessageParam]:
    result: list[ChatCompletionMessageParam] = []

    system_content = f"""
    The date is {datetime.date.today()}.
    """

    # Build system prompt from scopes
    if context.scopes:
        scopes_content = f"""
        You have access to following repositories:
        {'\n'.join(f"- /repositories/{s.internal_name} - {s.security_policy}" for s in context.scopes)}
        """
        system_content += "\n" + scopes_content
        logger.debug(system_content)
    
    if context.instructions:
        instructions_content = f"""
        Additional instructions:
        {context.instructions}
        """
        system_content += "\n" + context.instructions
        logger.debug(instructions_content)

    logger.debug(system_content)
    result.append({'role': 'system', 'content': system_content})


    pending_thinking: Optional[str] = None

    for message_id, element in context.messages:
        match element:
            case Thinking():
                pending_thinking = element.content

            case Message():
                msg: dict[str, object] = {'role': element.author, 'content': element.content}
                if element.author == 'assistant' and pending_thinking is not None:
                    msg['reasoning_content'] = pending_thinking
                    pending_thinking = None
                    logger.debug("Added thinking: %s", msg)
                result.append(cast(ChatCompletionMessageParam, msg))

            case ToolCallFinishedOrBlocked():
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
                    tc['reasoning_content'] = pending_thinking
                    tc['content'] = ''
                    pending_thinking = None
                result.append(cast(ChatCompletionMessageParam, tc))
                if element.status == 'completed':
                    result.append({
                        'role': 'tool',
                        'tool_call_id': str(message_id),
                        'content': str(element.result)
                    })

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

    def process(self, chunk: ChatCompletionChunk) -> tuple[Optional[StreamingElement], Optional[FinishedElement]]:
        if not chunk.choices:
            return None, None
        delta = chunk.choices[0].delta
        new_elem: Optional[StreamingElement] = None
        if delta.tool_calls:
            func = delta.tool_calls[0].function
            if func:
                name = func.name
                arguments = func.arguments or ""
                new_elem = StreamingToolCall(name, arguments)
        elif hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
             new_elem = StreamingThinking(delta.reasoning_content)
        elif delta.content is not None:
            new_elem = StreamingMessage(delta.content)

        finalized: Optional[FinishedElement] = None
        match (self.buffered_element, new_elem):
            case (None, elem):
                self.buffered_element = elem
            case (StreamingMessage(c1), StreamingMessage(c2)):
                self.buffered_element = StreamingMessage(c1 + c2)
            case (StreamingThinking(c1), StreamingThinking(c2)):
                self.buffered_element = StreamingThinking(c1 + c2)
            case (StreamingToolCall(n1, p1), StreamingToolCall(n2, p2)) if n2 is None:
                self.buffered_element = StreamingToolCall(n1, (p1 or "") + (p2 or ""))
            case _:
                if self.buffered_element:
                    logger.debug("Finalized element: %s", self.buffered_element)
                    match self.buffered_element:
                        case StreamingMessage(content=c):
                            finalized = FinishedMessage(content=c)
                        case StreamingThinking(content=c):
                            finalized = FinishedThinking(content=c)
                        case StreamingToolCall(name=n1, parameters=p):
                            finalized = FinishedToolCall(name=n1 or "", parameters=p or "")
                self.buffered_element = new_elem
        return new_elem, finalized


# TODO: This will become OpenAICompatibleProvider extending InferenceProvider
# def run_chat_completion_stream(model_id: str, context: list[tuple[int, ConversationElement]], functions: List[object]):
#     return completions_endpoint.chat.completions.create(
#         model=model_id,
#         messages=_to_oai_messages(context),
#         stream=True,
#         tools=functions,
#     )
#
#
# def list_models():
#     return completions_endpoint.models.list()