import os
import openai
from typing import Optional, Any, List

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from inference.engine import *

from domain import (
    Message,
    Thinking,
    ToolCallFinishedOrBlocked,
    ToolCallResult,
    ToolCallDecision,
    ConversationElement,
)

# LLM Configuration
LLAMA_CPP_ENDPOINT = os.getenv('LLAMA_CPP_ENDPOINT', 'http://localhost:1234')
completions_endpoint = openai.Client(api_key='dummy', base_url=LLAMA_CPP_ENDPOINT)


def _to_oai_messages(elements: list[tuple[int, ConversationElement]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    pending_thinking: Optional[str] = None

    for message_id, element in elements:
        match element:
            case Thinking():
                pending_thinking = element.content

            case Message():
                msg: dict[str, Any] = {'role': element.author, 'content': element.content}
                if element.author == 'assistant' and pending_thinking is not None:
                    msg['reasoning_content'] = pending_thinking
                    pending_thinking = None
                    print(f'[DEBUG]: Added thinking: {msg}')
                result.append(msg)

            case ToolCallFinishedOrBlocked():
                tc: dict[str, Any] = {
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
                result.append(tc)
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


class DeltaProcessor:
    def __init__(self):
        self.buffered_element: Optional[StreamingElement] = None

    def process(self, chunk: ChatCompletionChunk) -> tuple[Optional[StreamingElement], Optional[FinishedElement]]:
        if not chunk.choices:
            return None, None
        delta = chunk.choices[0].delta
        new_elem: Optional[StreamingElement] = None
        if delta.tool_calls:
            func = delta.tool_calls[0].function
            new_elem = StreamingToolCall(func.name, func.arguments)
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
                self.buffered_element = StreamingToolCall(n1, p1 + p2)
            case _:
                if self.buffered_element:
                    print(f'Finalized element: {self.buffered_element}')
                    match self.buffered_element:
                        case StreamingMessage(content=c):
                            finalized = FinishedMessage(content=c)
                        case StreamingThinking(content=c):
                            finalized = FinishedThinking(content=c)
                        case StreamingToolCall(name=n1, parameters=p):
                            finalized = FinishedToolCall(name=n1, parameters=p)
                self.buffered_element = new_elem
        return new_elem, finalized


def run_chat_completion_stream(model_id: str, context: list[tuple[int, ConversationElement]], functions: List[Any]):
    return completions_endpoint.chat.completions.create(
        model=model_id,
        messages=_to_oai_messages(context),
        stream=True,
        tools=functions,
    )


def list_models():
    return completions_endpoint.models.list()