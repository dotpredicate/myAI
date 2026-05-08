import os
import openai
from dataclasses import dataclass
from typing import Dict, Optional, Any, Union, List

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

# LLM Configuration
LLAMA_CPP_ENDPOINT = os.getenv('LLAMA_CPP_ENDPOINT', 'http://localhost:1234')
completions_endpoint = openai.Client(api_key='dummy', base_url=LLAMA_CPP_ENDPOINT)

@dataclass(frozen=True)
class StreamingMessage:
    content: str

@dataclass(frozen=True)
class StreamingThinking:
    content: str

@dataclass(frozen=True)
class StreamingToolCall:
    name: str
    parameters: str

StreamingElement = Union[StreamingMessage, StreamingThinking, StreamingToolCall]

@dataclass(frozen=True)
class FinishedMessage:
    content: str

@dataclass(frozen=True)
class FinishedThinking:
    content: str

@dataclass(frozen=True)
class FinishedToolCall:
    name: str
    parameters: str

@dataclass(frozen=True)
class FinishedToolCallResult:
    name: str
    parameters: str
    result: str
    is_blocking: bool = False

FinishedElement = Union[FinishedMessage, FinishedThinking, FinishedToolCall, FinishedToolCallResult]

def _to_oai_elements(message_id: int, element: Union[FinishedElement, dict]) -> list[dict[str, Any]]:
    # Internal helper for OAI format mapping
    if isinstance(element, dict):
        # Compatibility for historical messages from DB
        elem_type = element['type']
        if elem_type == 'thinking':
            return []
        if elem_type == 'message':
            return [{'role': element.get('role', 'user'), 'content': element['content']}]
        if elem_type == 'tool_call':
            tool_call = {
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'id': str(message_id),
                    'type': 'function',
                    'function': {
                        'name': element['name'],
                        'arguments': element['parameters'],
                    },
                }]
            }
            res = [tool_call]
            if element.get('status') != 'pending':
                res.append({
                    'role': 'tool',
                    'tool_call_id': str(message_id),
                    'content': str(element.get('result', ''))
                })
            return res
        if elem_type == 'tool_result':
            return [{
                'role': 'tool',
                'tool_call_id': str(element['original_message_id']),
                'content': str(element.get('result', ''))
            }]
        if elem_type == 'tool_decision' and element.get('decision') == 'reject':
            comment = element.get('comment', '')
            if comment:
                content = f"User rejected this tool call with comment: {comment}".strip()
            else:
                content = "User rejected this tool call"
            return [{
                'role': 'tool',
                'tool_call_id': str(element['original_message_id']),
                'content': content
            }]
        return []

    match element:
        case FinishedMessage(content=content):
            return [{'role': 'assistant', 'content': content}]
        case FinishedThinking():
            return []
        case FinishedToolCall(name=name, parameters=params):
            return [{
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'id': str(message_id),
                    'type': 'function',
                    'function': {'name': name, 'arguments': params},
                }]
            }]
        case FinishedToolCallResult(name=name, parameters=params, result=result):
            return [
                {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [{
                        'id': str(message_id),
                        'type': 'function',
                        'function': {'name': name, 'arguments': params},
                    }]
                },
                {
                    'role': 'tool',
                    'tool_call_id': str(message_id),
                    'content': result
                }
            ]
    return []

class ChatContext:
    def __init__(self, messages: List[Dict[str, Any]] = None):
        self._messages = messages or []

    def append_from_db(self, message_id: int, element: dict):
        self._messages.extend(_to_oai_elements(message_id, element))

    def append_finalized(self, message_id: int, element: FinishedElement):
        self._messages.extend(_to_oai_elements(message_id, element))

    def to_list(self) -> List[Dict[str, Any]]:
        return self._messages

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
            new_elem = StreamingToolCall(func.name or "", func.arguments or "")
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
            case (StreamingToolCall(n1, p1), StreamingToolCall(_, p2)):
                self.buffered_element = StreamingToolCall(n1, p1 + p2)
            case _:
                if self.buffered_element:
                    match self.buffered_element:
                        case StreamingMessage(content=c):
                            finalized = FinishedMessage(content=c)
                        case StreamingThinking(content=c):
                            finalized = FinishedThinking(content=c)
                        case StreamingToolCall(name=n1, parameters=p):
                            finalized = FinishedToolCall(name=n1, parameters=p)
                self.buffered_element = new_elem
        return new_elem, finalized

def run_chat_completion_stream(model_id: str, context: ChatContext, functions: List[Any]):
    return completions_endpoint.chat.completions.create(
        model=model_id,
        messages=context.to_list(),
        stream=True,
        tools=functions,
    )

def list_models():
    return completions_endpoint.models.list()
