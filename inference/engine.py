from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass(frozen=True)
class StreamingMessage:
    content: str

@dataclass(frozen=True)
class StreamingThinking:
    content: str

@dataclass(frozen=True)
class StreamingToolCall:
    name: Optional[str]
    parameters: Optional[str]

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

@dataclass(frozen=True)
class ContextMessage:
    author: Literal['user'] | Literal['assistant']
    content: str

@dataclass(frozen=True)
class ContextThinking:
    content: str

@dataclass(frozen=True)
class ContextToolCall:
    id: str
    name: str
    parameters: str
    status: Literal['pending']

@dataclass(frozen=True)
class ContextToolDecision:
    original_message_id: str
    decision: Literal['accept'] | Literal['reject']
    comment: Optional[str]

@dataclass(frozen=True)
class ContextToolResult:
    original_message_id: str
    result: str

ContextElement = Union[ContextMessage, ContextThinking, ContextToolCall, ContextToolDecision, ContextToolResult]
