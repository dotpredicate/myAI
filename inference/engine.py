from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Optional, TypeAlias, TypedDict, Union

from domain import ConversationElement, ScopeSpec

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


StreamingElement: TypeAlias = Union[StreamingMessage, StreamingThinking, StreamingToolCall]


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


FinishedElement: TypeAlias = Union[FinishedMessage, FinishedThinking, FinishedToolCall, FinishedToolCallResult]

FunctionParameters: TypeAlias = dict[str, object]

class FunctionDefinition(TypedDict):
    name: str
    description: str
    parameters: FunctionParameters

class Tool(TypedDict):
    name: str
    schema: FunctionDefinition
    executor: Callable[[FinishedToolCall, bool, list[ScopeSpec]], Awaitable[FinishedToolCallResult]]


@dataclass(frozen=True)
class Model:
    id: str
    created: int
    owned_by: str

class InferenceProvider(ABC):
    """Abstract interface for an inference provider.

    Only the two core inference operations are part of the contract.
    Lifecycle management (start/stop server etc.) is implementation-specific.
    """

    @abstractmethod
    def run_chat_completion_stream(
        self,
        model_id: str,
        context: list[tuple[int, ConversationElement]],
        functions: list[Tool],
    ) -> AsyncIterator[tuple[Optional[StreamingElement], Optional[FinishedElement]]]:
        ...

    @abstractmethod
    async def list_models(self) -> list[Model]:
        ...