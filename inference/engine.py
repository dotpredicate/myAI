from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional, TypeAlias, Union

from pydantic import ConfigDict

from domain import ConversationElement
from repositories import ScopeSpec
from tools import Tool

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


FinishedElement: TypeAlias = Union[FinishedMessage, FinishedThinking, FinishedToolCall]

@dataclass(frozen=True)
class ChatContext:
    model_config = ConfigDict(arbitrary_types_allowed=True)
    messages: list[tuple[int, ConversationElement]]
    scopes: list[ScopeSpec]
    tools: list[Tool]
    agent_prompt: Optional[str]


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
        context: ChatContext,
        functions: list[Tool],
    ) -> AsyncIterator[tuple[Optional[StreamingElement], Optional[FinishedElement]]]:
        ...

    @abstractmethod
    async def list_models(self) -> list[Model]:
        ...