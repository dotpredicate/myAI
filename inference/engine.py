from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, AsyncIterator, Optional, TypeAlias, Union

from domain import ConversationElement, ScopeSpec
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

EmbeddingInput: TypeAlias = Union[str, list[str], list[int], list[list[int]]]


@dataclass(frozen=True)
class TokenPiece:
    id: int
    piece: str | list[int]


class EmbeddingProvider(ABC):
    """Interface for producing embeddings and model tokenization details."""

    @abstractmethod
    async def embed(self, model: str, input: EmbeddingInput) -> list[list[float]]:
        ...

    @abstractmethod
    async def tokenize(self, text: str) -> list[TokenPiece]:
        ...


@dataclass(frozen=True)
class ChatContext:
    messages: list[tuple[int, ConversationElement]]
    scopes: list[ScopeSpec]
    tools: list[Tool]
    instructions: Optional[str]


@dataclass(frozen=True)
class Model:
    id: str
    created: int
    owned_by: str


class InferenceParamType(StrEnum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"
    FILE = "file"


@dataclass(frozen=True)
class InferenceParam:
    name: str
    type: InferenceParamType
    default: Any
    min: Optional[Any] = None
    max: Optional[Any] = None
    step: Optional[Any] = None
    description: Optional[str] = None


class InferenceProvider(ABC):
    """Abstract interface for an inference provider.

    Only the two core inference operations are part of the contract.
    Lifecycle management (start/stop server etc.) is implementation-specific.
    """

    @abstractmethod
    def run_chat_completion_stream(
        self,
        model_id: str,
        inference_config: dict[str, Any],
        context: ChatContext,
    ) -> AsyncIterator[tuple[Optional[StreamingElement], Optional[FinishedElement]]]:
        ...

    @abstractmethod
    async def list_models(self) -> list[Model]:
        ...

    def get_inference_params(self) -> list[InferenceParam]:
        return []


@dataclass(frozen=True)
class ImageGenerationRequest:
    model: str
    prompt: str
    params: dict[str, Any]


class ImageProvider(ABC):
    """Abstract interface for a text-to-image provider.

    Lifecycle management (start/stop server etc.) is implementation-specific.
    """

    @abstractmethod
    async def generate_image(self, request: ImageGenerationRequest) -> str:
        """Generate an image and return it base64-encoded."""
        ...

    async def list_models(self) -> list[dict[str, Any]]:
        return []

    async def get_inference_params(self, model_id: str) -> list[InferenceParam]:
        return []

    async def list_recipes(self) -> list[dict[str, Any]]:
        return []

    async def list_files(self) -> list[dict[str, Any]]:
        return []

    async def add_model(self, name: str, recipe: str, files: dict[str, str]) -> dict[str, Any]:
        raise NotImplementedError("this image provider does not support adding models")

    async def remove_model(self, model_id: str) -> bool:
        return False
