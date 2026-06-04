from .engine import (
    InferenceProvider,
    StreamingMessage,
    StreamingThinking,
    StreamingToolCall,
    FinishedMessage,
    FinishedThinking,
    FinishedToolCall,
    FinishedToolCallResult,
    StreamingElement,
    FinishedElement,
)
from .llama_cpp_server import LlamaCppServerProvider, LlamaCppEmbeddingServer
from .openai import DeltaProcessor

# Default provider instance for the built-in llama.cpp server
default_provider: LlamaCppServerProvider = LlamaCppServerProvider()