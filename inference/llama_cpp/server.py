import asyncio
from dataclasses import dataclass
import httpx
import openai
from typing import Any, AsyncIterator, Optional

import inference.managed_processes as mp
from inference.engine import ChatContext, EmbeddingInput, EmbeddingProvider, InferenceProvider, Model, StreamingElement, FinishedElement, InferenceParam, InferenceParamType, TokenPiece
from inference.managed_processes import get_free_port
from .gguf import list_cached_models
from .openai import DeltaProcessor, _to_oai_messages, _to_oai_tools
from log_config import get_logger

logger = get_logger(__name__)


def _parse_tokenization_response(payload: object) -> list[TokenPiece]:
    if not isinstance(payload, dict) or not isinstance(payload.get("tokens"), list):
        raise ValueError("Invalid llama.cpp tokenization response")

    tokens: list[TokenPiece] = []
    for raw_token in payload["tokens"]:
        if not isinstance(raw_token, dict):
            raise ValueError("Invalid llama.cpp tokenization response")

        token_id = raw_token.get("id")
        raw_piece = raw_token.get("piece")
        if type(token_id) is not int:
            raise ValueError("Invalid llama.cpp tokenization response")
        piece: str | list[int]
        if isinstance(raw_piece, str):
            piece = raw_piece
        elif isinstance(raw_piece, list) and all(
            type(byte) is int and 0 <= byte <= 255 for byte in raw_piece
        ):
            piece = raw_piece
        else:
            raise ValueError("Invalid llama.cpp tokenization response")

        tokens.append(TokenPiece(id=token_id, piece=piece))
    return tokens


@dataclass
class ModelServer:
    port: str
    client: openai.AsyncOpenAI


class LlamaCppServerProvider(InferenceProvider):

    def __init__(self):
        self._model_servers: dict[tuple[Optional[int], str], ModelServer] = {}
        self._creation_lock = asyncio.Lock()

    async def _get_or_start_server(self, model_id: str, n_ctx: Optional[int]) -> ModelServer:
        key = (n_ctx, model_id)
        if key in self._model_servers:
            server = self._model_servers[key]
            managed = mp.get_managed_process(server.port)
            if managed is not None and managed.process is not None and managed.process.poll() is None:
                await mp.wait_server_ready(server.port)
                return server

        async with self._creation_lock:
            if key in self._model_servers:
                server = self._model_servers[key]
                managed = mp.get_managed_process(server.port)
                if managed is not None and managed.process is not None and managed.process.poll() is None:
                    await mp.wait_server_ready(server.port)
                    return server
                self._model_servers.pop(key, None)

            port = str(get_free_port())
            endpoint = f'http://localhost:{port}'
            client = openai.AsyncOpenAI(api_key='dummy', base_url=endpoint)
            server = ModelServer(port=port, client=client)
            self._model_servers[key] = server

        args = [
            "--jinja",
            "-hf", model_id
        ]
        if n_ctx is not None:
            args += ["--ctx-size", str(n_ctx)]
        cmd = ["llama-server", "--port", port] + args

        async def _check() -> bool:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(f"http://localhost:{port}/v1/models")
                return resp.status_code in (200, 401)

        await mp.ensure_server_started(port, "llama.cpp", model_id, cmd, readiness_check=_check)
        return server

    async def run_chat_completion_stream(
        self,
        model_id: str,
        inference_config: dict[str, Any],
        context: ChatContext,
    ) -> AsyncIterator[tuple[Optional[StreamingElement], Optional[FinishedElement]]]:
        n_ctx: Optional[int] = inference_config.get('n_ctx')
        temperature: Optional[float] = inference_config.get('temperature')
        server = await self._get_or_start_server(model_id, n_ctx)
        async with mp.track_server_activity(server.port):
            raw_stream = await server.client.chat.completions.create(
                model=model_id,
                messages=_to_oai_messages(context),
                reasoning_effort='high',
                temperature=temperature,
                stream=True,
                tools=_to_oai_tools(context.tools),
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "preserve_thinking": True
                    }
                }
            )
            processor = DeltaProcessor()
            async for chunk in raw_stream:
                yield processor.process(chunk)
            finalized = processor.flush()
            if finalized is not None:
                yield None, finalized

    async def list_models(self) -> list[Model]:
        aliases = list_cached_models()
        return [
            Model(id=alias, created=0, owned_by='huggingface')
            for alias in aliases
        ]

    def get_inference_params(self) -> list[InferenceParam]:
        return [
            InferenceParam(
                name="n_ctx",
                type=InferenceParamType.INT,
                default=None,
                min=512,
                max=128000,
                step=512,
                description="Context window size in tokens",
            ),
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

class LlamaCppEmbeddingServer(EmbeddingProvider):
    """Manages a local llama.cpp server process for embeddings."""

    def __init__(self, model: str, port: str = "2345"):
        self.port = port
        self.model = model
        self.endpoint = f'http://localhost:{port}'
        self._client = openai.AsyncOpenAI(api_key='dummy', base_url=self.endpoint + '/v1')

    async def _ensure_started(self):
        async def _check() -> bool:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.post(
                    f"http://localhost:{self.port}/tokenize",
                    json={"content": "Hello, world!"}
                )
                return resp.status_code == 200
        cmd = ["llama-server", "--port", self.port, "--embedding", "-hf", self.model]
        await mp.ensure_server_started(self.port, "llama.cpp", self.model, cmd, readiness_check=_check)

    async def embed(self, model: str, input: EmbeddingInput) -> list[list[float]]:
        if model != self.model:
        # FIXME
            raise ValueError(f"This llama.cpp server only supports {self.model}, but attempted to embed with {model}")
        await self._ensure_started()
        async with mp.track_server_activity(self.port):
            resp = await self._client.embeddings.create(model=model, input=input)
            return [e.embedding for e in resp.data]

    async def tokenize(self, text: str) -> list[TokenPiece]:
        await self._ensure_started()
        async with mp.track_server_activity(self.port):
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.endpoint}/tokenize",
                    json={"content": text, "with_pieces": True}
                )
            return _parse_tokenization_response(response.json())
