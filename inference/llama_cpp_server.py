import asyncio
import os
import socket
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import httpx
import openai
from typing import Any, AsyncIterator, Optional

from inference.engine import ChatContext, InferenceProvider, Model, Tool, StreamingElement, FinishedElement, InferenceParam, InferenceParamType
from inference.hf_gguf import resolve_hf_alias, list_cached_models
from .openai import DeltaProcessor, _to_oai_messages, _to_oai_tools
from log_config import get_logger

logger = get_logger(__name__)


_server_processes: dict[str, subprocess.Popen] = {}

async def _wait_for_server(port: str, timeout: int, readiness_check: Callable[[], Awaitable[bool]]) -> None:
    logger.info("Waiting for server on port %s to be ready...", port)

    async def _poll() -> None:
        while True:
            try:
                if await readiness_check():
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(1)

    try:
        await asyncio.wait_for(_poll(), timeout=timeout)
        logger.info("Server on port %s is ready.", port)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Server on port {port} did not start within {timeout} seconds.")

async def _lazy_start_server(port: str, args: list[str], start_lock: asyncio.Lock, start_event: asyncio.Event, readiness_check: Callable[[], Awaitable[bool]]) -> None:
    async with start_lock:
        if port in _server_processes:
            await start_event.wait()
            return
        try:
            cmd = ["llama-server", "--port", port] + args
            logger.info("Starting server on port %s: %s", port, ' '.join(cmd))
            proc = subprocess.Popen(cmd)
            _server_processes[port] = proc
        except FileNotFoundError:
            logger.error("llama-server not found in PATH.")
            raise
        try:
            await _wait_for_server(port, 60, readiness_check)
        except Exception:
            raise
        start_event.set()

async def stop_llama_servers() -> None:
    """Terminate all managed llama-server processes concurrently.

    Each process gets up to 10 seconds to shut down gracefully (SIGTERM).
    If a process does not terminate within that time, it is killed with SIGKILL.
    """
    async def _stop_one(port: str, process: subprocess.Popen) -> None:
        logger.info("Terminating llama-server process %s on port %s", process.pid, port)
        process.terminate()
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, process.wait),
                timeout=10,
            )
            logger.info("Process %s terminated gracefully.", process.pid)
        except asyncio.TimeoutError:
            logger.warning(
                "Process %s did not terminate within 10s, sending SIGKILL.",
                process.pid,
            )
            process.kill()
            await loop.run_in_executor(None, process.wait)
            logger.info("Process %s killed.", process.pid)

    tasks = [
        _stop_one(port, process)
        for port, process in list(_server_processes.items())
    ]
    await asyncio.gather(*tasks)
    _server_processes.clear()


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


@dataclass
class ModelServer:
    port: str
    client: openai.AsyncOpenAI
    ready_event: asyncio.Event
    start_lock: asyncio.Lock


class LlamaCppServerProvider(InferenceProvider):

    def __init__(self):
        self._model_servers: dict[tuple[Optional[int], str], ModelServer] = {}
        self._creation_lock = asyncio.Lock()

    async def _get_or_start_server(self, model_id: str, n_ctx: Optional[int]) -> ModelServer:
        key = (n_ctx, model_id)
        if key in self._model_servers:
            server = self._model_servers[key]
            await server.ready_event.wait()
            return server

        async with self._creation_lock:
            if key in self._model_servers:
                server = self._model_servers[key]
                await server.ready_event.wait()
                return server

            port = str(get_free_port())
            ready_event = asyncio.Event()
            start_lock = asyncio.Lock()
            endpoint = os.getenv('LLAMA_CPP_ENDPOINT', f'http://localhost:{port}')
            client = openai.AsyncOpenAI(api_key='dummy', base_url=endpoint)
            server = ModelServer(port=port, client=client, ready_event=ready_event, start_lock=start_lock)
            self._model_servers[key] = server

        args = [
            "--jinja",
            "-hf", model_id
        ]
        if n_ctx is not None:
            args += ["--ctx-size", str(n_ctx)]

        async def _check() -> bool:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(f"http://localhost:{port}/v1/models")
                return resp.status_code in (200, 401)

        await _lazy_start_server(port, args, start_lock, ready_event, readiness_check=_check)
        return server

    async def run_chat_completion_stream(
        self,
        model_id: str,
        inference_config: dict[str, Any],
        context: ChatContext,
    ) -> AsyncIterator[tuple[Optional[StreamingElement], Optional[FinishedElement]]]:
        n_ctx: int = inference_config.get('n_ctx')
        temperature: float = inference_config.get('temperature')
        server = await self._get_or_start_server(model_id, n_ctx)
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

class LlamaCppEmbeddingServer:
    """Manages a local llama.cpp server process for embeddings."""

    def __init__(self, model: str, port: str = "2345"):
        self.port = port
        self.model = model
        self.endpoint = f'http://localhost:{port}'
        self._client = openai.AsyncOpenAI(api_key='dummy', base_url=self.endpoint + '/v1')
        self.ready_event = asyncio.Event()
        self.start_lock = asyncio.Lock()

    async def _lazy_start(self):
        async def _check() -> bool:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.post(
                    f"http://localhost:{self.port}/tokenize",
                    json={"content": "Hello, world!"}
                )
                return resp.status_code == 200
        await _lazy_start_server(self.port, ["--embedding", "-hf", self.model], self.start_lock, self.ready_event, readiness_check=_check)

    async def embed(self, model: str, input: str | list[str] | list[int] | list[list[int]]) -> list[list[float]]:
        if model != self.model:
        # FIXME
            raise ValueError(f"This llama.cpp server only supports {self.model}, but attempted to embed with {model}")
        await self._lazy_start()
        resp = await self._client.embeddings.create(model=model, input=input)
        return [e.embedding for e in resp.data]

    async def tokenize(self, text: str) -> list[dict[str, object]]:
        await self._lazy_start()
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{self.endpoint}/tokenize",
                json={"content": text, "with_pieces": True}
            )
        return response.json()["tokens"]