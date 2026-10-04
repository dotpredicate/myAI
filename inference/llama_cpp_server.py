import asyncio
from contextlib import asynccontextmanager, suppress
import os
import socket
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import httpx
import openai
import time
from typing import Any, AsyncIterator, Optional

from inference.engine import ChatContext, EmbeddingInput, EmbeddingProvider, InferenceProvider, Model, StreamingElement, FinishedElement, InferenceParam, InferenceParamType, TokenPiece
from inference.hf_gguf import list_cached_models
from .openai import DeltaProcessor, _to_oai_messages, _to_oai_tools
from log_config import get_logger

logger = get_logger(__name__)


IDLE_TIMEOUT_SECONDS = 5 * 60
IDLE_CHECK_INTERVAL_SECONDS = 30


@dataclass
class ManagedServer:
    process: subprocess.Popen[Any]
    model: str
    last_used: float
    active_requests: int = 0


_server_processes: dict[str, ManagedServer] = {}
_idle_monitor_task: asyncio.Task[None] | None = None


@asynccontextmanager
async def _track_server_activity(port: str) -> AsyncIterator[None]:
    server = _server_processes[port]
    server.active_requests += 1
    server.last_used = time.monotonic()

    try:
        yield
    finally:
        if _server_processes.get(port) is server:
            server.active_requests -= 1
            server.last_used = time.monotonic()


async def _stop_managed_server(port: str, server: ManagedServer) -> None:
    process = server.process
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


async def _idle_monitor(check_interval: float = IDLE_CHECK_INTERVAL_SECONDS) -> None:
    try:
        while True:
            await asyncio.sleep(check_interval)
            now = time.monotonic()
            for port, server in list(_server_processes.items()):
                if server.process.poll() is not None:
                    _server_processes.pop(port, None)
                elif server.active_requests == 0 and now - server.last_used >= IDLE_TIMEOUT_SECONDS:
                    _server_processes.pop(port, None)
                    await _stop_managed_server(port, server)
                    logger.info("Stopped idle llama-server process on port %s.", port)
    except asyncio.CancelledError:
        raise

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

async def _lazy_start_server(port: str, model: str, args: list[str], start_lock: asyncio.Lock, start_event: asyncio.Event, readiness_check: Callable[[], Awaitable[bool]]) -> None:
    async with start_lock:
        existing = _server_processes.get(port)
        if existing is not None and existing.process.poll() is None:
            await start_event.wait()
            return
        if existing is not None:
            _server_processes.pop(port, None)

        try:
            cmd = ["llama-server", "--port", port] + args
            logger.info("Starting server on port %s: %s", port, ' '.join(cmd))
            proc = subprocess.Popen(cmd)
            _server_processes[port] = ManagedServer(proc, model, time.monotonic())
        except FileNotFoundError:
            logger.error("llama-server not found in PATH.")
            raise

        global _idle_monitor_task
        if _idle_monitor_task is None or _idle_monitor_task.done():
            _idle_monitor_task = asyncio.create_task(_idle_monitor())

        try:
            await _wait_for_server(port, 60, readiness_check)
        except Exception:
            _server_processes.pop(port, None)
            proc.terminate()
            raise
        start_event.set()


def get_llama_server_status() -> list[dict[str, object]]:
    """Return the current status of all llama.cpp servers managed by this process."""
    now = time.monotonic()
    return [
        {
            "port": port,
            "model": server.model,
            "pid": server.process.pid,
            "status": "running" if server.process.poll() is None else "exited",
            "active_requests": server.active_requests,
            "idle_seconds": max(0, int(now - server.last_used)),
        }
        for port, server in _server_processes.items()
    ]


async def stop_llama_server(port: str) -> bool:
    """Stop one managed llama.cpp server. Return False when it is not registered."""
    server = _server_processes.pop(port, None)
    if server is None:
        return False
    await _stop_managed_server(port, server)
    return True

async def stop_llama_servers() -> None:
    """Terminate all managed llama-server processes concurrently.

    Each process gets up to 10 seconds to shut down gracefully (SIGTERM).
    If a process does not terminate within that time, it is killed with SIGKILL.
    """
    global _idle_monitor_task
    if _idle_monitor_task is not None:
        _idle_monitor_task.cancel()
        with suppress(asyncio.CancelledError):
            await _idle_monitor_task
        _idle_monitor_task = None

    servers = list(_server_processes.items())
    _server_processes.clear()
    tasks = [_stop_managed_server(port, server) for port, server in servers]
    await asyncio.gather(*tasks)


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


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
            managed = _server_processes.get(server.port)
            if managed is not None and managed.process.poll() is None:
                await server.ready_event.wait()
                return server

        async with self._creation_lock:
            if key in self._model_servers:
                server = self._model_servers[key]
                managed = _server_processes.get(server.port)
                if managed is not None and managed.process.poll() is None:
                    await server.ready_event.wait()
                    return server
                self._model_servers.pop(key, None)

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

        await _lazy_start_server(port, model_id, args, start_lock, ready_event, readiness_check=_check)
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
        async with _track_server_activity(server.port):
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
        await _lazy_start_server(self.port, self.model, ["--embedding", "-hf", self.model], self.start_lock, self.ready_event, readiness_check=_check)

    async def embed(self, model: str, input: EmbeddingInput) -> list[list[float]]:
        if model != self.model:
        # FIXME
            raise ValueError(f"This llama.cpp server only supports {self.model}, but attempted to embed with {model}")
        await self._lazy_start()
        async with _track_server_activity(self.port):
            resp = await self._client.embeddings.create(model=model, input=input)
            return [e.embedding for e in resp.data]

    async def tokenize(self, text: str) -> list[TokenPiece]:
        await self._lazy_start()
        async with _track_server_activity(self.port):
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.endpoint}/tokenize",
                    json={"content": text, "with_pieces": True}
                )
            return _parse_tokenization_response(response.json())
