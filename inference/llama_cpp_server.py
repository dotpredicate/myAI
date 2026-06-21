import asyncio
import os
import subprocess
from collections.abc import Awaitable, Callable
import httpx
import openai
from typing import AsyncIterator, Optional

from inference.engine import ChatContext, InferenceProvider, Model, Tool, StreamingElement, FinishedElement
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
                "Process %s did not terminate within 20s, sending SIGKILL.",
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

class LlamaCppServerProvider(InferenceProvider):
    """Inference provider backed by a local llama.cpp server process."""

    def __init__(self, port: str = "1234",):
        self.port = port
        endpoint = os.getenv('LLAMA_CPP_ENDPOINT', f'http://localhost:{port}')
        self._client = openai.AsyncOpenAI(api_key='dummy', base_url=endpoint)
        self.ready_event = asyncio.Event()
        self.start_lock = asyncio.Lock()

    async def _lazy_start(self):
        async def _check() -> bool:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(f"http://localhost:{self.port}/v1/models")
                return resp.status_code in (200, 401)
        await _lazy_start_server(self.port, ["--offline", "--jinja", "--chat-template-kwargs", '{"preserve_thinking":true}'], self.start_lock, self.ready_event, readiness_check=_check)

    async def run_chat_completion_stream(
        self,
        model_id: str,
        context: ChatContext,
        tools: list[Tool],
    ) -> AsyncIterator[tuple[Optional[StreamingElement], Optional[FinishedElement]]]:
        await self._lazy_start()
        raw_stream = await self._client.chat.completions.create(
            model=model_id,
            messages=_to_oai_messages(context),
            reasoning_effort='high',
            stream=True,
            tools=_to_oai_tools(tools),
        )
        processor = DeltaProcessor()
        async for chunk in raw_stream:
            yield processor.process(chunk)

    async def list_models(self) -> list[Model]:
        await self._lazy_start()
        raw_models = await self._client.models.list()
        return [
            Model(id=m.id, created=m.created, owned_by=m.owned_by)
            for m in raw_models.data
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
            raise ValueError(f"This llama.cpp server only supports {self.model}, wanted to embed with {model}")
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