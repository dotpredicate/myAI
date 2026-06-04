import asyncio
import os
import subprocess
import httpx
import openai
from typing import AsyncIterator, Optional

from domain import ConversationElement
from inference.engine import InferenceProvider, Model, Tool, StreamingElement, FinishedElement
from .openai import DeltaProcessor, _to_oai_messages, _to_oai_tools
from log_config import get_logger

logger = get_logger(__name__)


_server_processes: dict[str, subprocess.Popen] = {}

async def _wait_for_server(port: str, timeout: int) -> None:
    url = f"http://localhost:{port}/v1/models"
    logger.info("Waiting for server on port %s to be ready...", port)

    async def _poll() -> None:
        async with httpx.AsyncClient(timeout=1.0) as client:
            while True:
                try:
                    response = await client.get(url)
                    if response.status_code in (200, 401):
                        return
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(1)

    try:
        await asyncio.wait_for(_poll(), timeout=timeout)
        logger.info("Server on port %s is ready.", port)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Server on port {port} did not start within {timeout} seconds.")

async def _lazy_start_server(port: str, args: list[str]) -> None:
    if port in _server_processes:
        return
    try:
        cmd = ["llama-server", "--port", port] + args
        logger.info("Starting server on port %s: %s", port, ' '.join(cmd))
        proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
        _server_processes[port] = proc
    except FileNotFoundError:
        logger.error("llama-server not found in PATH.")
        raise
    await _wait_for_server(port, 60)

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

    def __init__(self, port: str = "1234", embedding_port: str = "2345"):
        self.port = port
        self.embedding_port = embedding_port
        endpoint = os.getenv('LLAMA_CPP_ENDPOINT', f'http://localhost:{port}')
        self._client = openai.Client(api_key='dummy', base_url=endpoint)

    async def _lazy_start(self):
        await _lazy_start_server(self.port, ["--offline", "--jinja", "--chat-template-kwargs", '{"preserve_thinking":true}'])

    async def run_chat_completion_stream(
        self,
        model_id: str,
        context: list[tuple[int, ConversationElement]],
        tools: list[Tool],
    ) -> AsyncIterator[tuple[Optional[StreamingElement], Optional[FinishedElement]]]:
        await self._lazy_start()
        raw_stream = self._client.chat.completions.create(  # type: ignore[call-overload]
            model=model_id,
            messages=_to_oai_messages(context),
            reasoning_effort='high',
            stream=True,
            tools=_to_oai_tools(tools),
        )
        processor = DeltaProcessor()
        for chunk in raw_stream:
            yield processor.process(chunk)

    async def list_models(self) -> list[Model]:
        await self._lazy_start()
        raw_models = self._client.models.list()
        return [
            Model(id=m.id, created=m.created, owned_by=m.owned_by)
            for m in raw_models
        ]

class LlamaCppEmbeddingServer:
    """Manages a local llama.cpp server process for embeddings."""

    def __init__(self, model: str, port: str = "2345"):
        self.port = port
        self.model = model
        self.endpoint = f'http://localhost:{port}'
        self._client = openai.Client(api_key='dummy', base_url=self.endpoint + '/v1')

    async def _lazy_start(self):
        await _lazy_start_server(self.port, ["--embedding", "-hf", self.model])

    async def embed(self, model: str, input: str | list[str] | list[int] | list[list[int]]) -> list[list[float]]:
        if model != self.model:
        # FIXME
            raise ValueError(f"This llama.cpp server only supports {self.model}, wanted to embed with {model}")
        await self._lazy_start()
        resp = self._client.embeddings.create(model=model, input=input)
        return [e.embedding for e in resp.data]

    async def tokenize(self, text: str) -> list[dict[str, object]]:
        await self._lazy_start()
        # HACK: The first request exceeds the default timeout because the model is being loaded
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.endpoint}/tokenize",
                json={"content": text, "with_pieces": True}
            )
        return response.json()["tokens"]