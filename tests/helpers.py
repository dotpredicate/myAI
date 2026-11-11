import asyncio
import unittest
import tempfile
from pathlib import Path
import httpx
import logging
from typing import AsyncIterator, Optional, Any, List, Tuple

from backend import database
from backend import paths
from backend.domain import WorkspaceConfig
from backend.inference.engine import (
    EmbeddingInput,
    EmbeddingProvider,
    ImageGenerationRequest,
    ImageProvider,
    InferenceProvider,
    InferenceParam,
    InferenceParamType,
    Model,
    ChatContext,
    TokenPiece,
    StreamingElement,
    FinishedElement,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


class Yield:
    def __init__(self, streaming: Optional[StreamingElement] = None, finished: Optional[FinishedElement] = None):
        self.streaming = streaming
        self.finished = finished


class Wait:
    def __init__(self):
        self.event = asyncio.Event()

    def set(self):
        self.event.set()

    async def wait(self):
        await self.event.wait()


class MockE2EProvider(InferenceProvider):

    def __init__(self, models: Optional[List[Model]] = None):
        self.models = models or []
        self._streams: List[List[Any]] = []
        self.calls = 0
        self.contexts: List[ChatContext] = []

    def schedule_stream(self, *actions: Any) -> None:
        self._streams.append(list(actions))

    async def run_chat_completion_stream(
        self,
        model_id: str,
        inference_config: dict[str, Any],
        context: ChatContext
    ) -> AsyncIterator[Tuple[Optional[StreamingElement], Optional[FinishedElement]]]:
        actions = self._streams[self.calls] if self.calls < len(self._streams) else []
        self.calls += 1
        self.contexts.append(context)
        for action in actions:
            if isinstance(action, Yield):
                yield action.streaming, action.finished
            elif isinstance(action, Wait):
                await action.event.wait()

    async def list_models(self) -> list[Model]:
        return self.models


class MockImageProvider(ImageProvider):

    model_id = "mock-image"

    def __init__(self) -> None:
        self.requests: list[ImageGenerationRequest] = []
        self.result = "aGVsbG8="
        self.error: Optional[Exception] = None

    async def list_models(self) -> list[dict[str, Any]]:
        return [{"id": self.model_id, "name": "Mock Image", "available": True, "missing": []}]

    async def get_inference_params(self, model_id: str) -> list[InferenceParam]:
        if model_id != self.model_id:
            return []
        return [InferenceParam(name="steps", type=InferenceParamType.INT, default=20)]

    async def generate_image(self, request: ImageGenerationRequest) -> str:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


class DeterministicEmbeddingProvider(EmbeddingProvider):

    def __init__(self) -> None:
        self.embed_calls = 0
        self.tokenize_calls = 0

    async def embed(self, model: str, input: EmbeddingInput) -> list[list[float]]:
        self.embed_calls += 1
        result_count = 1 if isinstance(input, str) else len(input)
        vector = [1.0] + [0.0] * 767
        return [vector.copy() for _ in range(result_count)]

    async def tokenize(self, text: str) -> list[TokenPiece]:
        self.tokenize_calls += 1
        return [TokenPiece(id=index, piece=char) for index, char in enumerate(text)]


class BaseTestCase(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        from backend.app import app
        from backend.inference.providers import reload_providers
        from backend.inference.registry import registry, image_registry

        self._myai_dir = tempfile.TemporaryDirectory()
        paths.use_dir(Path(self._myai_dir.name))
        await database.init_database()

        image_registry._providers.clear()
        await reload_providers()

        self._repo_dir = tempfile.TemporaryDirectory()

        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )

        self.mock_provider = MockE2EProvider()
        registry.register("mock_e2e", "Mock E2E", "", self.mock_provider)

        self.mock_image_provider = MockImageProvider()
        image_registry.register("stable-diffusion.cpp", "Mock Image", "", self.mock_image_provider)

    async def _helper_create_repo(
        self,
        internal_name: str,
        path: str,
        repo_type: str = "plain",
    ):
        payload = {
            "display_name": internal_name.replace("_", " ").title(),
            "internal_name": internal_name,
            "path": path,
            "type": repo_type,
        }
        return await self.client.post("/api/repositories", json=payload)

    async def _helper_create_workspace(self, path: str | None = None) -> WorkspaceConfig:
        response = await self.client.post(
            "/api/workspaces/resolve",
            json={"path": path or self._repo_dir.name},
        )
        if response.status_code != 200:
            raise AssertionError(response.text)
        return WorkspaceConfig.model_validate(response.json()["workspace"])

    async def _helper_repo_source(self, path: str | None = None) -> str:
        """Key of the repository that resolving a path registered as a source."""
        workspace = await self._helper_create_workspace(path)
        return workspace.sources[0].internal_name

    async def _helper_prompt(
        self,
        prompt: str,
        *,
        provider_key: str = "mock_e2e",
        model_id: str = "model-1",
        agent_id: str | None = None,
        path: str | None = None,
    ):
        workspace = await self._helper_create_workspace(path)
        created = await self.client.post(
            "/api/conversations",
            json={"workspace_id": workspace.id, "agent_id": agent_id},
        )
        if created.status_code != 201:
            raise AssertionError(created.text)
        conversation_id = created.json()["id"]
        return await self.client.post(
            f"/api/conversations/{conversation_id}/prompt",
            json={
                "prompt": prompt,
                "provider_key": provider_key,
                "model_id": model_id,
                "inference_config": {},
            },
        )

    async def asyncTearDown(self):
        from backend import search
        task = getattr(search, '_sync_task', None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        await self.client.aclose()
        self._myai_dir.cleanup()
        self._repo_dir.cleanup()
