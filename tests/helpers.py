"""Shared test helpers for integration and E2E tests."""

import asyncio
import time
import unittest
import tempfile
import atexit
import httpx
import logging
from typing import AsyncIterator, Optional, Any, List, Tuple

from testcontainers.postgres import PostgresContainer  # type: ignore

import database
from inference.engine import (
    InferenceProvider,
    Model,
    ChatContext,
    StreamingMessage,
    StreamingElement,
    FinishedMessage,
    FinishedElement,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

t0 = time.time()
_container = PostgresContainer("pgvector/pgvector:pg16")
_container.start()
logging.info(f"[SETUP] Postgres container started in {time.time()-t0:.1f}s")

t0 = time.time()
_db_url = _container.get_connection_url(driver=None)
database.DATABASE_URL = _db_url
asyncio.run(database.init_database())
logging.info(f"[SETUP] Database initialized in {time.time()-t0:.1f}s")
atexit.register(_container.stop)


class MockInferenceProvider(InferenceProvider):
    """Simple mock that returns a single static response."""

    def __init__(self, response_content: str = "Mock response", models: Optional[List[Model]] = None):
        self.response_content = response_content
        self.models = models or []

    async def run_chat_completion_stream(
        self,
        model_id: str,
        context: ChatContext,
        functions: list[Any],
    ) -> AsyncIterator[tuple[Optional[StreamingMessage], Optional[FinishedMessage]]]:
        yield (StreamingMessage(content=self.response_content), None)
        yield (None, FinishedMessage(content=self.response_content))

    async def list_models(self) -> list[Model]:
        return self.models


class MockE2EProvider(InferenceProvider):

    def __init__(
        self,
        stream: List[Tuple[Optional[StreamingElement], Optional[FinishedElement]]],
        models: Optional[List[Model]] = None,
    ):
        self.stream = iter(stream)
        self.models = models or []

    async def run_chat_completion_stream(
        self,
        model_id: str,
        context: ChatContext,
        functions: List[Any],
    ) -> AsyncIterator[Tuple[Optional[StreamingElement], Optional[FinishedElement]]]:
        for pair in self.stream:
            yield pair

    async def list_models(self) -> list[Model]:
        return self.models


class BaseTestCase(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        from app import app
        import database
        from inference.registry import registry
        import system

        registry._providers.clear()
        async with database.mk_conn() as conn, conn.cursor() as cur:
            await conn.set_autocommit(False)
            await cur.execute(
                "TRUNCATE agents, agent_repository_access, conversations, "
                "messages, repositories, documents RESTART IDENTITY CASCADE;"
            )
            await conn.commit()

        self._workspace_dir = tempfile.TemporaryDirectory()
        self._repo_dir = tempfile.TemporaryDirectory()
        system.WORKSPACE_DIR = self._workspace_dir.name

        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )

    async def _helper_create_repo(
        self,
        internal_name: str,
        path: str,
        repo_type: str = "plain",
        security: str = "read-only",
    ):
        payload = {
            "display_name": internal_name.replace("_", " ").title(),
            "internal_name": internal_name,
            "path": path,
            "type": repo_type,
            "security": security,
        }
        return await self.client.post("/api/repositories", json=payload)

    async def asyncTearDown(self):
        await self.client.aclose()
        self._workspace_dir.cleanup()
        self._repo_dir.cleanup()