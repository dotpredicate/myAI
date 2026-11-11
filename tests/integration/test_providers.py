
from backend import database
from backend.inference.providers import reload_providers
from tests.helpers import BaseTestCase


class TestProviders(BaseTestCase):
    async def test_builtin_provider_lives_only_in_registry(self):
        await reload_providers()

        listing = (await self.client.get("/api/providers")).json()["providers"]
        llama = next(provider for provider in listing if provider["key"] == "llama.cpp")
        self.assertEqual(llama["kind"], "builtin")
        self.assertFalse(llama["has_api_key"])

        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute("SELECT count(*) FROM inference_providers WHERE key = 'llama.cpp'")
            count = await cur.fetchone()
        self.assertEqual(count[0], 0)

        response = await self.client.get("/api/providers/llama.cpp")
        self.assertEqual(response.status_code, 404)

        response = await self.client.delete("/api/providers/llama.cpp")
        self.assertEqual(response.status_code, 404)

    async def test_provider_crud_and_redaction(self):
        response = await self.client.post("/api/providers", json={
            "key": "remote",
            "display_name": "Remote",
            "description": "Remote API",
            "base_url": "https://example.com/v1/",
            "api_key": "secret",
            "headers": {"Authorization": "Bearer secret"},
            "models": ["model-a"],
            "timeout_seconds": 30,
        })
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()
        self.assertEqual(payload["key"], "remote")
        self.assertEqual(payload["base_url"], "https://example.com/v1")
        self.assertTrue(payload["has_api_key"])
        self.assertNotIn("api_key", payload)
        self.assertNotIn("headers", payload)

        listing = (await self.client.get("/api/providers")).json()["providers"]
        remote = next(provider for provider in listing if provider["key"] == "remote")
        self.assertTrue(remote["has_api_key"])
        self.assertNotIn("api_key", remote)
        self.assertEqual(remote["kind"], "openai_compatible")

        detail = await self.client.get("/api/providers/remote")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["base_url"], "https://example.com/v1")
        self.assertNotIn("api_key", detail.json())
        self.assertNotIn("headers", detail.json())

        response = await self.client.patch("/api/providers/remote", json={
            "display_name": "Remote API",
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["has_api_key"])

        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute("SELECT headers FROM inference_providers WHERE key = 'remote'")
            row = await cur.fetchone()
        self.assertEqual(row[0], {"Authorization": "Bearer secret"})

        response = await self.client.patch("/api/providers/remote", json={
            "description": None,
            "api_key": None,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["has_api_key"])
        self.assertEqual(response.json()["display_name"], "Remote API")
        self.assertEqual(response.json()["description"], "")

        response = await self.client.patch("/api/providers/remote", json={
            "base_urls": "https://example.com/v1",
        })
        self.assertEqual(response.status_code, 422)

        response = await self.client.patch("/api/providers/remote", json={
            "api_key": "",
        })
        self.assertEqual(response.status_code, 422)

        response = await self.client.delete("/api/providers/remote")
        self.assertEqual(response.status_code, 200)
        listing = (await self.client.get("/api/providers")).json()["providers"]
        self.assertNotIn("remote", {provider["key"] for provider in listing})

        response = await self.client.delete("/api/providers/remote")
        self.assertEqual(response.status_code, 404)

    async def test_provider_validation(self):
        response = await self.client.post("/api/providers", json={
            "key": "bad key",
            "display_name": "Bad",
            "base_url": "https://example.com/v1",
        })
        self.assertEqual(response.status_code, 400)

        response = await self.client.post("/api/providers", json={
            "key": "remote",
            "display_name": "",
            "base_url": "https://example.com/v1",
        })
        self.assertEqual(response.status_code, 422)

        response = await self.client.post("/api/providers", json={
            "key": "remote",
            "display_name": "Remote",
            "base_url": "ftp://example.com",
        })
        self.assertEqual(response.status_code, 422)

        response = await self.client.post("/api/providers", json={
            "key": "llama.cpp",
            "display_name": "Built-in",
            "base_url": "https://example.com/v1",
        })
        self.assertEqual(response.status_code, 409)

        response = await self.client.post("/api/providers", json={
            "key": "mock_e2e",
            "display_name": "Duplicate",
            "base_url": "https://example.com/v1",
        })
        self.assertEqual(response.status_code, 409)

        response = await self.client.post("/api/providers", json={
            "key": "headers",
            "display_name": "Headers",
            "base_url": "https://example.com/v1",
            "headers": {"X-Test": "a\nb"},
        })
        self.assertEqual(response.status_code, 422)

        response = await self.client.post("/api/providers", json={
            "key": "empty_key",
            "display_name": "Empty key",
            "base_url": "https://example.com/v1",
            "api_key": "",
        })
        self.assertEqual(response.status_code, 422)

    async def test_cannot_delete_provider_used_by_agent(self):
        response = await self.client.post("/api/providers", json={
            "key": "remote",
            "display_name": "Remote",
            "base_url": "https://example.com/v1",
        })
        self.assertEqual(response.status_code, 201, response.text)

        response = await self.client.post("/api/agents", json={
            "display_name": "Remote Agent",
            "internal_name": "remote_agent",
            "description": "",
            "instructions": "",
            "provider_key": "remote",
            "model_id": "model-a",
            "inference_config": {},
            "repository_access": [],
        })
        self.assertEqual(response.status_code, 201, response.text)

        response = await self.client.delete("/api/providers/remote")
        self.assertEqual(response.status_code, 409)

    async def test_agent_can_use_external_provider(self):
        response = await self.client.post("/api/providers", json={
            "key": "remote",
            "display_name": "Remote",
            "base_url": "https://example.com/v1",
            "models": ["model-a"],
        })
        self.assertEqual(response.status_code, 201, response.text)

        response = await self.client.post("/api/agents", json={
            "display_name": "Remote Agent",
            "internal_name": "remote_agent",
            "description": "",
            "instructions": "",
            "provider_key": "remote",
            "model_id": "model-a",
            "inference_config": {},
            "repository_access": [],
        })
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["provider_key"], "remote")
