import unittest
from tests.helpers import BaseTestCase, MockInferenceProvider
from inference.registry import registry

class TestAgents(BaseTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        registry.register(
            "mock_provider",
            "Mock Provider",
            "A mock provider for testing",
            MockInferenceProvider(response_content="Mock response")
        )

    async def test_create_agent(self):
        payload = {
            "display_name": "Test Agent",
            "internal_name": "test_agent",
            "description": "Testing",
            "instructions": "Be good",
            "provider_key": "mock_provider",
            "model_id": "dummy_model",
            "inference_config": {},
            "repository_access": []
        }
        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 201)
        data = create_res.json()
        self.assertEqual(data["internal_name"], "test_agent")
        self.assertEqual(data["display_name"], "Test Agent")

    async def test_create_agent_missing_fields(self):
        payload = {
            "display_name": "",
            "internal_name": "",
            "description": "",
            "instructions": "",
            "provider_key": "",
            "model_id": "",
            "repository_access": []
        }
        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 400)

    async def test_create_agent_invalid_name(self):
        payload = {
            "display_name": "Bad Agent",
            "internal_name": "bad agent!",
            "description": "",
            "instructions": "",
            "provider_key": "mock_provider",
            "model_id": "dummy_model",
            "repository_access": []
        }
        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 400)

    async def test_create_agent_missing_provider(self):
        payload = {
            "display_name": "No Provider",
            "internal_name": "no_provider",
            "description": "",
            "instructions": "",
            "provider_key": "nonexistent",
            "model_id": "dummy_model",
            "repository_access": []
        }
        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 400)

    async def test_update_agent(self):
        payload = {
            "display_name": "Original",
            "internal_name": "updatable",
            "description": "Original description",
            "instructions": "Original instructions",
            "provider_key": "mock_provider",
            "model_id": "model_1",
            "inference_config": {},
            "repository_access": []
        }
        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 201)

        update_payload = {
            "display_name": "Updated",
            "description": "Updated description",
        }
        update_res = await self.client.put("/api/agents/updatable", json=update_payload)
        self.assertEqual(update_res.status_code, 200)
        data = update_res.json()
        self.assertEqual(data["display_name"], "Updated")
        self.assertEqual(data["description"], "Updated description")

    async def test_delete_agent(self):
        payload = {
            "display_name": "To Delete",
            "internal_name": "to_delete",
            "description": "",
            "instructions": "Delete me",
            "provider_key": "mock_provider",
            "model_id": "dummy_model",
            "inference_config": {},
            "repository_access": []
        }
        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 201)

        delete_res = await self.client.delete("/api/agents/to_delete")
        self.assertEqual(delete_res.status_code, 200)

        get_res = await self.client.get("/api/agents")
        agents = get_res.json()["agents"]
        self.assertFalse(any(a["internal_name"] == "to_delete" for a in agents))

    async def test_get_agents(self):
        get_res = await self.client.get("/api/agents")
        self.assertEqual(get_res.status_code, 200)
        self.assertIn("agents", get_res.json())


if __name__ == "__main__":
    unittest.main()