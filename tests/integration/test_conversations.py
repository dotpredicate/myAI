import unittest
from tests.helpers import BaseTestCase, MockInferenceProvider
from inference.registry import registry

class TestConversations(BaseTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        registry.register(
            "helpful_provider",
            "Helpful Provider",
            "Always helpful",
            MockInferenceProvider(response_content="I am here to help!")
        )
        registry.register(
            "rude_provider",
            "Rude Provider",
            "Always rude",
            MockInferenceProvider(response_content="Go away!")
        )

    async def test_helpful_agent_creation(self):
        payload = {
            "display_name": "Helpful Agent",
            "internal_name": "helpful_agent",
            "description": "A helpful agent",
            "instructions": "Be helpful",
            "provider_key": "helpful_provider",
            "model_id": "model-1",
            "inference_config": {},
            "repository_access": []
        }
        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 201)
        
        list_res = await self.client.get("/api/agents")
        self.assertEqual(list_res.status_code, 200)
        agents = list_res.json()["agents"]
        self.assertTrue(any(a["internal_name"] == "helpful_agent" for a in agents))

    async def test_rude_agent_creation(self):
        payload = {
            "display_name": "Rude Agent",
            "internal_name": "rude_agent",
            "description": "A rude agent",
            "instructions": "Be rude",
            "provider_key": "rude_provider",
            "model_id": "model-2",
            "inference_config": {},
            "repository_access": []
        }
        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 201)

        list_res = await self.client.get("/api/agents")
        self.assertEqual(list_res.status_code, 200)
        agents = list_res.json()["agents"]
        self.assertTrue(any(a["internal_name"] == "rude_agent" for a in agents))

if __name__ == "__main__":
    unittest.main()
